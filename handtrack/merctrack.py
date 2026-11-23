"""Continuous per-hand tracking with Monado Mercury's keypoint model (opt-in: --backend mercury).

MediaPipe stays the DISCOVERY path (finds hands, initial 21-joint 3D).  Once a hand has a 3D
joint state, this tracker re-observes it every cycle with one Mercury crop per hand (cameras
alternate), using the hand's own previous joints as ROI and temporal hint.  A single view gives a
full articulated 3D hand: Mercury predicts each joint's ray AND its depth relative to the middle
knuckle in hand-size units; absolute depth is anchored to the 60 Hz stereo palm position from the
optical-flow tracker when available, else to the previous state.  When the other camera's result
for the same hand is fresh (< STEREO_MS), joints are triangulated instead.

Views per hand (views): "both" crops the hand from both cameras of the same frame (at most
max_crops crops per job (MAX_CROPS = 3, or 4 with --merc-crops 4 on a 4-crop NPU image: both
hands in both views); with two hands and three crops the second view takes turns), so stereo joints come from
one instant.  "alternate" (the original) crops one camera per frame and triangulates against the
other camera's previous result, captured a frame earlier: a moving hand has moved between the two
views, which puts a triangulated palm 2-7 cm off at 0.3-0.6 m/s (more at distance) and tilts the
palm when the wrist turns.  "auto" = both when inference is on the NPU, alternate on the CPU.
"stereo3d" (handtrack/stereo3d.py): both cameras of the same pair for every hand, FeatureNN-style --
the keypoint model on both crops (same-instant triangulation) plus a 3D stereo student on the pair;
the 3D hand shape is placed on the triangulated palm centre.
With stereo_comp the older view's rays are shifted by the palm's motion since its capture
(from the 60 Hz two-camera palm positions), which removes the translation part of that error.

The Mercury model license is UNVERIFIED (local evaluation only).
"""
import time

import numpy as np

from handtrack import stereo3d
from handtrack.calib import triangulate
from handtrack.palm import PALM_IDX, finger_curls

STEREO_MS = 40.0          # other view's rays this fresh -> triangulate joints
COLLAPSE_M = 0.07         # two tracked hands closer than this (palm centres) = one physical hand
CURL_SCALE = (2.0, 4.0, 4.0, 4.0, 4.0)   # Mercury curl output (negative = flexed) -> 0..1, thumb first
                                         # (provisional mapping, not calibrated against ground truth)
MAX_CROPS = 3             # crops per Mercury job (one NPU request)
LOST_AFTER_S = 0.35       # no accepted Mercury observation for this long -> back to discovery


def view_joints(P, hand_size, origin, rays, rel_depth):
    """One camera view's joints: its rays at Mercury's relative depths, with the middle knuckle at the
    distance it has in the accepted hand P (head frame)."""
    d9 = float(np.linalg.norm(P[9] - origin))
    return origin + rays * (d9 + rel_depth * hand_size)[:, None]


class HandState:
    def __init__(self, side, P, t):
        self.side = side
        self.P = np.asarray(P, float)             # (21,3) head frame
        self.t = t
        self.hand_size = float(np.linalg.norm(self.P[0] - self.P[9]))
        self.next_slot = 1
        self.last_submit = 0.0                    # monotonic s of the last job that included this hand
        self.vel = np.zeros(3)                    # palm velocity (m/s) from the last two accepted observations
        self.last_rays = {}                       # slot -> (t, origin, rays head frame, palm position at t or None)
        self.last_both = 0.0                      # monotonic s of the last job with both views of this hand
        self.misses = 0
        self.created = t
        self.curls_by_view = {}                   # slot -> (t, mercury curls (5,), variances (5,))
        self.geo_views = {}                       # slot -> (t, origin, rays head frame, Mercury relative depths)


class MercuryTracker:
    def __init__(self, palm, pool, worker):
        self.palm = palm                          # PalmTracker: pose[slot], _pose(), curl_filt
        self.pool = pool                          # dedicated ProcessPoolExecutor(1)
        self.worker = worker                      # module with run_mercury(jobs, imgs)
        self.ring = None                          # framering.FrameRing: raw frames via shared memory (--frame-ring)
        self.hands = {}                           # side -> HandState
        self.job = None
        self.job_info = {}                        # submit path / capture seq / stage times of self.job
        self.idle_future = None
        self.curl_source = "mercury"
        self.predict_roi = False
        self.stereo_ms = STEREO_MS                # max capture-time gap for triangulating two views (0 = same capture only)
        self.curl_latest = False                  # True: curls from this observation only (no blend with older views)
        self.curl_geo = []                        # finger indices (0 thumb .. 4 pinky) whose curl comes from each
                                                  # view's own single-view joints instead (view_joints)
        self.hint_reset = False                   # True: no keypoint hint for a crop placed after a miss / lost flow
        self.sched = "batch"                      # "batch": all hands in one job (original);
                                                  # "perhand": one hand per job on the newest frame,
                                                  # least recently serviced hand first
        self.max_crops = MAX_CROPS                # crops a job plans (4: --merc-crops 4; the worker
                                                  # splits a job to what its backend takes)
        self.views = "auto"                       # "both" | "alternate" | "auto" | "stereo3d" (module docstring)
        self.s3d_curls = "geometric"              # stereo3d: curls from the joints ("geometric") or the keypoint model ("mercury")
        self.job_kind = None                      # "s3d" while a stereo3d job is in flight
        # Mercury-detector discovery waiting to ride in the next tracking job (live.py sets it:
        # {future, pick, ncrop, det_size}; submit adds t_cap, det_imgs and the tracked palms, and
        # collect resolves the future with run_discover's result)
        self.discover_req = None
        self.job_disc = None
        self.last_submit_t = None                 # time.monotonic() of the last submitted tracking job
        self.stereo_comp = True                   # shift an older view's rays by the palm motion since its capture
        self.last_inference = "cpu"               # backend of the last result (views = auto)
        self.collapsed = None
        self.next_id = 0
        self.records = []                         # per-observation timing records, drained by the caller
        self.stats = {"runs": 0, "accepted": 0, "rejected": 0, "stereo": 0, "mono": 0, "ms": []}

    def tracked(self):
        return set(self.hands)

    @staticmethod
    def palm_c(P):
        return np.asarray(P)[[0, 5, 9, 17]].mean(0)

    def seed(self, side, P, t):
        """New 3D joints from MediaPipe discovery: start (or re-anchor) tracking this hand.
        Never start a hand on top of the other tracked hand."""
        if P is None:
            return
        other = self.hands.get("right" if side == "left" else "left")
        if other is not None and np.linalg.norm(self.palm_c(P) - self.palm_c(other.P)) < COLLAPSE_M:
            self.stats["seed_blocked"] = self.stats.get("seed_blocked", 0) + 1
            return
        hs = self.hands.get(side)
        if hs is None or t - hs.t > LOST_AFTER_S:
            self.hands[side] = HandState(side, P, t)

    def busy(self):
        return self.job is not None

    def expire(self, t):
        """Drop hands with no accepted observation for LOST_AFTER_S at capture time t (s), as collect
        does. For a hand no crop can be placed for (behind / too close to both cameras), no job
        comes back to drop it, so it would stay tracked indefinitely. Returns the dropped sides."""
        gone = [sd for sd, hs in self.hands.items() if t - hs.t > LOST_AFTER_S]
        for sd in gone:
            del self.hands[sd]
        return gone

    def idle(self):
        """Queue one power release after the current observations finish."""
        if self.job is None and self.idle_future is None:
            self.idle_future = self.pool.submit(self.worker.idle_mercury)

    def submit(self, imgs, t_cap_ns, fast_pos, anchor_pos=None, info=None, raw_imgs=None):
        """One Mercury crop per tracked hand on this frame (alternating cameras); in "perhand" mode
        only the least recently serviced hand.  fast_pos places the crop (its image location is right
        even when its depth is held); anchor_pos (default fast_pos) is the depth anchor for single-view
        joints and should only contain trustworthy (two-camera) positions.  info (path, seq, t_read,
        t_ready) is copied into this job's telemetry records.  raw_imgs (the published camera frames):
        Mercury crops from these at full resolution; imgs (detector images) still seed the flow."""
        if self.job is not None or not self.hands:
            return False
        if self.views == "stereo3d":
            return self._submit_s3d(imgs, t_cap_ns, fast_pos, anchor_pos, info, raw_imgs)
        jobs = []
        sides = list(self.hands)
        if self.sched == "perhand":
            sides = [min(sides, key=lambda sd: self.hands[sd].last_submit)]
        t_sub = time.monotonic()
        both = self.views == "both" or (self.views == "auto" and self.last_inference != "cpu")
        # second views take turns between hands: the hand that had both views longest ago goes first
        spare = self.max_crops - len(sides) if both else 0
        second = set(sorted(sides, key=lambda sd: self.hands[sd].last_both)[:max(0, spare)])
        for side in sides:
            hs = self.hands[side]
            hs.last_submit = t_sub
            P = hs.P
            fp = fast_pos.get(side)
            expand = None
            if fp is not None:                    # move the ROI with the 60 Hz palm position
                P = P + (np.asarray(fp[0]) - P[PALM_IDX[:1] + [5, 9, 17]].mean(0))
            elif self.predict_roi:
                # fast tracker lost the hand (typically a quick slap): put the crop where the hand is
                # heading, not where it was (bounded extrapolation), and look wider
                dt = min(max(t_cap_ns / 1e9 - hs.t, 0.0), 0.15)
                P = P + hs.vel * dt
            if self.predict_roi:
                spd = float(np.linalg.norm(hs.vel))
                if fp is None or spd > 1.0:
                    expand = 1.7 * (1.35 if fp is None or spd > 2.5 else 1.15)
            slot = hs.next_slot
            hs.next_slot = 1 - slot
            R, t = self.palm.pose[slot]
            jc = (P - t) @ R
            if (jc[:, 2] < 0.08).any():          # behind / too close to this camera: try the other one
                slot = 1 - slot
                R, t = self.palm.pose[slot]
                jc = (P - t) @ R
                if (jc[:, 2] < 0.08).any():
                    continue
            # a crop after a rejected observation or without a fresh flow position is a recovery crop:
            # the previous keypoints describe a hand that is no longer there, so do not feed them as a hint
            hint = not (self.hint_reset and (hs.misses > 0 or fp is None))
            if side in second:                    # the other camera of the same frame too
                R2, t2 = self.palm.pose[1 - slot]
                jc2 = (P - t2) @ R2
                if not (jc2[:, 2] < 0.08).any():
                    jobs.append((1 - slot, side, jc2, expand, hint))
                    hs.last_both = t_sub
            jobs.append((slot, side, jc, expand, hint))
        if not jobs:
            return False
        self.idle_future = None
        used = sorted({j[0] for j in jobs})
        disc, self.job_disc = self.discover_req, None
        if disc is not None and raw_imgs is not None:
            self.discover_req = None
            disc.update(t_cap=t_cap_ns, det_imgs=dict(imgs) if imgs is not None else None)
            req = {k: disc[k] for k in ("pick", "ncrop", "det_size")}
            req["tracked"] = {sd: self.palm_c(hs.P) for sd, hs in self.hands.items()}
            ref = self.ring.put(raw_imgs) if self.ring is not None else None
            fut = self.pool.submit(self.worker.run_mercury, jobs,
                                   ref if ref is not None else {s: raw_imgs[s] for s in (0, 1)}, True, req)
            if ref is not None:
                self.ring.release_when_done(ref, fut)
            self.job_disc = disc
        elif raw_imgs is not None:
            ref = self.ring.put(raw_imgs) if self.ring is not None else None
            fut = self.pool.submit(self.worker.run_mercury, jobs,
                                   ref if ref is not None else {s: raw_imgs[s] for s in used}, True)
            if ref is not None:
                self.ring.release_when_done(ref, fut)
        else:
            fut = self.pool.submit(self.worker.run_mercury, jobs, {s: imgs[s] for s in used})
        self.job = (fut, t_cap_ns, fast_pos,
                    {s: imgs[s] for s in used} if imgs is not None else None, time.monotonic_ns(), len(jobs))
        if anchor_pos is not None:
            self.job = self.job[:2] + (anchor_pos,) + self.job[3:]
        self.job_used = used
        self.job_info = dict(info or {})
        self.last_submit_t = t_sub
        return True

    def pending(self):
        """A job submitted with imgs=None (live.py --merc-preflow, before its capture's detector images and
        flow positions existed) that complete() has not filled in yet."""
        return self.job is not None and self.job_kind != "s3d" and self.job[3] is None

    def complete(self, imgs, anchor_pos, t_ready=None):
        """--merc-preflow: the submitted capture's detector images and flow-position depth anchor, now that
        they exist; collect() waits for this, so the job is applied exactly as one submitted after flow."""
        if not self.pending():
            return False
        self.job = self.job[:2] + (anchor_pos,) + ({s: imgs[s] for s in self.job_used},) + self.job[4:]
        if self.job_disc is not None and self.job_disc.get("det_imgs") is None:
            self.job_disc["det_imgs"] = dict(imgs)
        if t_ready is not None:
            self.job_info["t_ready"] = t_ready
        return True

    def collect(self):
        """If the in-flight job finished: update hand states.
        Returns {side: (pose tuple, slot, det_px, t_capture_s, image_the_crop_came_from)}."""
        if self.job is None or not self.job[0].done() or self.pending():
            return {}
        if self.job_kind == "s3d":
            return self._collect_s3d()
        fut, t_ns, fast_pos, job_imgs, t_sub_ns, njob = self.job
        info = self.job_info
        self.job, self.job_info = None, {}
        disc, self.job_disc = self.job_disc, None
        t = t_ns / 1e9
        t_col_ns = time.monotonic_ns()
        try:
            res = fut.result()
        except Exception as exc:
            if disc is not None:
                disc["future"].set_exception(exc)
            return {}
        if disc is not None:
            disc["future"].set_result(res["discover"])
            res = res["obs"]
        out = {}
        def valid(r):
            return r is not None and r["is_hand"] >= 0.97
        # a valid result followed by another valid one for the same hand in this job (both views of
        # one frame): keep its rays for the later one to triangulate with, and pose the hand once
        later = [any(rs[1] == side and valid(rs[2]) for rs in res[i + 1:]) for i, (_, side, *_) in enumerate(res)]
        for (slot, side, r, dsec, *tim), paired in zip(res, later):
            self.stats["runs"] += 1
            self.stats["ms"].append(dsec * 1e3)
            tim = tim[0] if tim else {}
            rec = {"id": self.next_id, "side": side, "slot": slot, "t_cap": t_ns, "t_sub": t_sub_ns,
                   "t0": tim.get("t0", 0), "t1": tim.get("t1", 0), "cpu_ns": tim.get("cpu_ns", 0),
                   "wait_ns": tim.get("wait_ns", 0), "t_col": t_col_ns, "njob": njob, "outcome": "rej", "rq": None,
                   "hint": int(tim.get("hint", True)), "inference": tim.get("inference", "unknown"),
                   "request_crops": tim.get("request_crops", 0), "dsp_body_ms": tim.get("dsp_body_ms", 0.0),
                   "path": info.get("path", "frame"), "seq": info.get("seq", 0),
                   "t_read": info.get("t_read", 0), "t_ready": info.get("t_ready", 0)}
            self.next_id += 1
            self.records.append(rec)
            hs = self.hands.get(side)
            if hs is None:
                rec["outcome"] = "gone"
                continue
            self.last_inference = rec["inference"]
            if not valid(r):
                self.stats["rejected"] += 1
                hs.misses += 1
                continue
            R, o = self.palm.pose[slot]
            rays = r["rays"] @ R.T                       # head frame
            fp_now = fast_pos.get(side)
            fp_now = np.asarray(fp_now[0], float) if fp_now is not None else None
            hs.last_rays[slot] = (t, o, rays, fp_now)
            hs.geo_views[slot] = (t, o, rays, np.asarray(r["rel_depth"], float))
            if paired:
                hs.curls_by_view[slot] = (t, np.asarray(r["curls"], float), np.asarray(r["curl_var"], float))
                rec["outcome"] = "pair"
                continue
            other = hs.last_rays.get(1 - slot)
            P = None
            if other is not None and abs(t - other[0]) * 1e3 <= self.stereo_ms and \
                    (self.stereo_ms > 0 or t == other[0]):
                (to, oo, ro, fpo) = other
                if self.stereo_comp and t != to:
                    # the hand moved since the other view was captured: move that view with it
                    move = fp_now - fpo if fp_now is not None and fpo is not None else hs.vel * (t - to)
                    if np.linalg.norm(move) < 0.1:
                        oo = oo + move
                    rec["comp_mm"] = float(np.linalg.norm(move)) * 1e3
                Pm, gap, s1, s2 = triangulate(np.broadcast_to(o, rays.shape), rays, np.broadcast_to(oo, ro.shape), ro)
                if np.median(gap) < 0.02 and (s1 > 0.05).all() and (s2 > 0.05).all():
                    P = Pm
                    hs.hand_size = 0.9 * hs.hand_size + 0.1 * float(np.linalg.norm(P[0] - P[9]))
                    self.stats["stereo"] += 1
                    rec["outcome"] = "stereo"
            if P is None:
                # single view: rays + Mercury's relative depths; absolute depth of the middle knuckle from
                # the 60 Hz stereo palm position if fresh, else from the previous state
                fp = fast_pos.get(side)
                anchor = np.asarray(fp[0]) if fp is not None else hs.P[[0, 5, 9, 17]].mean(0)
                prevc = hs.P[[0, 5, 9, 17]].mean(0)
                d9 = np.linalg.norm(hs.P[9] + (anchor - prevc) - o)
                dist = d9 + r["rel_depth"] * hs.hand_size
                P = o + rays * dist[:, None]
                # re-anchor so the palm centre sits on the stereo palm position
                if fp is not None:
                    P = P + (anchor - P[[0, 5, 9, 17]].mean(0))
                self.stats["mono"] += 1
                rec["outcome"] = "mono"
            jump = np.linalg.norm(P[[0, 5, 9, 17]].mean(0) - hs.P[[0, 5, 9, 17]].mean(0))
            if jump > 0.25 and t - hs.t < 0.2:         # implausible: keep the old state, count a miss
                self.stats["rejected"] += 1
                hs.misses += 1
                rec["outcome"] = "jump"
                continue
            dtv = t - hs.t
            if 0.005 < dtv < 0.2:
                v = (self.palm_c(P) - self.palm_c(hs.P)) / dtv
                sp = np.linalg.norm(v)
                hs.vel = v * min(1.0, 6.0 / sp) if sp > 0 else v      # bounded: 6 m/s
            else:
                hs.vel = np.zeros(3)
            hs.P, hs.t, hs.misses = P, t, 0
            self.stats["accepted"] += 1
            rec["rq"] = self.palm.raw_quat(side, P)
            hs.curls_by_view[slot] = (t, np.asarray(r["curls"], float), np.asarray(r["curl_var"], float))
            if self.curl_source == "mercury":
                # Mercury's own curl output (consistent across views): variance-weighted mean of fresh views
                vs = [(c, v) for (tc, c, v) in hs.curls_by_view.values() if t - tc < 0.2] if not self.curl_latest \
                    else [hs.curls_by_view[slot][1:]]
                wsum = sum(1 / v for _, v in vs)
                mc = sum(c / v for c, v in vs) / wsum
                raw_curls = np.clip(-mc / np.array(CURL_SCALE), 0, 1)
            else:
                raw_curls = finger_curls(P)
            if self.curl_geo:
                # single-view joints of each fresh view: unlike triangulated joints, whose depth noise
                # inflates bone angles (open hands read curled), these keep Mercury's own finger depths
                gv = [hs.geo_views[slot]] if self.curl_latest else \
                    [v for v in hs.geo_views.values() if t - v[0] < 0.2]
                g = np.mean([finger_curls(view_joints(P, hs.hand_size, *v[1:])) for v in gv], 0)
                raw_curls = np.array(raw_curls, float)
                raw_curls[self.curl_geo] = g[self.curl_geo]
            curls = self.palm.curl_filt[side](raw_curls, t)
            pose = self.palm._pose(side, P, t, conf=r["is_hand"]) + (curls,)
            out[side] = (pose, slot, r["det_px"], t, job_imgs[slot])
        for side in list(self.hands):
            if t - self.hands[side].t > LOST_AFTER_S:
                del self.hands[side]
        # collapse guard: both trackers on the same physical hand -> drop the newer track, re-discover
        self.collapsed = None
        if len(self.hands) == 2:
            L, R = self.hands["left"], self.hands["right"]
            if np.linalg.norm(self.palm_c(L.P) - self.palm_c(R.P)) < COLLAPSE_M:
                drop = "left" if L.created > R.created else "right"
                del self.hands[drop]
                out.pop(drop, None)
                self.collapsed = drop
                self.stats["collapses"] = self.stats.get("collapses", 0) + 1
        return out

    # ---- views = "stereo3d" (handtrack/stereo3d.py) -------------------------------------------------
    def _submit_s3d(self, imgs, t_cap_ns, fast_pos, anchor_pos, info, raw_imgs):
        """both cameras' crops of every tracked hand from THIS pair (the cameras expose together);
        placement = the hand's joints moved with the 60 Hz palm position, as in submit()."""
        sides = list(self.hands)
        if self.sched == "perhand":
            sides = [min(sides, key=lambda sd: self.hands[sd].last_submit)]
        t_sub = time.monotonic()
        jobs = []
        for side in sides:
            hs = self.hands[side]
            hs.last_submit = t_sub
            P = hs.P
            fp = fast_pos.get(side)
            if fp is not None:
                P = P + (np.asarray(fp[0]) - P[PALM_IDX[:1] + [5, 9, 17]].mean(0))
            elif self.predict_roi:
                P = P + hs.vel * min(max(t_cap_ns / 1e9 - hs.t, 0.0), 0.15)
            jcs = {}
            for slot in (0, 1):
                R, t = self.palm.pose[slot]
                jcs[slot] = (P - t) @ R
            if any((jc[:, 2] < 0.08).any() for jc in jcs.values()):
                continue
            hint = not (self.hint_reset and (hs.misses > 0 or fp is None))
            jobs.append((side, jcs, None, hint))
            hs.last_both = t_sub
        if not jobs:
            return False
        self.idle_future = None
        src = raw_imgs if raw_imgs is not None else imgs
        fut = self.pool.submit(self.worker.run_stereo3d, jobs, {0: src[0], 1: src[1]}, raw_imgs is not None)
        self.job = (fut, t_cap_ns, fast_pos, {0: imgs[0], 1: imgs[1]}, time.monotonic_ns(), 2 * len(jobs))
        self.job_kind = "s3d"
        self.job_info = dict(info or {})
        return True

    def _collect_s3d(self):
        fut, t_ns, fast_pos, job_imgs, t_sub_ns, njob = self.job
        info = self.job_info
        self.job, self.job_info, self.job_kind = None, {}, None
        t = t_ns / 1e9
        t_col_ns = time.monotonic_ns()
        try:
            res = fut.result()
        except Exception:
            return {}
        out = {}
        for side, rs, J3, p3, v0, dsec, tim in res:
            self.stats["runs"] += 1
            self.stats["ms"].append(dsec * 1e3)
            rec = {"id": self.next_id, "side": side, "slot": v0, "t_cap": t_ns, "t_sub": t_sub_ns,
                   "t0": tim.get("t0", 0), "t1": tim.get("t1", 0), "cpu_ns": tim.get("cpu_ns", 0),
                   "wait_ns": tim.get("wait_ns", 0), "t_col": t_col_ns, "njob": njob, "outcome": "rej", "rq": None,
                   "hint": int(tim.get("hint", True)), "inference": tim.get("inference", "unknown"),
                   "request_crops": tim.get("request_crops", 0), "dsp_body_ms": tim.get("dsp_body_ms", 0.0),
                   "path": info.get("path", "frame"), "seq": info.get("seq", 0),
                   "t_read": info.get("t_read", 0), "t_ready": info.get("t_ready", 0)}
            self.next_id += 1
            self.records.append(rec)
            hs = self.hands.get(side)
            if hs is None:
                rec["outcome"] = "gone"
                continue
            self.last_inference = rec["inference"]
            ok = {sl: r for sl, r in rs.items() if r is not None and r["is_hand"] >= 0.97}
            if not ok:
                self.stats["rejected"] += 1
                hs.misses += 1
                continue
            rays_h = {}
            for sl, r in ok.items():
                R, o = self.palm.pose[sl]
                rays_h[sl] = (o, r["rays"] @ R.T)
                fpn = fast_pos.get(side)
                hs.last_rays[sl] = (t, o, rays_h[sl][1], np.asarray(fpn[0], float) if fpn is not None else None)
                hs.geo_views[sl] = (t, o, rays_h[sl][1], np.asarray(r["rel_depth"], float))
                hs.curls_by_view[sl] = (t, np.asarray(r["curls"], float), np.asarray(r["curl_var"], float))
            P_tri = None
            if len(rays_h) == 2:
                (oa, ra), (ob, rb) = rays_h[0], rays_h[1]
                Pm, gap, s1, s2 = triangulate(np.broadcast_to(oa, ra.shape), ra, np.broadcast_to(ob, rb.shape), rb)
                if np.median(gap) < 0.02 and (s1 > 0.05).all() and (s2 > 0.05).all():
                    P_tri = Pm
            P3h = None
            if J3 is not None:
                R0, o0 = self.palm.pose[v0]
                P3h = J3 @ R0.T + o0
            if P3h is not None and P_tri is not None:
                P, rec["outcome"] = stereo3d.hybrid(P_tri, P3h), "s3d_hyb"
            elif P3h is not None:
                P, rec["outcome"] = P3h, "s3d"
            elif P_tri is not None:
                P, rec["outcome"] = P_tri, "stereo"
            else:
                self.stats["rejected"] += 1
                hs.misses += 1
                continue
            width = float(np.linalg.norm(P[5] - P[17]))
            jump = np.linalg.norm(self.palm_c(P) - self.palm_c(hs.P))
            if not np.isfinite(P).all() or not 0.045 < width < 0.12 or (jump > 0.25 and t - hs.t < 0.2):
                self.stats["rejected"] += 1
                hs.misses += 1
                rec["outcome"] = "jump"
                continue
            dtv = t - hs.t
            if 0.005 < dtv < 0.2:
                v = (self.palm_c(P) - self.palm_c(hs.P)) / dtv
                sp = np.linalg.norm(v)
                hs.vel = v * min(1.0, 6.0 / sp) if sp > 0 else v
            else:
                hs.vel = np.zeros(3)
            hs.hand_size = 0.9 * hs.hand_size + 0.1 * float(np.linalg.norm(P[0] - P[9]))
            hs.P, hs.t, hs.misses = P, t, 0
            self.stats["accepted"] += 1
            self.stats["stereo"] += 1
            rec["rq"] = self.palm.raw_quat(side, P)
            if self.s3d_curls == "geometric":
                raw_curls = finger_curls(P)
            else:
                vs = [hs.curls_by_view[sl][1:] for sl in ok]
                wsum = sum(1 / v for _, v in vs)
                raw_curls = np.clip(-(sum(c / v for c, v in vs) / wsum) / np.array(CURL_SCALE), 0, 1)
            curls = self.palm.curl_filt[side](raw_curls, t)
            conf = max(r["is_hand"] for r in ok.values())
            pose = self.palm._pose(side, P, t, conf=conf) + (curls,)
            ks = v0 if v0 in ok else next(iter(ok))
            out[side] = (pose, ks, ok[ks]["det_px"], t, job_imgs[ks])
        for side in list(self.hands):
            if t - self.hands[side].t > LOST_AFTER_S:
                del self.hands[side]
        self.collapsed = None
        if len(self.hands) == 2:
            L, R = self.hands["left"], self.hands["right"]
            if np.linalg.norm(self.palm_c(L.P) - self.palm_c(R.P)) < COLLAPSE_M:
                drop = "left" if L.created > R.created else "right"
                del self.hands[drop]
                out.pop(drop, None)
                self.collapsed = drop
                self.stats["collapses"] = self.stats.get("collapses", 0) + 1
        return out
