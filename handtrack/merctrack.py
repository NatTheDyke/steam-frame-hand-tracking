"""Continuous per-hand tracking with Monado Mercury's keypoint model (opt-in: --backend mercury).

MediaPipe stays the DISCOVERY path (finds hands, initial 21-joint 3D).  Once a hand has a 3D
joint state, this tracker re-observes it every cycle with one Mercury crop per hand (cameras
alternate), using the hand's own previous joints as ROI and temporal hint.  A single view gives a
full articulated 3D hand: Mercury predicts each joint's ray AND its depth relative to the middle
knuckle in hand-size units; absolute depth is anchored to the 60 Hz stereo palm position from the
optical-flow tracker when available, else to the previous state.  When the other camera's result
for the same hand is fresh (< STEREO_MS), joints are triangulated instead.

Views per hand (views): "both" crops the hand from both cameras of the same frame (at most
MAX_CROPS crops per job; with two hands the second view takes turns), so stereo joints come from
one instant.  "alternate" (the original) crops one camera per frame and triangulates against the
other camera's previous result, captured a frame earlier: a moving hand has moved between the two
views, which puts a triangulated palm 2-7 cm off at 0.3-0.6 m/s (more at distance) and tilts the
palm when the wrist turns.  "auto" = both when inference is on the NPU, alternate on the CPU.
With stereo_comp the older view's rays are shifted by the palm's motion since its capture
(from the 60 Hz two-camera palm positions), which removes the translation part of that error.

The Mercury model license is UNVERIFIED (local evaluation only).
"""
import time

import numpy as np

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
        self.views = "auto"                       # "both" | "alternate" | "auto" (module docstring)
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
        jobs = []
        sides = list(self.hands)
        if self.sched == "perhand":
            sides = [min(sides, key=lambda sd: self.hands[sd].last_submit)]
        t_sub = time.monotonic()
        both = self.views == "both" or (self.views == "auto" and self.last_inference != "cpu")
        # second views take turns between hands: the hand that had both views longest ago goes first
        spare = MAX_CROPS - len(sides) if both else 0
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
        if raw_imgs is not None:
            fut = self.pool.submit(self.worker.run_mercury, jobs, {s: raw_imgs[s] for s in used}, True)
        else:
            fut = self.pool.submit(self.worker.run_mercury, jobs, {s: imgs[s] for s in used})
        self.job = (fut, t_cap_ns, fast_pos,
                    {s: imgs[s] for s in used}, time.monotonic_ns(), len(jobs))
        if anchor_pos is not None:
            self.job = self.job[:2] + (anchor_pos,) + self.job[3:]
        self.job_info = dict(info or {})
        return True

    def collect(self):
        """If the in-flight job finished: update hand states.
        Returns {side: (pose tuple, slot, det_px, t_capture_s, image_the_crop_came_from)}."""
        if self.job is None or not self.job[0].done():
            return {}
        fut, t_ns, fast_pos, job_imgs, t_sub_ns, njob = self.job
        info = self.job_info
        self.job, self.job_info = None, {}
        t = t_ns / 1e9
        t_col_ns = time.monotonic_ns()
        try:
            res = fut.result()
        except Exception:
            return {}
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
