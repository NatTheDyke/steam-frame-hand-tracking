"""live.py's tracking loop on framehands.HandTracker (--loop api).

The classic loop in live.py keeps its own copy of the tracking lifecycle (flow, keypoint jobs,
discovery, composing a hand); here HandTracker does that and this module only connects it: camera
frames in, hands to the PC bridge and telemetry out, the profile, the enabled hands and the thermal
guard applied. It serves the setup the headset runs (Mercury tracking and discovery in the worker
process); unsupported() names the options it has no equivalent for.

Differences from the classic loop: a result that arrives between camera frames updates the curls at
once, but the next keypoint job is sent with the next frame (classic may send it between frames).
"""
import functools
import sys
import threading
import time

import numpy as np

from handtrack import mercdet, mpworker

SIDES = ("left", "right")


class WorkerEngine:
    """HandTracker's engine in the Mercury worker process."""

    run_mercury = staticmethod(mpworker.run_mercury)
    idle_mercury = staticmethod(mpworker.idle_mercury)

    def __init__(self, det_size, ncrop):
        self.discover = functools.partial(mpworker.discover, det_size, ncrop)
        self.layout = mercdet.LAYOUTS[ncrop]


# option: (attribute of the parsed arguments, the values --loop api supports)
_SUPPORTED = (("backend", ("mercury",)), ("discover", ("mercury",)), ("merc_sched", ("batch",)),
              ("merc_views", ("auto", "both", "alternate")), ("curl_source", ("mercury",)), ("merc_crop", ("auto", "raw")),
              ("merc_predict_roi", (False,)), ("merc_stereo_ms", (40.0,)), ("curl_latest", (False,)),
              ("merc_hint_reset", (False,)), ("flow_agree", (False,)), ("fast_orient", (False,)), ("flow_v2", (True,)),
              ("blend_ms", (0.0,)), ("merc_fresh", (False,)), ("mercury_shadow", (False,)), ("size", (640,)),
              ("max_speed", (4.0,)), ("merc_cadence", ("capture",)), ("no_http", (True,)), ("log", (None, "")))


def unsupported(a):
    """The options of a that --loop api does not implement, as text (empty: fine)."""
    import inspect
    try:
        from framehands import HandTracker
        if "engine" not in inspect.signature(HandTracker).parameters:
            raise ImportError
    except ImportError:
        return ["(the installed framehands: it needs HandTracker with an engine option)"]
    return ["--" + k.replace("_", "-") for k, ok in _SUPPORTED if getattr(a, k) not in ok]


def smoothing(pv, rot_smooth_ms):
    from framehands.tracker import Smoothing
    glide = rot_smooth_ms if pv["rot_glide_ms"] is None else pv["rot_glide_ms"]
    return Smoothing(axis=pv["axis_filter"], centre=pv["centre_filter"], curl=pv["curl_filter"],
                     position=pv["pos_filter"], rotation_glide_ms=glide)


class ApiLoop:
    def __init__(self, a, *, rig, src, shm, bridge, tel, thermal, pool, ring, profile, hands_cfg, stats, lock, stop, fingers):
        from framehands import HandTracker
        self.a, self.src, self.shm, self.bridge, self.tel, self.thermal = a, src, shm, bridge, tel, thermal
        self.profile, self.hands_cfg, self.stats, self.lock, self.stop = profile, hands_cfg, stats, lock, stop
        self.det_size, self.hz = a.size, a.merc_hz
        self.tracker = HandTracker(
            rig, engine=WorkerEngine(a.size, a.det_crops), executor=pool, ring=ring, source=src.name,
            keypoint_hz=a.merc_hz, detect_hz=a.max_hz, idle_detect_hz=a.idle_hz, redetect_hz=a.discover_hz,
            views=a.merc_views, max_crops=a.merc_crops, curl_joint_fingers=tuple(fingers), preflow=a.merc_preflow, events=True)
        self.enabled = None
        self.off_ns = 0
        self.last = None                # (poses, t_cap ns, ages) of the newest packet
        self.ms, self.counts = [], {}   # crop times, and the stats at the last report

    # ---------------------------------------------------------------- settings
    def apply_profile(self, pv):
        self.tracker.smoothing = smoothing(pv, self.a.rot_smooth_ms)
        self.off_ns = int(pv["send_offset_ms"] * 1e6)
        v = pv["merc_views"] or self.a.merc_views
        self.tracker.views = v if v != "stereo3d" else "alternate"
        self.tracker.stereo_comp = (self.a.merc_stereo_comp == "on") if pv["stereo_comp"] is None else pv["stereo_comp"]
        if self.bridge is not None:
            for sd in SIDES:
                self.bridge.splay_filt[sd].set(pv["curl_filter"])
        if self.tel is not None:
            self.tel.write("P,%d,%s" % (time.monotonic_ns(), ";".join(
                "%s=%s" % (k, "off" if v is None else (v if not isinstance(v, tuple) else "%g/%g" % v))
                for k, v in sorted(pv.items()))))

    def apply_hands(self):
        """HandTracker tracks at least one hand: with none enabled, frames are not processed."""
        en = self.hands_cfg.get()
        if en != self.enabled:
            self.enabled = set(en)
            if en:
                self.tracker.enabled_hands = tuple(sd for sd in SIDES if sd in en)
            else:
                self.tracker.reset()
        if self.bridge is not None:
            self.bridge.enabled_mask = (1 if "left" in en else 0) | (2 if "right" in en else 0)
        return bool(en)

    def throttle(self):
        """Thermal guard: keypoint and detector rates, and frames skipped while hot"""
        iv = self.thermal.interval(self.a.merc_hz)
        self.tracker.keypoint_hz = 0.0 if iv is None else 1.0 / iv
        iv = self.thermal.interval(self.a.max_hz)
        self.tracker.detect_hz = 0.0 if iv is None else 1.0 / iv
        return self.thermal.temp >= self.thermal.SLOWER_C

    # ---------------------------------------------------------------- output
    def send(self, hands, t_cap):
        poses, ages = {sd: None for sd in SIDES}, {}
        for h in hands:
            poses[h.side] = (h.palm_position, h.palm_orientation, h.confidence, h.joints, h.curls)
            ages[h.side] = (h.position_age_ms, h.orientation_age_ms)
        if self.bridge is not None:
            self.bridge.send(poses, t_cap + self.off_ns, ages)
        self.last = (poses, t_cap, ages)
        t_send = time.monotonic_ns()
        if self.tel is not None:
            for h in hands:
                s = self.tracker.sources[h.side]
                late = (t_send - t_cap) / 1e6
                self.tel.write("F,%d,%d,%s,%s,%.1f,%s,%.1f,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
                    t_cap, t_send, h.side, "mediapipe" if s.position == "observation" else s.position,
                    late + h.position_age_ms, "merc_" + s.orientation, late + h.orientation_age_ms, late + h.joints_age_ms,
                    s.points[0], s.points[1], *h.palm_position, *h.palm_orientation))
        with self.lock:
            st = self.stats()
            st["lat"].append((time.monotonic_ns() - t_cap) / 1e6)
            st["pairs"] += 1
            st["sent"] += len(hands)

    def send_curls(self, hands):
        """a result between camera frames: the newest packet again, with the new curls (classic's
        send_curl_update: same capture time and pose, so the driver takes only the curls)"""
        if not hands or self.last is None or self.bridge is None:
            return
        poses, t_cap, ages = self.last
        poses = dict(poses)
        changed = [h for h in hands if poses.get(h.side) is not None]
        for h in changed:
            poses[h.side] = poses[h.side][:4] + (h.curls,)
        if not changed:
            return
        self.bridge.send(poses, t_cap + self.off_ns, ages)
        self.last = (poses, t_cap, ages)
        if self.tel is not None:
            t_send = time.monotonic_ns()
            for h in changed:
                self.tel.write("U,%d,%d,%s,%.1f" % (t_cap, t_send, h.side, (t_send - t_cap) / 1e6 + h.joints_age_ms))

    def events(self):
        for ev in self.tracker.take_events():
            kind, v = ev[0], ev[1:]
            if kind == "observation":
                self._observation(v[0])
            elif self.tel is None:
                pass
            elif kind == "keypoints":
                self.tel.write("K,%d,%s,%d" % (int(v[0] * 1e9), v[1], v[2]))
            elif kind == "collapse":
                self.tel.write("C,%d,%s" % (self.last[1] if self.last else 0, v[0]))
            elif kind == "identity":
                self.tel.write("I,%d,%s" % (int(v[0] * 1e9), v[1]))
            elif kind == "discovery":
                now = time.monotonic_ns()
                for sl, (n, dt) in v[1].items():
                    self.tel.write("M,%d,%d,mdet,%d,%.1f,%d" % (int(v[0] * 1e9), now, sl, dt * 1e3, n))
            if kind == "identity":
                with self.lock:
                    self.stats()["id_fix"] += 1
            elif kind == "discovery":
                with self.lock:
                    st = self.stats()
                    st["det"] += [dt for _, dt in v[1].values()]
                    st["frames"] += len(v[1])
                    st["hands"] += sum(n for n, _ in v[1].values())

    def report(self, dt):
        """live.py's "mercury tracking" line since the last call"""
        now = self.tracker.stats
        d = {k: now.get(k, 0) - self.counts.get(k, 0) for k in set(now) | set(self.counts)}
        self.counts, ms, self.ms = now, self.ms, []
        return ("   mercury tracking: %d crops (%.1f/s), %.1f ms median, accepted %d (stereo %d, single-view %d), "
                "rejected %d, collapses %d, seeds blocked %d, tracked %s" % (
                    d["runs"], d["runs"] / dt, np.median(ms) if ms else 0, d["accepted"], d["stereo"], d["mono"],
                    d["rejected"], d.get("collapses", 0), d.get("seed_blocked", 0), sorted(self.tracker.tracked)))

    def _observation(self, rc):
        self.ms.append((rc["t1"] - rc["t0"]) / 1e6)
        if self.tel is None:
            return
        q = rc["rq"] if rc["rq"] is not None else (0, 0, 0, 0)
        self.tel.write("R,%d,%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%d,%s,%.4f,%.4f,%.4f,%.4f,%d,%s,%d,%.3f,%s,%d,%d,%d" % (
            rc["id"], rc["side"], rc["slot"], rc["t_cap"], rc["t_sub"], rc["t0"], rc["t1"], rc["t_col"],
            rc["cpu_ns"] / 1e6, rc["wait_ns"] / 1e6, rc["njob"], rc["outcome"], *q, rc["hint"], rc["inference"],
            rc["request_crops"], rc["dsp_body_ms"], rc["path"], rc["seq"], rc["t_read"], rc["t_ready"]))

    # ---------------------------------------------------------------- the loop
    def run(self):
        from framehands import Frame
        self.apply_profile(self.profile.values)
        last, frame_no = {0: 0, 1: 0}, 0
        while not self.stop.is_set():
            pv = self.profile.poll()
            if pv is not None:
                self.apply_profile(pv)
            on = self.apply_hands()
            r0, r1 = self.shm.read(0, last[0]), self.shm.read(1, last[1])
            if not r0 or not r1:
                try:
                    changed = self.tracker.poll()
                    self.send_curls(changed if self.a.merc_early else ())
                except Exception as e:                # noqa: BLE001 -- as in update()
                    print("tracker: %r" % (e,), file=sys.stderr, flush=True)
                self.events()
                time.sleep(0.002)
                continue
            if abs(r0[1] - r1[1]) > 8e6:              # not the same capture: wait for the lagging camera
                if r0[1] < r1[1]:
                    last[0] = r0[0]
                else:
                    last[1] = r1[0]
                continue
            last[0], last[1] = r0[0], r1[0]
            frame_no += 1
            hot = self.throttle()
            t_cap = min(r0[1], r1[1])
            if not on:
                self.send((), t_cap)
                continue
            if hot and frame_no % 2 == 1:
                continue                              # a frame skipped for heat sends nothing
            tf = time.perf_counter()
            try:
                hands = self.tracker.update(Frame(t_cap, r0[2], r1[2]))
            except Exception as e:                    # noqa: BLE001 -- a failed job loses one frame, not the tracker
                print("tracker: %r" % (e,), file=sys.stderr, flush=True)
                self.events()
                continue
            with self.lock:
                self.stats()["fast_ms"].append((time.perf_counter() - tf) * 1e3)
            self.send(hands, self.tracker.frame_time_ns)
            self.events()
