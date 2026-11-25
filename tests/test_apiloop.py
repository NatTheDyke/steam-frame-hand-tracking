"""live.py --loop api: what it connects to HandTracker, with stand-ins for the tracker and the devices."""
import os
import sys
import threading
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from handtrack import apiloop, live
    from framehands import HandTracker
    import inspect
    WHY = None if "engine" in inspect.signature(HandTracker).parameters else "this framehands has no engine option"
except ImportError as e:
    WHY = repr(e)


class Tel(list):
    def write(self, line):
        self.append(line)


class Bridge:
    def __init__(self):
        self.sent, self.enabled_mask = [], None
        self.splay_filt = {sd: types.SimpleNamespace(set=lambda v: None) for sd in ("left", "right")}

    def send(self, poses, t, ages):
        self.sent.append((dict(poses), t, ages))


class Thermal:
    SLOWER_C = 90.0

    def __init__(self):
        self.temp, self.paused = 50.0, False

    def interval(self, hz):
        return None if self.paused else 1.0 / (2.0 if self.temp >= self.SLOWER_C else hz)


class FakeTracker:
    def __init__(self):
        self.frames, self.polls, self.reset_calls, self.enabled_hands = [], [], 0, None
        self.keypoint_hz = self.detect_hz = None
        self.smoothing = self.views = self.stereo_comp = None
        self.frame_time_ns, self.sources, self.ev, self.fail = 0, {}, [], None
        self.out, self.stats, self.tracked = (), {}, set()

    def reset(self):
        self.reset_calls += 1

    def update(self, frame):
        if self.fail:
            e, self.fail = self.fail, None
            raise e
        self.frames.append(frame)
        self.frame_time_ns = frame.timestamp_ns
        return self.out

    def poll(self):
        return self.polls.pop(0) if self.polls else ()

    def take_events(self):
        ev, self.ev = self.ev, []
        return ev


def hand(side="left", curl=0.2):
    return types.SimpleNamespace(side=side, palm_position=np.zeros(3), palm_orientation=np.array([1.0, 0, 0, 0]), confidence=0.9,
                                 joints=np.zeros((21, 3)), curls=np.full(5, curl), position_age_ms=1.0, orientation_age_ms=2.0,
                                 joints_age_ms=3.0)


class Shm:
    """frames: capture times; every read of a camera gets the next one, then nothing (and stops the loop)"""

    def __init__(self, times, stop):
        self.times, self.n, self.stop = times, 0, stop

    def read(self, cam, last):
        if self.n >= 2 * len(self.times):
            self.stop.set()
            return None
        k = self.n // 2
        self.n += 1
        return (k + 1, self.times[k], np.zeros((4, 4), np.uint8))


def args(**kw):
    d = dict(backend="mercury", discover="mercury", merc_sched="batch", merc_views="auto", curl_source="mercury", merc_crop="auto",
             merc_predict_roi=False, merc_stereo_ms=40.0, curl_latest=False, merc_hint_reset=False, flow_agree=False,
             fast_orient=False, flow_v2=True, blend_ms=0.0, merc_fresh=False, mercury_shadow=False, size=640, max_speed=4.0,
             merc_cadence="capture", no_http=True, log=None, merc_hz=30.0, max_hz=7.0, idle_hz=5.0, discover_hz=2.0,
             merc_crops=4, merc_preflow=True, rot_smooth_ms=40.0, merc_stereo_comp="on", merc_early=True, det_crops=2)
    d.update(kw)
    return types.SimpleNamespace(**d)


@unittest.skipIf(WHY, WHY)
class ApiLoopTests(unittest.TestCase):
    def setUp(self):
        self.stop, self.tel, self.bridge, self.thermal = threading.Event(), Tel(), Bridge(), Thermal()
        self.hands = {"left", "right"}
        self.st = {"lat": [], "pairs": 0, "sent": 0, "fast_ms": [], "id_fix": 0, "det": [], "frames": 0, "hands": 0}

    def make(self, n=4, **kw):
        self.tr = FakeTracker()
        loop = apiloop.ApiLoop.__new__(apiloop.ApiLoop)
        loop.a, loop.src, loop.bridge, loop.tel, loop.thermal = args(**kw), None, self.bridge, self.tel, self.thermal
        loop.shm = Shm([10 ** 9 + k * 8_740_000 for k in range(n)], self.stop)
        loop.profile = types.SimpleNamespace(poll=lambda: None, values=dict(live.PROFILE_DEFAULTS))
        loop.hands_cfg = types.SimpleNamespace(get=lambda: set(self.hands))
        loop.stats, loop.lock, loop.stop = lambda: self.st, threading.Lock(), self.stop
        loop.tracker, loop.enabled, loop.off_ns, loop.last, loop.ms, loop.counts = self.tr, None, 0, None, [], {}
        return loop

    def test_hands_go_to_the_bridge_and_the_telemetry(self):
        loop = self.make()
        self.tr.out = (hand("left"),)
        self.tr.sources = {"left": types.SimpleNamespace(position="stereo", orientation="avg2", points=(3, 4))}
        loop.run()
        self.assertEqual(len(self.bridge.sent), 4)
        poses, t, ages = self.bridge.sent[-1]
        self.assertIsNone(poses["right"])
        self.assertEqual((len(poses["left"]), ages), (5, {"left": (1.0, 2.0)}))
        self.assertEqual(t, self.tr.frames[-1].timestamp_ns)
        f = [ln.split(",") for ln in self.tel if ln.startswith("F,")]
        self.assertEqual((len(f), f[0][3], f[0][4], f[0][6], f[0][9], f[0][10], len(f[0])), (4, "left", "stereo", "merc_avg2", "3", "4", 18))
        self.assertEqual((self.st["pairs"], self.st["sent"]), (4, 4))

    def test_no_enabled_hand_sends_empty_packets(self):
        self.hands = set()
        loop = self.make()
        loop.run()
        self.assertEqual((len(self.tr.frames), self.tr.reset_calls), (0, 1))
        self.assertEqual(len(self.bridge.sent), 4)
        self.assertEqual(self.bridge.enabled_mask, 0)
        self.assertTrue(all(p["left"] is None and p["right"] is None for p, _, _ in self.bridge.sent))

    def test_one_enabled_hand(self):
        self.hands = {"right"}
        self.make().run()
        self.assertEqual((self.tr.enabled_hands, self.bridge.enabled_mask), (("right",), 2))

    def test_hot_skips_every_other_frame_and_paused_stops_the_models(self):
        loop = self.make(n=6)
        self.thermal.temp = 91.0
        loop.run()
        self.assertEqual(len(self.tr.frames), 3)
        self.assertEqual(self.tr.keypoint_hz, 2.0)
        self.stop.clear()
        loop = self.make(n=2)
        self.thermal.temp, self.thermal.paused = 50.0, True
        loop.run()
        self.assertEqual((self.tr.keypoint_hz, self.tr.detect_hz), (0.0, 0.0))
        self.assertEqual(len(self.tr.frames), 2)                # the flow goes on

    def test_a_failure_in_the_tracker_loses_a_frame_not_the_loop(self):
        loop = self.make()
        self.tr.fail = RuntimeError("job")
        loop.run()
        self.assertEqual(len(self.tr.frames), 3)

    def test_curls_between_frames_go_out_with_the_last_packet(self):
        loop = self.make(n=1)
        self.tr.out = (hand("left", 0.2), hand("right", 0.2))
        self.tr.sources = {sd: types.SimpleNamespace(position="stereo", orientation="avg2", points=(1, 1)) for sd in ("left", "right")}
        self.tr.polls = [(hand("left", 0.9),)]
        loop.run()
        (p0, t0, _), (p1, t1, _) = self.bridge.sent
        self.assertEqual(t0, t1)
        self.assertEqual((p0["left"][4][0], p1["left"][4][0], p1["right"][4][0]), (0.2, 0.9, 0.2))
        self.assertEqual(sum(ln.startswith("U,") for ln in self.tel), 1)

    def test_the_profile_reaches_the_tracker(self):
        loop = self.make(n=0)
        pv = dict(live.PROFILE_DEFAULTS, axis_filter=(1.0, 2.0), merc_views="both", stereo_comp=False, send_offset_ms=3.0)
        loop.apply_profile(pv)
        self.assertEqual((self.tr.smoothing.axis, self.tr.smoothing.rotation_glide_ms, self.tr.views, self.tr.stereo_comp, loop.off_ns),
                         ((1.0, 2.0), 25.0, "both", False, 3_000_000))
        self.assertTrue(self.tel[-1].startswith("P,"))

    def test_unsupported_options_are_named(self):
        self.assertEqual(apiloop.unsupported(args()), [])
        self.assertEqual(apiloop.unsupported(args(flow_agree=True, no_http=False)), ["--flow-agree", "--no-http"])


if __name__ == "__main__":
    unittest.main()
