"""The tracker, framehands, framenpu and the NPU service together, against known hands.

    FI_MODELS=<dir with student_ft7.onnx, grayscale_detection_160x160.onnx> python -m unittest discover -s integration

The models are only opened (their hashes and manifests are checked): the stand-in session answers."""
import collections
import os
from pathlib import Path
import signal
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack  # noqa: E402

try:                                                    # --loop api needs a HandTracker with an engine option
    import inspect
    from framehands import HandTracker
    API = "engine" in inspect.signature(HandTracker).parameters
except ImportError:
    API = False

D = os.environ.get("FI_MODELS")
MODELS = [Path(D or "") / n for n in ("student_ft7.onnx", "grayscale_detection_160x160.onnx")]
WHY = ("FI_MODELS not set" if not D else next(("%s missing" % m for m in MODELS if not m.is_file()), None)) or stack.blocked()

# the tracker measures 0.3 mm median / 0.9 mm p99 here; ft7 decoded half a cell off is ~50 mm
MEDIAN_M, P99_M = 0.002, 0.006
OPS = ("infer", "frame", "many")


def axes(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


if WHY and os.environ.get("FI_REQUIRED"):              # CI: a skip would look like a pass
    raise RuntimeError(WHY)


@unittest.skipIf(WHY, WHY)
class ClosedLoop(unittest.TestCase):
    loop = "classic"

    def stack(self, **kw):
        return stack.Stack(MODELS, args=("--loop", self.loop), **kw)

    def up(self, **kw):
        s = self.stack(**kw)
        s.__enter__()
        self.addCleanup(s.__exit__)
        self.assertTrue(s.wait(s.tracked, 40), "no hands tracked\n" + s.log("live")[-3000:] + s.log("serve")[-1000:])
        return s

    def assert_accurate(self, s, since):
        for side, e in s.errors(since).items():
            self.assertGreater(len(e), 60, side)
            self.assertLess(stack.quantile(e, 0.5), MEDIAN_M, side)
            self.assertLess(stack.quantile(e, 0.99), P99_M, side)

    def test_hands_are_where_they_are(self):
        s = self.up()
        time.sleep(1)                                  # the filters settle
        n = len(s.driver.packets)
        r = len(s.requests())
        c0 = s.driver.packets[-1][2]
        time.sleep(6)
        self.assert_accurate(s, n)
        got = s.driver.packets[n:]
        self.assertEqual(s.driver.bad, 0)
        self.assertGreater(s.driver.pongs, 20)
        self.assertTrue(all(b[1] == a[1] + 1 for a, b in zip(got, got[1:])), "a bridge packet was lost")
        self.assertGreater(sum(all(h["valid"] for h in p[3].values()) for p in got) / len(got), 0.98)
        for p in got[::20]:
            for side, h in p[3].items():               # palm to the face, fingers up
                R = axes(h["quat"])
                self.assertLess(np.degrees(np.arccos(np.clip(-R[1, 2], -1, 1))), 5, side + ": fingers")
                self.assertLess(np.degrees(np.arccos(np.clip(-R[2, 1], -1, 1))), 5, side + ": back of the hand")
                self.assertLess(h["curl"].max(), 0.2, side + ": open hand")
        # the requests of those seconds: a crop per hand and frame, never over a request's limit
        frames = (s.driver.packets[-1][2] - c0) * s.fps / 1e9
        crops = [(m, k) for op, reqs in s.requests()[r:] if op in OPS for m, k in reqs]
        kp = sum(k for m, k in crops if m == 0)
        self.assertTrue(all(1 <= k <= (4, 2)[m] for m, k in crops))
        self.assertLess(abs(kp / frames - 2), 0.15, "keypoint crops per camera frame")
        self.assertLess(sum(op in OPS for op, _ in s.requests()[r:]) / frames, 1.3, "requests per camera frame")
        hints = [v for op, v in s.requests()[r:] if op == "hint"]
        self.assertGreater(len(hints), 0.9 * kp, "crops without a hint")
        self.assertLess(stack.quantile(hints, 0.5), 0.05, "hints are not where the joints are")
        self.assertIn("mercury-ft7, heatmap cell offset 0", s.log("live"))
        self.assertNotRegex(s.log("live"), "using CPU|Traceback")
        self.assertNotIn("WARNING", s.log("serve"))
        # stopping leaves the device closed, once
        self.assertEqual(s.stop("live"), 0, s.log("live")[-2000:])
        s.wait(lambda: s.requests()[-1][0] == "close", 5)
        count = collections.Counter(k for k, _ in s.requests())
        self.assertEqual((count["open"], count["close"], s.requests()[-1][0]), (1, 1, "close"))

    def test_one_hand_is_the_hand_it_is(self):
        """With two hands the tracker can sort their sides out by position; alone, the detector's side
        and the crop's mirroring have to be right."""
        for side, other in (("left", "right"), ("right", "left")):
            with self.subTest(side), self.stack(env={"FI_HAND": side}) as s:
                self.assertTrue(s.wait(lambda: s.tracked(30, (side,)), 30), s.log("live")[-2000:])
                self.assertFalse(s.tracked(1, (other,)), "the other hand was reported")

    def test_a_restarted_tracker_gets_the_npu_back(self):
        s = self.up(env={"FI_CLOSE_S": "0.4"})         # slower than the device's
        for _ in range(2):
            # gone without a goodbye, and back while its session is still closing
            p = s.procs.pop("live")
            children = [int(c) for t in Path("/proc/%d/task" % p.pid).iterdir() for c in (t / "children").read_text().split()]
            p.kill()
            p.wait()
            for c in children:
                try:
                    os.kill(c, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            n = len(s.driver.packets)
            s.start_tracker()
            self.assertTrue(s.wait(lambda: s.tracked(30, since=n), 40), s.log("live")[-3000:])
        self.assertNotIn("using CPU", s.log("live"))
        self.assertEqual(collections.Counter(k for k, _ in s.requests())["open"], 3)

    def test_a_device_fault_moves_the_tracker_to_the_cpu_and_is_never_retried(self):
        s = self.up(env={"FI_FAIL_AFTER": "60"})
        self.assertTrue(s.wait(lambda: "using CPU" in s.log("live"), 30), s.log("live")[-2000:])
        n = len(s.driver.packets)
        time.sleep(4)
        self.assertEqual(s.log("live").count("using CPU"), 1)
        self.assertEqual(s.log("serve").count("session opened"), 1)
        self.assertEqual([k for k, _ in s.requests()].count("open"), 1)
        self.assertGreater(len(s.driver.packets) - n, 60, "the tracking loop stopped")
        self.assertNotIn("Traceback", s.log("live"))
        self.assertIsNone(s.procs["live"].poll())

    def test_a_killed_service_leaves_a_running_tracker(self):
        s = self.up()
        p = s.procs.pop("serve")
        p.kill()
        p.wait()
        self.assertTrue(s.wait(lambda: "using CPU" in s.log("live"), 30), s.log("live")[-2000:])
        n = len(s.driver.packets)
        time.sleep(3)
        self.assertGreater(len(s.driver.packets) - n, 45, "the tracking loop stopped")
        self.assertNotIn("Traceback", s.log("live"))

    def test_a_second_tracker_is_turned_away(self):
        import framenpu
        s = self.up()
        r = len(s.requests())
        with self.assertLogs("framenpu", "WARNING") as got:
            session = framenpu.open([(str(MODELS[0]), "mercury-ft7")], "service", socket=s.sock)
        try:
            self.assertEqual(session.name, "cpu")
            self.assertIn("busy", got.output[0])
        finally:
            session.close()
        n = len(s.driver.packets)
        time.sleep(2)
        self.assert_accurate(s, n)
        self.assertGreater(len(s.requests()), r)
        self.assertNotIn("using CPU", s.log("live"))


@unittest.skipIf(WHY, WHY)
@unittest.skipUnless(API, "this framehands' HandTracker has no engine option")
class ClosedLoopApi(ClosedLoop):
    """The same, with the tracking loop on framehands.HandTracker (live.py --loop api)."""
    loop = "api"


if __name__ == "__main__":
    unittest.main()
