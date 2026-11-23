"""NPU service: wire format, one-client ownership, session lifetime and error handling, with a
stand-in session (no DSP needed)."""
from pathlib import Path
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtrack import npuservice as ns
from handtrack.inference import FallbackBackend, ServiceBackend


class FakeSession:
    """Deterministic outputs derived from the inputs, so the round trip can be checked."""
    opened = closed = 0

    def __init__(self, fail_on=None, det=None):
        FakeSession.opened += 1
        self.fail_on, self.idles, self.last_timing, self.det = fail_on, 0, None, det
        self.frames = 0

    def infer(self, feeds, model=0):
        out = []
        if self.det is not None and model == self.det:  # the detector (session index det)
            for f in feeds:
                assert f["inputImg"].shape == (1, 1, 160, 160)
                s = float(f["inputImg"].sum())
                out.append({name: np.full(shape, s, np.float32) for name, shape in ns.DET_OUTPUTS})
            self.last_timing = {"crops": len(feeds), "body_ms": np.float64(1.7)}
            return out
        if model == 1:  # the 3D stereo student: outputs from the pair's inputs
            for f in feeds:
                assert f["inputImg"].shape == (1, 2, 128, 128) and f["geom"].shape == (1, 16)
                s = float(f["inputImg"].sum() + f["geom"].sum())
                out.append({name: np.full(shape, s, np.float32) for name, shape in ns.S3D_OUTPUTS})
            self.last_timing = {"crops": len(feeds), "body_ms": np.float64(1.7)}
            return out
        for f in feeds:
            assert f["inputImg"].shape == (1, 1, 128, 128) and f["inputImg"].dtype == np.float32
            s = float(f["inputImg"].sum() + f["lastKeypoints"].sum() + f["useLastKeypoints"][0])
            if self.fail_on is not None and s == self.fail_on:
                raise RuntimeError("device fault")
            out.append({name: np.full(shape, s, np.float32) for name, shape in ns.OUTPUTS})
        self.last_timing = {"crops": len(feeds), "body_ms": np.float64(4.9)}
        return out

    def infer_many(self, reqs):
        outs, timings = [], []
        for m, f in reqs:
            outs.append(self.infer(f, model=m))
            timings.append(self.last_timing)
        self.last_timing = {"requests": timings, "body_ms": sum(t["body_ms"] for t in timings)}
        return outs

    def infer_frame(self, reqs):
        assert len({m for m, _ in reqs}) == len(reqs) <= 2
        self.frames += 1
        outs = self.infer_many(reqs)
        self.last_timing["segments"] = len(reqs)
        return outs

    def idle(self):
        self.idles += 1

    def close(self):
        FakeSession.closed += 1


def feed(v):
    return {"inputImg": np.full((1, 1, 128, 128), v, np.float32), "lastKeypoints": np.zeros((1, 42), np.float32),
            "useLastKeypoints": np.array([1.0], np.float32)}


def pair(v):
    return {"inputImg": np.full((1, 2, 128, 128), v, np.float32), "lastKeypoints": np.zeros((1, 84), np.float32),
            "useLastKeypoints": np.array([1.0], np.float32), "geom": np.ones((1, 16), np.float32)}


def det_crop(v):
    return {"inputImg": np.full((1, 1, 160, 160), v, np.float32)}


class ServiceTests(unittest.TestCase):
    s3d = det = None
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "npu.sock")
        FakeSession.opened = FakeSession.closed = 0
        self.fail_on = None
        self.srv = ns.Server("model", "jobs", self.path, os.getuid(),
                             session_factory=lambda: FakeSession(self.fail_on, det=1 if self.det else None),
                             s3d_model=self.s3d, det_model=self.det)
        self.t = threading.Thread(target=self.srv.run, daemon=True)
        self.t.start()
        for _ in range(100):
            if os.path.exists(self.path):
                break
            time.sleep(0.01)

    def tearDown(self):
        self.srv.stop = True
        self.t.join(5)
        os.rmdir(self.dir)

    def wait_closed(self, n):
        for _ in range(200):
            if FakeSession.closed >= n:
                return
            time.sleep(0.01)

    def test_a_three_crop_image_refuses_four_and_says_so(self):
        c = ns.Client(self.path)
        self.assertEqual(c.max_crops, 3)
        with self.assertRaises(ValueError):                    # the client knows the limit
            c.infer([feed(1)] * 4)
        c.info = dict(c.info, crops=4)                         # a client that believes otherwise
        with self.assertRaises(RuntimeError):                  # the service refuses
            c.infer([feed(1)] * 4)
        c.close()

    def test_a_four_crop_image_serves_four_in_order(self):
        self.srv.crops = 4
        c = ns.Client(self.path)
        self.assertEqual(c.max_crops, 4)
        outs = c.infer([feed(i + 1) for i in range(4)])
        for i, o in enumerate(outs):
            for name, shape in ns.OUTPUTS:
                self.assertTrue(np.all(o[name] == (i + 1) * 128 * 128 + 1))
        c.close()

    def test_image_crops_come_from_the_record(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ns._image_crops(d), 3)            # no record
            for rec, want in (({"max_crops": 4}, 4), ({}, 3), ({"max_crops": 9}, 3)):
                with open(os.path.join(d, "link.json"), "w") as f:
                    json.dump(rec, f)
                self.assertEqual(ns._image_crops(d), want)

    def test_round_trip_one_to_three_crops_in_order(self):
        c = ns.Client(self.path)
        for k in (1, 2, 3):
            outs = c.infer([feed(i + 1) for i in range(k)])
            self.assertEqual(len(outs), k)
            for i, o in enumerate(outs):
                want = (i + 1) * 128 * 128 + 1
                for name, shape in ns.OUTPUTS:
                    self.assertEqual(o[name].shape, shape)
                    self.assertTrue(np.all(o[name] == want))
        self.assertEqual(c.last_timing, {"crops": 3, "body_ms": 4.9})
        c.close()

    def test_keypoint_only_service_refuses_the_3d_student(self):
        c = ns.Client(self.path)
        self.assertEqual(c.models, (0,))
        with self.assertRaises(RuntimeError):
            c.infer([pair(1)], model=1)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)     # a raw model-1 request: error reply
        s.settimeout(2)
        s.connect(self.path)
        c.close()
        self.wait_closed(0)
        ns.send_msg(s, {"op": "infer", "n": 1, "model": 1}, ns.pack_arrays([pair(1)], ns.S3D_INPUTS))
        resp, _ = ns.recv_msg(s, lambda h: 0)
        self.assertFalse(resp["ok"])
        s.close()

    def test_socket_is_private(self):
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_session_opens_on_first_infer_and_closes_on_disconnect(self):
        c = ns.Client(self.path)
        self.assertEqual(FakeSession.opened, 0)            # ping alone does not touch the DSP
        c.infer([feed(1)])
        c.idle()
        self.assertEqual(FakeSession.opened, 1)
        c.close()
        self.wait_closed(1)
        self.assertEqual(FakeSession.closed, 1)

    def test_second_client_is_turned_away_and_falls_back_to_cpu(self):
        c1 = ns.Client(self.path)
        with self.assertRaises(RuntimeError):
            ns.Client(self.path)
        c1.close()

    def test_device_error_closes_session_and_reports(self):
        self.fail_on = 1 * 128 * 128 + 1
        c = ns.Client(self.path)
        with self.assertRaises(RuntimeError):
            c.infer([feed(1)])
        self.wait_closed(1)
        self.assertEqual(FakeSession.closed, 1)
        c.close()

    def test_malformed_requests_are_refused(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(self.path)
        ns.send_msg(s, {"op": "infer", "n": 5})               # more than four crops: dropped
        self.assertEqual(s.recv(16), b"")
        s.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(self.path)
        s.sendall(struct.pack("<I", 1 << 30))                 # absurd header length: dropped
        self.assertEqual(s.recv(16), b"")
        s.close()
        c = ns.Client(self.path)                              # the service is still fine
        self.assertEqual(len(c.infer([feed(2)])), 1)
        c.close()

    def test_fallback_backend_uses_cpu_without_a_service(self):
        class Cpu:
            name = "cpu"

            def infer(self, feeds):
                return [{n: np.zeros(s, np.float32) for n, s in ns.OUTPUTS} for _ in feeds]

            def close(self):
                pass
        b = FallbackBackend(Cpu(), lambda: ServiceBackend(os.path.join(self.dir, "missing.sock")))
        self.assertEqual(b.name, "cpu")
        self.assertEqual(len(b.infer([feed(1)])), 1)
        b2 = FallbackBackend(Cpu(), lambda: ServiceBackend(self.path))
        self.assertEqual(b2.name, "framenpu")
        self.assertTrue(np.all(b2.infer([feed(1)])[0]["curls"] == 128 * 128 + 1))
        b2.close()


if __name__ == "__main__":
    unittest.main()


class Stereo3DServiceTests(ServiceTests):
    """the same service with the 3D stereo student as model 1"""
    s3d = "s3d.onnx"

    def test_keypoint_only_service_refuses_the_3d_student(self):
        self.skipTest("this service serves the 3D student")

    def test_3d_student_round_trip_one_or_two_pairs_between_keypoint_requests(self):
        c = ns.Client(self.path)
        self.assertEqual(c.models, (0, 1))
        for k in (1, 2):
            c.infer([feed(1)])
            outs = c.infer([pair(i + 1) for i in range(k)], model=1)
            for i, o in enumerate(outs):
                want = (i + 1) * 2 * 128 * 128 + 16
                for name, shape in ns.S3D_OUTPUTS:
                    self.assertEqual(o[name].shape, shape)
                    self.assertTrue(np.all(o[name] == want))
        self.assertEqual(FakeSession.opened, 1)                 # one session serves both models
        with self.assertRaises(ValueError):
            c.infer([pair(1)] * 3, model=1)                    # at most two pairs per request
        c.close()

    def test_service_stereo3d_latches_to_cpu_on_failure(self):
        from handtrack import stereo3d

        class Broken:
            models, last_timing = (0, 1), None

            def infer(self, feeds, model=0):
                raise RuntimeError("NPU service: device fault")

        class Cpu:
            runs = 0

            def run(self, *a):
                Cpu.runs += 1
                return np.zeros((21, 3)), 1.0

        s3 = stereo3d.ServiceStereo3D(Broken(), Cpu)
        q, P = (1.0, 0.0, 0.0, 0.0), np.random.default_rng(0).normal(0, 0.05, (21, 3)) + [0, 0, 0.4]
        prep = (0, None, q, 0.2, False, {"inputImg": np.zeros((1, 1, 128, 128), np.float32),
                                          "lastKeypoints": np.zeros((1, 42), np.float32),
                                          "useLastKeypoints": np.zeros(1, np.float32)})
        for _ in range(2):
            s3.run(prep, prep, P, True, np.eye(3), np.zeros(3))
        self.assertEqual((s3.name, Cpu.runs), ("cpu", 2))


class DetectorServiceTests(ServiceTests):
    """the service with Mercury's hand detector as model 2 (session index 1) and infer_frame"""
    det = "det.onnx"

    def test_detector_round_trip_and_frame(self):
        c = ns.Client(self.path)
        self.assertEqual((c.models, c.max_frame), ((0, 2), ns.MAX_FRAME))
        for k in (1, 2):
            outs = c.infer([det_crop(i + 1) for i in range(k)], model=2)
            for i, o in enumerate(outs):
                for name, shape in ns.DET_OUTPUTS:
                    self.assertEqual(o[name].shape, shape)
                    self.assertTrue(np.all(o[name] == (i + 1) * 160 * 160))
        with self.assertRaises(ValueError):
            c.infer([det_crop(1)] * 3, model=2)
        for reqs in ([(2, [det_crop(3)]), (0, [feed(1), feed(2)])], [(0, [feed(5)]), (2, [det_crop(1), det_crop(2)])]):
            got = c.infer_frame(reqs)
            self.assertEqual(c.last_timing["segments"], 2)
            for (m, f), g in zip(reqs, got):
                for a, b in zip(c.infer(f, model=m), g):
                    for k in a:
                        np.testing.assert_array_equal(a[k], b[k])
        with self.assertRaises(ValueError):
            c.infer_frame([(0, [feed(1)]), (0, [feed(2)])])  # distinct models
        with self.assertRaises(ValueError):
            c.infer_frame([(0, [feed(1)]), (2, [det_crop(1)]), (2, [det_crop(2)])])
        with self.assertRaises(RuntimeError):
            c.infer([pair(1)], model=1)  # the 3D student is not served
        self.assertEqual(FakeSession.opened, 1)
        c.close()

    def test_keypoint_only_service_refuses_the_3d_student(self):
        self.skipTest("this service also serves the detector")

    def test_malformed_infer_frame_is_dropped(self):
        for reqs in ([], [{"model": 0, "n": 1}, {"model": 0, "n": 1}], [{"model": 2, "n": 3}],
                     [{"model": 0, "n": 1}, {"model": 2, "n": 1}, {"model": 1, "n": 1}]):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(self.path)
            ns.send_msg(s, {"op": "infer_frame", "reqs": reqs})
            self.assertEqual(s.recv(16), b"", reqs)
            s.close()
        c = ns.Client(self.path)
        self.assertEqual(len(c.infer_frame([(2, [det_crop(1)])])[0]), 1)
        c.close()

    def test_3d_student_and_detector_are_refused_together(self):
        with self.assertRaises(ValueError):
            ns.Server("model", "jobs", self.path + ".x", os.getuid(), s3d_model="s3d.onnx", det_model="det.onnx")


class InferManyTests(Stereo3DServiceTests):
    """infer_many: several requests of both models in one round trip"""

    def test_infer_many_equals_single_requests(self):
        c = ns.Client(self.path)
        self.assertEqual(c.max_seq, ns.MAX_SEQ)
        reqs = [(0, [feed(1), feed(2), feed(3)]), (0, [feed(4)]), (1, [pair(1), pair(2)])]
        many = c.infer_many(reqs)
        self.assertEqual([len(r) for r in many], [3, 1, 2])
        for (m, f), got in zip(reqs, many):
            for a, b in zip(c.infer(f, model=m), got):
                self.assertEqual(sorted(a), sorted(b))
                for k in a:
                    np.testing.assert_array_equal(a[k], b[k])
        c.infer_many(reqs)
        self.assertEqual(len(c.last_timing["requests"]), 3)
        with self.assertRaises(ValueError):
            c.infer_many([(0, [feed(1)])] * (ns.MAX_SEQ + 1))
        with self.assertRaises(ValueError):
            c.infer_many([(1, [pair(1)] * 3)])
        c.close()

    def test_malformed_infer_many_is_dropped(self):
        for reqs in ([], [{"model": 0, "n": 5}], [{"model": 1, "n": 3}], [{"model": 7, "n": 1}], "x",
                     [{"model": 0, "n": 1}] * (ns.MAX_SEQ + 1)):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(self.path)
            ns.send_msg(s, {"op": "infer_many", "reqs": reqs})
            self.assertEqual(s.recv(16), b"", reqs)
            s.close()
        c = ns.Client(self.path)
        self.assertEqual(len(c.infer_many([(1, [pair(1)])])[0]), 1)
        c.close()

    def test_stereo_frame_is_one_round_trip_and_matches_per_hand_requests(self):
        from handtrack import mpworker, stereo3d
        from handtrack.inference import FallbackBackend, ServiceBackend

        client = ns.Client(self.path)
        calls = []
        orig = client.infer_many

        def counted(reqs):
            calls.append([(m, len(f)) for m, f in reqs])
            return orig(reqs)
        client.infer_many = counted

        class Svc(ServiceBackend):
            def __init__(self):
                self.client = client

        class Cpu:
            def infer(self, feeds):
                raise AssertionError("CPU keypoints used")

            def close(self):
                pass

        backend = FallbackBackend(Cpu(), Svc)
        rng = np.random.default_rng(1)

        class MK:
            def __init__(self):
                self.backend = backend

            def _prepare(self, sl, img, jc, right, use_hint, expand, raw):
                q = (1.0, 0.0, 0.0, 0.0)
                v = float(sl * 10 + img)
                return (sl, None, q, 0.2, right, feed(v))

            def _decode(self, prep, o):
                return {"v": float(np.ravel(o["heatmap_xy"])[0])}

        mk = MK()
        s3 = stereo3d.ServiceStereo3D(client, lambda: (_ for _ in ()).throw(AssertionError("CPU 3D used")))
        P = rng.normal(0, 0.03, (21, 3)) + [0, 0, 0.4]
        jobs = [("left", {0: P, 1: P}, 1.0, True), ("right", {0: P + 0.01, 1: P + 0.01}, 1.0, False)]
        imgs = {0: 1, 1: 2}
        old = (mpworker.init_mercury, mpworker.init_stereo3d, mpworker._many_off)
        mpworker.init_mercury = lambda: mk
        mpworker.init_stereo3d = lambda: (s3, ([0, 1], np.eye(3), np.array([0.1, 0, 0])))
        try:
            mpworker._many_off = False
            many = mpworker.run_stereo3d(jobs, imgs)
            self.assertEqual(calls, [[(1, 2), (0, 3), (0, 1)]])      # the whole frame, one round trip
            mpworker._many_off = True
            single = mpworker.run_stereo3d(jobs, imgs)
            self.assertEqual(len(calls), 1)
            for a, b in zip(many, single):
                self.assertEqual(a[0], b[0])
                self.assertEqual(a[1], b[1])
                np.testing.assert_array_equal(a[2], b[2])
                self.assertEqual(a[3], b[3])
            # a failed frame latches both models off the NPU, never retried there
            mpworker._many_off = False
            client.infer_many = lambda reqs: (_ for _ in ()).throw(RuntimeError("NPU service: device fault"))
            backend.cpu = type("C", (), {"infer": lambda self, f: [
                {n: np.zeros(sh, np.float32) for n, sh in ns.OUTPUTS} for _ in f], "close": lambda self: None})()
            s3.cpu_factory = lambda: type("S", (), {"run": lambda self, *a: (np.zeros((21, 3)), 1.0)})()
            out = mpworker.run_stereo3d(jobs, imgs)
            self.assertTrue(mpworker._many_off)
            self.assertEqual((backend.name, s3.name), ("cpu", "cpu"))
            self.assertEqual(len(out), 2)
        finally:
            mpworker.init_mercury, mpworker.init_stereo3d, mpworker._many_off = old
            client.close()
