"""The Mercury worker (mpworker) on one framenpu session: a stereo frame is one service round
trip (infer_many) and matches the per-hand requests; a failure latches both models to the CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import framenpu  # noqa: E402
from framenpu import service, session as S  # noqa: E402

KP, S3D = "mercury-ft7", "mercury-stereo3d-tpc1"


def feed(v):
    return {"inputImg": np.full((1, 1, 128, 128), v, np.float32), "lastKeypoints": np.zeros((1, 42), np.float32),
            "useLastKeypoints": np.array([1.0], np.float32)}


class FakeDSP:
    """service-side session: outputs filled with a sum of the inputs"""

    def __init__(self):
        self.last_timing = None

    def infer(self, feeds, model=0):
        specs = service.S3D_OUTPUTS if model == 1 else service.OUTPUTS
        out = [{n: np.full(sh, float(sum(np.asarray(v).sum() for v in f.values())), np.float32) for n, sh in specs}
               for f in feeds]
        self.last_timing = {"crops": len(feeds), "body_ms": 1.0}
        return out

    def infer_many(self, reqs):
        outs = [self.infer(f, model=m) for m, f in reqs]
        self.last_timing = {"requests": [{"crops": len(f), "body_ms": 1.0} for _, f in reqs], "body_ms": 2.0}
        return outs

    def idle(self):
        pass

    def close(self):
        pass


class CPU(S._Base):
    """the session's CPU stand-in: zeros, or a failure while `allowed` is False"""
    name, allowed = "cpu", False

    def max_crops(self, model):
        return 4

    def infer(self, model, crops):
        if not CPU.allowed:
            raise AssertionError("CPU used for " + model)
        e = self.entries[model]
        return [{n: np.zeros(sh, np.float32) for n, sh in e.shapes.items()} for _ in crops]


def model_dir(root, name):
    doc = json.loads((Path(S.BUNDLED) / f"{name}.json").read_text())
    d = Path(root) / name
    d.mkdir()
    blob = name.encode()
    (d / doc["weights"]["file"]).write_bytes(blob)
    doc["weights"]["sha256"] = hashlib.sha256(blob).hexdigest()
    (d / "model.json").write_text(json.dumps(doc))
    return d


class StereoFrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "npu.sock")
        self.srv = service.Server("model", "jobs", self.path, os.getuid(), session_factory=FakeDSP,
                                  s3d_model="s3d.onnx", crops=4)
        self.t = threading.Thread(target=self.srv.run, daemon=True)
        self.t.start()
        for _ in range(100):
            if os.path.exists(self.path):
                break
            time.sleep(0.01)
        CPU.allowed = False

    def tearDown(self):
        self.srv.stop = True
        self.t.join(5)
        self.tmp.cleanup()

    def test_stereo_frame_is_one_round_trip_and_matches_per_hand_requests(self):
        from handtrack import mpworker, stereo3d
        from handtrack.mercury_kp import KeypointView

        session = framenpu.open([model_dir(self.tmp.name, KP), model_dir(self.tmp.name, S3D)], "service",
                                socket=self.path, _cpu=lambda entries, threads: CPU(entries))
        client = session.accelerator.client
        calls, orig = [], client.infer_many
        client.infer_many = lambda reqs, **k: calls.append([(m, len(f)) for m, f in reqs]) or orig(reqs, **k)

        class MK:
            def __init__(self):
                self.session, self.backend = session, KeypointView(session, KP)

            def _prepare(self, sl, img, jc, right, use_hint, expand, raw):
                return (sl, None, (1.0, 0.0, 0.0, 0.0), 0.2, right, feed(float(sl * 10 + img)))

            def _decode(self, prep, o):
                return {"v": float(np.ravel(o["heatmap_xy"])[0])}

        mk, s3 = MK(), stereo3d.SessionStereo3D(session, S3D)
        P = np.random.default_rng(1).normal(0, 0.03, (21, 3)) + [0, 0, 0.4]
        jobs = [("left", {0: P, 1: P}, 1.0, True), ("right", {0: P + 0.01, 1: P + 0.01}, 1.0, False)]
        imgs = {0: 1, 1: 2}
        old = (mpworker.init_mercury, mpworker.init_stereo3d, mpworker._many_off, dict(mpworker._names))
        mpworker.init_mercury = lambda: mk
        mpworker.init_stereo3d = lambda: (s3, ([0, 1], np.eye(3), np.array([0.1, 0, 0])))
        mpworker._names.clear()
        mpworker._names.update(keypoint=KP, stereo3d=S3D)
        try:
            mpworker._many_off = False
            many = mpworker.run_stereo3d(jobs, imgs)
            self.assertEqual(calls, [[(1, 2), (0, 3), (0, 1)]])      # the whole frame, one round trip
            self.assertEqual({r[6]["inference"] for r in many} | {r[6]["s3d_inference"] for r in many},
                             {"framenpu"})
            mpworker._many_off = True
            single = mpworker.run_stereo3d(jobs, imgs)
            self.assertEqual(len(calls), 1)
            for a, b in zip(many, single):
                self.assertEqual((a[0], a[1], a[3]), (b[0], b[1], b[3]))
                np.testing.assert_array_equal(a[2], b[2])
            # a failed frame latches the session (both models) to the CPU, never retried there
            mpworker._many_off = False
            CPU.allowed = True
            client.infer_many = lambda reqs, **k: (_ for _ in ()).throw(RuntimeError("NPU service: device fault"))
            with self.assertLogs("framenpu.session", "WARNING"):
                out = mpworker.run_stereo3d(jobs, imgs)
            self.assertEqual(len(out), 2)
            self.assertEqual((mk.backend.name, s3.name), ("cpu", "cpu"))
            self.assertEqual({r[6]["inference"] for r in out}, {"cpu"})
            self.assertIsNone(mpworker._many_session(mk, s3))        # per hand on the CPU from here on
        finally:
            mpworker.init_mercury, mpworker.init_stereo3d, mpworker._many_off = old[:3]
            mpworker._names.clear()
            mpworker._names.update(old[3])
            session.close()


if __name__ == "__main__":
    unittest.main()
