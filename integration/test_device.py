"""The service, its client and the CDSP backend together, on framenpu-cdsp's lane simulator with the
real weights: `framenpu serve` as installed, and framenpu.open's three backends against each other.

    FI_MODELS=<dir>  FI_CDSP_TESTS=<framenpu-cdsp checkout>/tests  (the simulator is not in the wheel)"""
import getpass
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

D, T = os.environ.get("FI_MODELS"), os.environ.get("FI_CDSP_TESTS")
KP, DET = (str(Path(D or "") / n) for n in ("student_ft7.onnx", "grayscale_detection_160x160.onnx"))
NAMES = ("mercury-ft7", "mercury-detector-160")
# The aarch64 native host kernels sum some heads in another order than NumPy; framenpu-cdsp's
# tests/test_native_host.py holds them to 1e-5, and these outputs to equality.
NATIVE, EXACT = platform.machine() == "aarch64", ("heatmap_xy", "scalar_extras", "curls")
WHY = "FI_MODELS and FI_CDSP_TESTS not set" if not (D and T) else None


def keypoint_crops(rng, n):
    return [{"inputImg": rng.uniform(-1, 2, (1, 1, 128, 128)).astype(np.float32),
             "lastKeypoints": rng.uniform(0, 1, (1, 42)).astype(np.float32),
             "useLastKeypoints": np.array([float(k % 2)], np.float32)} for k in range(n)]


def detector_crops(rng, n):
    return [{"inputImg": rng.uniform(-1, 2, (1, 1, 160, 160)).astype(np.float32)} for _ in range(n)]


if WHY and os.environ.get("FI_REQUIRED"):              # CI: a skip would look like a pass
    raise RuntimeError(WHY)


@unittest.skipIf(WHY, WHY)
class DeviceThroughTheService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="fh")
        cls.sock = os.path.join(cls.tmp.name, "npu.sock")
        cls.env = dict(os.environ, FRAMENPU_DEVICES="cdsp=sim_plugin:SimPlugin", PYTHONPATH=os.pathsep.join([str(Path(T).resolve())] + os.environ.get("PYTHONPATH", "").split(os.pathsep)).rstrip(os.pathsep))
        cls.log = open(os.path.join(cls.tmp.name, "log"), "w+")
        cls.proc = subprocess.Popen([os.environ.get("FI_SERVICE_PYTHON") or sys.executable, "-m", "framenpu", "serve",
                                     "--model", KP, "--det-model", DET, "--unpinned", "--user", getpass.getuser(),
                                     "--socket", cls.sock], env=cls.env, stdout=cls.log, stderr=subprocess.STDOUT)
        end = time.monotonic() + 30
        while not os.path.exists(cls.sock) and cls.proc.poll() is None and time.monotonic() < end:
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(20)
        cls.log.close()
        cls.tmp.cleanup()

    def open(self, backend):
        import framenpu
        s = framenpu.open([(KP, NAMES[0]), (DET, NAMES[1])], backend, **({"socket": self.sock} if backend == "service" else {}))
        self.addCleanup(s.close)
        return s

    def text(self):
        self.log.flush()
        return Path(self.log.name).read_text()

    def test_the_service_answers_bit_for_bit_what_the_fp16_contract_says(self):
        from framenpu import mercury
        s = self.open("service")
        self.assertEqual([s.backend_of(m) for m in NAMES], ["framenpu"] * 2, self.text())
        self.assertEqual([s.max_crops(m) for m in NAMES], [4, 2])
        rng = np.random.default_rng(11)
        graph, dgraph = mercury.load_keypoint(KP, require_pin=False), mercury.load_detector(DET)
        kp, det = keypoint_crops(rng, 4), detector_crops(rng, 2)

        def check(name, crops, outs):
            g = graph if name == NAMES[0] else dgraph
            self.assertEqual(len(outs), len(crops))
            for feed, out in zip(crops, outs):
                want = mercury.run(g, feed, "fp16")
                for k in out:
                    msg = "%s %s" % (name, k)
                    if NATIVE and k not in EXACT:
                        np.testing.assert_allclose(out[k], want[k], rtol=0, atol=1e-5, err_msg=msg)
                    else:
                        np.testing.assert_array_equal(out[k], want[k], err_msg=msg)
        for n in (1, 4):
            check(NAMES[0], kp[:n], s.infer(NAMES[0], kp[:n]))
        check(NAMES[1], det, s.infer(NAMES[1], det))
        for call in (s.infer_frame, s.infer_many):
            a, b = call([(NAMES[0], kp[:2]), (NAMES[1], det)])
            check(NAMES[0], kp[:2], a)
            check(NAMES[1], det, b)
        s.idle()
        check(NAMES[0], kp[:1], s.infer(NAMES[0], kp[:1]))
        self.assertEqual([s.backend_of(m) for m in NAMES], ["framenpu"] * 2, self.text())

    def test_framehands_splits_what_one_request_cannot_carry(self):
        """Six observations through framehands and the service: 4 + 2 crops, and the same answers as
        one at a time (nothing padded, nothing reordered)."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import world
        from framehands import source
        from framehands.calib import Rig
        from framehands.mercury_kp import MercuryKeypoint
        path = os.path.join(self.tmp.name, "c.json")
        world.write_calibration(path)
        rig, src = Rig(path), source.SOURCES["slam"]
        src._angles = {k: round(source.upright_angle(rig, n), 1) for k, n in src.slot_cams.items()}
        hands = world.truth(0)
        imgs, obs = world.render(hands), []
        for i in range(6):
            slot, side = i % 2, ("left", "right")[(i // 2) % 2]
            Rm, t = rig.head_from_cam(src.slot_cams[slot])
            obs.append(dict(slot=slot, det_img=imgs[slot], raw=True, joints_cv=(hands[side] - t) @ Rm, flip=side == "right",
                            use_hint=i < 4, expand=1.6 + 0.1 * i))
        s = self.open("service")
        sent, infer = [], s.infer
        s.infer = lambda m, c: (sent.append(len(c)), infer(m, c))[1]
        mk = MercuryKeypoint(KP, rig, src.slot_cams, 640, source=src, session=s, model=NAMES[0], cell_offset=0)
        many = mk.run_many(obs)
        self.assertEqual(sent, [4, 2], self.text())
        for a, o in zip(many, obs):
            b = mk.run_many([o])[0]
            for k in ("rays", "conf", "rel_depth"):
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        self.assertEqual(s.backend_of(NAMES[0]), "framenpu", self.text())

    def test_the_backends_agree_on_what_a_hand_is(self):
        """CPU (fp32 ONNX) against the device's FP16 arithmetic: the same heatmap peaks"""
        rng = np.random.default_rng(3)
        crops = keypoint_crops(rng, 3)
        a, b = self.open("cpu").infer(NAMES[0], crops), self.open("service").infer(NAMES[0], crops)
        for x, y in zip(a, b):
            self.assertEqual(set(x), set(y))
            for k in x:
                self.assertEqual(x[k].shape, y[k].shape)
                scale = np.abs(x[k]).max() + 1e-6
                self.assertLess(np.abs(x[k] - y[k]).max() / scale, 0.05, k)

    def test_crops_over_the_limit_are_refused_without_losing_the_device(self):
        s = self.open("service")
        crops = keypoint_crops(np.random.default_rng(1), 5)
        with self.assertRaises(ValueError):
            s.infer(NAMES[0], crops)
        s.infer(NAMES[0], crops[:4])
        self.assertEqual(s.backend_of(NAMES[0]), "framenpu", self.text())


if __name__ == "__main__":
    unittest.main()
