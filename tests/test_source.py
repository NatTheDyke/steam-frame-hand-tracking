"""Camera-source geometry (handtrack/source.py): passthrough must match the original formulas exactly,
slam (arbitrary-angle rotation, non-square sensor) must be self-consistent with its images."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtrack import source as camsource
from handtrack.palm import det_to_raw as old_det_to_raw
from handtrack.source import Source, upright_angle

S = 640
PT = camsource.SOURCES["passthrough"]
# slam geometry with the calibrated angles of the test unit (no calibration file needed here)
SLAM = Source("slam", "/dev/null", {0: "slam_right", 1: "slam_left"}, (1056, 1024), (16, 0, 1024),
              1e9 / 30.0, angles={0: 124.9, 1: -124.2})


def old_raw_to_det(raw, slot, S, raw_size=2464):
    """The original passthrough formula (palm.det_px / MercuryKeypoint.raw_to_det before sources)."""
    k = raw_size / S
    x = (raw[:, 0] + 0.5) / k - 0.5
    y = (raw[:, 1] + 0.5) / k - 0.5
    return np.stack([S - 1 - y, x], 1) if slot == 0 else np.stack([y, S - 1 - x], 1)


class PassthroughUnchanged(unittest.TestCase):
    def test_raw_to_det_matches_original(self):
        raw = np.random.default_rng(1).uniform(0, 2464, (200, 2))
        for slot in (0, 1):
            np.testing.assert_allclose(PT.raw_to_det(raw, slot, S), old_raw_to_det(raw, slot, S), atol=1e-9)

    def test_det_to_raw_matches_original(self):
        det = np.random.default_rng(2).uniform(0, S, (200, 2))
        for slot in (0, 1):
            np.testing.assert_allclose(PT.det_to_raw(det, slot, S), old_det_to_raw(det, slot, S), atol=1e-9)

    def test_det_image_matches_original(self):
        img = np.random.default_rng(3).integers(0, 255, (S, S), np.uint8)   # ptgrab publishes 640x640
        for slot, code in ((0, cv2.ROTATE_90_CLOCKWISE), (1, cv2.ROTATE_90_COUNTERCLOCKWISE)):
            np.testing.assert_array_equal(PT.det_image(img, slot, S), cv2.rotate(img, code))


class SlamGeometry(unittest.TestCase):
    def test_roundtrip(self):
        raw = np.random.default_rng(4).uniform([16, 0], [1040, 1024], (500, 2))
        for slot in (0, 1):
            back = SLAM.det_to_raw(SLAM.raw_to_det(raw, slot, S), slot, S)
            np.testing.assert_allclose(back, raw, atol=1e-9)

    def test_image_follows_mapping(self):
        """A bright spot at a raw pixel shows up in the detector image where raw_to_det says."""
        for slot in (0, 1):
            for p in ((300.0, 200.0), (700.0, 650.0), (528.0, 512.0), (200.0, 800.0)):
                img = np.zeros((1024, 1056), np.uint8)
                cv2.circle(img, (int(p[0]), int(p[1])), 12, 255, -1)
                det = SLAM.det_image(img, slot, S).astype(float)
                ys, xs = np.nonzero(det > 40)
                w = det[ys, xs]
                found = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
                want = SLAM.raw_to_det(np.array([p]), slot, S)[0]
                self.assertLess(np.linalg.norm(found - want), 1.0, (slot, p, found, want))

    def test_det_image_is_square(self):
        img = np.zeros((1024, 1056), np.uint8)
        for slot in (0, 1):
            self.assertEqual(SLAM.det_image(img, slot, 512).shape, (512, 512))

    def test_upright_angle(self):
        """Camera rolled by phi about its optical axis -> angle that undoes it."""
        class Rig:
            def __init__(self, R): self.R = R
            def head_from_cam(self, name): return self.R, np.zeros(3)
        for want in (90.0, -90.0, 125.0, -124.0, 0.0):
            a = np.radians(want)
            # camera x/y axes in head frame such that head-down (-Y) appears at angle `want` from image +y
            down_cam = np.array([np.sin(a), np.cos(a), 0.0])           # head-down seen in camera coords
            x_head = np.array([down_cam[1], -down_cam[0], 0.0])        # pick camera axes consistent with it
            R = np.zeros((3, 3))
            # R maps camera -> head; we need R.T @ (0,-1,0) = down_cam, and orthonormal
            R[1, :] = -down_cam
            R[0, :] = np.cross(R[1, :], [0, 0, 1.0])
            R[2, :] = np.cross(R[0, :], R[1, :])
            self.assertAlmostEqual(upright_angle(Rig(R), "x"), want, places=6)


class ActiveSource(unittest.TestCase):
    def test_env_selects_source(self):
        import os
        old = os.environ.get(camsource.ENV)
        try:
            os.environ[camsource.ENV] = "slam"
            self.assertEqual(camsource.active().name, "slam")
            os.environ.pop(camsource.ENV)
            self.assertEqual(camsource.active().name, "passthrough")
            os.environ[camsource.ENV] = "bogus"
            with self.assertRaises(ValueError):
                camsource.active()
        finally:
            if old is None:
                os.environ.pop(camsource.ENV, None)
            else:
                os.environ[camsource.ENV] = old


if __name__ == "__main__":
    unittest.main()
