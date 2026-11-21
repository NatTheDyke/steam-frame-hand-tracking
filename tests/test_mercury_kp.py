"""Mercury crop preparation (handtrack/mercury_kp.py): the restructured make_instructions and crop
map must equal the original formulas (kept here verbatim as references). Synthetic Kannala-Brandt
cameras, so no calibration file is needed."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
    from handtrack import mercury_kp as M
    from handtrack.calib import KBCamera
    from handtrack.source import Source
except ImportError as e:                               # cv2 lives on the headset
    M = None
    MISSING = repr(e)

SLAM = Source("slam", "/dev/null", {0: "slam_right", 1: "slam_left"}, (1056, 1024), (16, 0, 1024),
              1e9 / 30.0, angles={0: 124.9, 1: -124.2})


def ref_make_instructions(joints_m, flip, expand=1.7, twist=0.0):
    """The original make_instructions."""
    dir3 = joints_m[9] / np.linalg.norm(joints_m[9])
    q = M.direction(dir3, twist)
    old = dir3
    for _ in range(8):
        sg = M.project_unscaled(joints_m, q)
        mn, mx = sg.min(0), sg.max(0)
        c = (mn + mx) / 2
        r = max(c[0] - mn[0], c[1] - mn[1])
        new = M.q_rot(q, M.sg_unproject(c[0], c[1]))[0]
        q = M.direction(new, twist)
        if np.linalg.norm(old - new) < 1e-4:
            break
        old = new
    sg = M.project_unscaled(joints_m, q)
    fwd = np.linalg.norm(sg[0] - sg[9])
    side = np.linalg.norm(sg[5] - sg[17])
    palm_l = max(fwd, side)
    r = max(r * expand, palm_l * 0.5 * (2.2 / 1.65) * expand)
    kp = np.stack([sg[:, 0] / r * (-1 if flip else 1), -sg[:, 1] / r], 1)
    norms = np.linalg.norm(joints_m, axis=1)
    hand_size = np.linalg.norm(joints_m[0] - joints_m[9])
    rel = (norms - norms[9]) / max(hand_size, 1e-6)
    return q, r, kp.astype(np.float32), rel


def ref_crop_pixels(mk, slot, q, r, flip):
    """The original crop's ray construction and projection."""
    W = M.WSIZE
    sx = (-r + 2 * r * mk.gx / W) if not flip else (r - 2 * r * mk.gx / W)
    sy = r - 2 * r * mk.gy / W
    d = np.stack([2 * sx, 2 * sy, sx * sx + sy * sy - 1], -1).reshape(-1, 3)
    d = M.q_rot(q, d)
    d[:, 1] *= -1
    d[:, 2] *= -1
    return mk.cam[slot].project(d).reshape(W, W, 2)


def camera(name, fx, cx, cy, k):
    return {"sourceCamera": name, "width": 1056, "height": 1024,
            "intrinsics": {"fx": fx, "fy": fx * 1.001, "cx": cx, "cy": cy,
                           "k1": k[0], "k2": k[1], "k3": k[2], "k4": k[3]},
            "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}}


class Rig:
    cams = {"slam_right": KBCamera(camera("slam_right", 280.0, 524.0, 512.0, (0.02, -0.004, 0.001, -0.0002))),
            "slam_left": KBCamera(camera("slam_left", 282.0, 522.0, 510.0, (0.018, -0.003, 0.0008, -0.0001)))}


def hands(rng, n):
    for _ in range(n):
        c = np.array([rng.uniform(-.3, .3), rng.uniform(-.15, .35), rng.uniform(.15, .6)])
        yield (c + rng.normal(0, .04, (21, 3))) * np.array([1, -1, -1]), bool(rng.integers(2)), int(rng.integers(2))


@unittest.skipIf(M is None, "needs cv2 (headset): %s" % (None if M else MISSING))
class CropPreparation(unittest.TestCase):
    def setUp(self):
        self.mk = M.MercuryKeypoint(None, Rig(), SLAM.slot_cams, 640, backend=object(), source=SLAM)

    def test_q_matrix_rotates_like_q_rot(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            v = rng.normal(size=(7, 3))
            np.testing.assert_allclose(v @ M.q_matrix(q).T, M.q_rot(q, v), atol=1e-14)

    def test_make_instructions_matches_original(self):
        rng = np.random.default_rng(2)
        for jm, flip, _ in hands(rng, 400):
            for twist in (0.0, 0.3):
                a, b = M.make_instructions(jm, flip, twist=twist), ref_make_instructions(jm, flip, twist=twist)
                np.testing.assert_allclose(a[0], b[0], atol=1e-13)
                self.assertAlmostEqual(a[1], b[1], delta=1e-13)
                np.testing.assert_allclose(a[2], b[2], atol=1e-6)
                np.testing.assert_array_equal(a[3], b[3])

    def test_direction_matches_original(self):
        for v in ([0, 0, -1], [0, 0, 1], [1e-9, 0, 1], [0.3, -0.2, -0.9], [0.5, 0.5, 0.1]):
            v = np.array(v, float)
            for twist in (0.0, 1.0):
                np.testing.assert_allclose(M._direction(v / np.linalg.norm(v), twist),
                                           M.direction(v / np.linalg.norm(v), twist), atol=1e-14)

    def test_crop_pixels_match_original(self):
        rng = np.random.default_rng(3)
        worst = 0.0
        for jm, flip, slot in hands(rng, 200):
            q, r, _, _ = M.make_instructions(jm, flip)
            a, b = self.mk.crop_pixels(slot, q, r, flip), ref_crop_pixels(self.mk, slot, q, r, flip)
            worst = max(worst, float(np.abs(a - b).max()))
        self.assertLess(worst, 1e-8)

    def test_crops_match_original(self):
        rng = np.random.default_rng(4)
        raw = cv2.GaussianBlur(rng.integers(0, 255, (1024, 1056), dtype=np.uint8), (0, 0), 2)
        det = SLAM.det_image(raw, 0, 640)
        same = 0
        for jm, flip, slot in hands(rng, 200):
            q, r, _, _ = M.make_instructions(jm, flip)
            for img, is_raw in ((raw, True), (det, False)):
                px = ref_crop_pixels(self.mk, slot, q, r, flip)
                f = img.shape[1] / SLAM.raw_w
                m = ((px + 0.5) * f - 0.5) if is_raw else self.mk.raw_to_det(px, slot)
                m = m.astype(np.float32)
                ref = cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                got = self.mk.crop(slot, img, q, r, flip, raw=is_raw)
                self.assertLessEqual(int(np.abs(got.astype(int) - ref.astype(int)).max()), 1)
                same += np.array_equal(got, ref)
        self.assertGreaterEqual(same, 396)          # float64 rounding may move a sample across 1/32 px


if __name__ == "__main__":
    unittest.main()
