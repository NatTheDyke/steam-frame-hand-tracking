"""Mercury-detector discovery (handtrack/mercdet.py): crops, decode, two-camera agreement and the
triangulated seed, on synthetic Kannala-Brandt cameras posed like the SLAM pair (no calibration
file, no model)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2
    from handtrack import mercdet as D
    from handtrack import mercury_kp as MK
    from handtrack.calib import KBCamera
    from handtrack.source import Source
except ImportError as e:                               # cv2 lives on the headset
    D = None
    MISSING = repr(e)

SLAM = Source("slam", "/dev/null", {0: "slam_right", 1: "slam_left"}, (1056, 1024), (16, 0, 1024),
              1e9 / 30.0, angles={0: 124.9, 1: -124.2}) if D is not None else None


def camera(name, fx, cx, cy, k):
    return {"sourceCamera": name, "width": 1056, "height": 1024,
            "intrinsics": {"fx": fx, "fy": fx * 1.001, "cx": cx, "cy": cy,
                           "k1": k[0], "k2": k[1], "k3": k[2], "k4": k[3]},
            "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}}


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


class Rig:
    """Two fisheye cameras 10 cm apart, toed out and pitched down a little (head = R cam + t)."""

    def __init__(self):
        self.cams = {"slam_right": KBCamera(camera("slam_right", 280.0, 524.0, 512.0, (0.02, -0.004, 0.001, -0.0002))),
                     "slam_left": KBCamera(camera("slam_left", 282.0, 522.0, 510.0, (0.018, -0.003, 0.0008, -0.0001)))}
        self.poses = {"slam_right": (rot((0, 1, 0), 12) @ rot((1, 0, 0), 20), np.array([0.05, -0.02, 0.0])),
                      "slam_left": (rot((0, 1, 0), -12) @ rot((1, 0, 0), 20), np.array([-0.05, -0.02, 0.0]))}

    def head_from_cam(self, name):
        return self.poses[name]


def det_px(geom, slot, X):
    """head-frame points (N, 3) -> FULL det pixels in this camera"""
    R, t = geom.pose[slot]
    raw = geom.cam[slot].project((np.atleast_2d(X) - t) @ R)
    return geom.src.raw_to_det(raw, slot, D.FULL)


def hand(rng, X, size=0.09):
    """21 joints around palm centre X, wrist -> middle knuckle `size` (palm centre = mean of joints
    0, 5, 9, 17)"""
    P = rng.normal(0, 0.035, (21, 3))
    P *= size / np.linalg.norm(P[0] - P[9])
    return P - P[list(D.PALM)].mean(0) + X


@unittest.skipIf(D is None, "needs cv2 (headset): %s" % (None if D else MISSING))
class MercDetTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.geom = D.Geometry(SLAM, Rig())

    def points(self, n):
        """head-frame points both cameras see, 0.2-0.7 m away"""
        out = []
        while len(out) < n:
            X = np.array([self.rng.uniform(-.25, .25), self.rng.uniform(-.35, .05), self.rng.uniform(.2, .6)])
            ok = all(((X - self.geom.pose[s][1]) @ self.geom.pose[s][0])[2] > 0.1 for s in (0, 1))
            if ok and all(((0 < det_px(self.geom, s, X)) & (det_px(self.geom, s, X) < D.FULL)).all() for s in (0, 1)):
                out.append(X)
        return out

    def test_crops_and_feeds(self):
        up = self.rng.integers(0, 255, (D.FULL, D.FULL), np.uint8)
        x, y, w = D.LAYOUTS[2][1]
        want = cv2.resize(up[y:y + w, x:x + w], (D.DET, D.DET), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(D.crop(up, x, y, w), want)
        edge = D.crop(up, 900, -50, 200)                       # zero outside the image
        self.assertEqual(edge.shape, (D.DET, D.DET))
        frame = self.rng.integers(0, 255, (1024, 1056), np.uint8)
        fs = D.feeds(SLAM, 0, frame, D.LAYOUTS[4])
        self.assertEqual([i for i, _ in fs], [0, 1, 2, 3])
        self.assertEqual([i for i, _ in D.feeds(SLAM, 1, frame, D.LAYOUTS[4], [2])], [2])
        # each crop is rotated alone, bit-identical to cropping the whole upright image
        for slot in (0, 1):
            up = SLAM.det_image(frame, slot, D.FULL)
            for x, y, w in D.LAYOUTS[4]:
                np.testing.assert_array_equal(D.upright_region(SLAM, slot, frame, x, y, w), D.crop(up, x, y, w))
            np.testing.assert_array_equal(D.upright_region(SLAM, slot, frame, 900, -50, 200), D.crop(up, 900, -50, 200))
        v = fs[0][1]["inputImg"]
        self.assertEqual((v.shape, v.dtype), ((1, 1, D.DET, D.DET), np.float32))
        self.assertAlmostEqual(float(v.std()), 0.25, places=4)
        self.assertAlmostEqual(float(v.mean()), 0.5, places=4)
        self.assertEqual(D.feeds(SLAM, 0, np.zeros((1024, 1056), np.uint8), D.LAYOUTS[2]), [])

    def test_decode(self):
        lay = D.LAYOUTS[2]
        o = {"hand_exists": np.array([[0.9, 0.2]]), "cx": np.array([[0.5, 0.0]]), "cy": np.array([[-0.5, 0.0]]),
             "size": np.array([[0.25, 0.3]])}
        a, = D.decode(1, [1], [o], lay)
        x, y, w = lay[1]
        self.assertEqual((a.slot, a.crop, a.side), (1, 1, "left"))
        np.testing.assert_allclose(a.c, [x + 0.75 * w, y + 0.25 * w])
        self.assertAlmostEqual(a.size, D.SIZE_GAIN * 0.25 * w)
        self.assertEqual(len(D.decode(1, [1], [o], lay, thr=0.1)), 2)

    def answer(self, slot, side, X, p=0.9, jitter=0.0):
        c = det_px(self.geom, slot, X)[0] + self.rng.normal(0, jitter, 2)
        return D.Answer(slot, 0, side, p, c, 120.0)

    def test_confirm_recovers_the_point(self):
        for X in self.points(20):
            m = D.confirm([self.answer(0, "left", X), self.answer(1, "left", X)], self.geom)
            self.assertEqual(list(m), ["left"])
            np.testing.assert_allclose(m["left"].X, X, atol=1e-6)
            self.assertLess(m["left"].gap, 1e-6)

    def test_confirm_rejects_disagreement(self):
        X, Y = self.points(2)
        g = self.geom
        self.assertEqual(D.confirm([self.answer(0, "left", X), self.answer(1, "right", X)], g), {})   # sides differ
        far = X + np.array([0.0, 0.12, 0.0])
        self.assertEqual(D.confirm([self.answer(0, "right", X), self.answer(1, "right", far)], g), {})
        near = X * (0.1 / np.linalg.norm(X))                                                       # too close to the head
        self.assertEqual(D.confirm([D.Answer(0, 0, "left", .9, det_px(g, 0, near)[0], 99.),
                                    D.Answer(1, 0, "left", .9, det_px(g, 1, near)[0], 99.)], g), {})
        # the best pair per side wins; a distractor in one camera is ignored
        m = D.confirm([self.answer(0, "right", X), self.answer(0, "right", Y), self.answer(1, "right", X, jitter=1.0)], g)
        # (1 px at 0.5 m over this 10 cm baseline is ~2 cm of depth)
        self.assertLess(np.linalg.norm(m["right"].X - X), 0.04)
        self.assertLess(np.linalg.norm(m["right"].X - X), np.linalg.norm(m["right"].X - Y))
        # both sides on one physical hand: the more probable one stays
        m = D.confirm([self.answer(0, "left", X, 0.6), self.answer(1, "left", X, 0.6),
                       self.answer(0, "right", X, 0.9), self.answer(1, "right", X, 0.95)], g)
        self.assertEqual(list(m), ["right"])
        m = D.confirm([self.answer(0, "left", X), self.answer(1, "left", X),
                       self.answer(0, "right", Y), self.answer(1, "right", Y)], g)
        self.assertEqual(sorted(m), ["left", "right"])

    def test_untracked_drops_matches_on_tracked_hands(self):
        X, Y = self.points(2)
        m = D.confirm([self.answer(0, "left", X), self.answer(1, "left", X),
                       self.answer(0, "right", Y), self.answer(1, "right", Y)], self.geom)
        self.assertEqual(sorted(D.untracked(m, self.geom, {})), ["left", "right"])
        # any tracked hand at the answer counts, whatever its side; 2 cm away still is that hand
        self.assertEqual(list(D.untracked(m, self.geom, {"right": X + [0.02, 0, 0]})), ["right"])
        self.assertEqual(D.untracked(m, self.geom, {"left": X, "right": Y}), {})
        # a tracked palm far from the answer in one camera's image is another hand
        a1 = m["left"].a1
        far = m["left"]._replace(a1=a1._replace(c=a1.c + [1.2 * a1.size, 0]))
        self.assertEqual(list(D.untracked({"left": far}, self.geom, {"left": X})), ["left"])

    def test_seed_crop_is_centred_and_sized(self):
        mk = MK.MercuryKeypoint(None, Rig(), SLAM.slot_cams, 640, backend=object(), source=SLAM)
        for X in self.points(10):
            for slot in (0, 1):
                c = det_px(self.geom, slot, X)[0]
                q, r = D.seed_crop(self.geom, slot, c, 100.0)
                px = mk.crop_pixels(slot, q, r, False)
                mid = (px[63, 63] + px[63, 64] + px[64, 63] + px[64, 64]) / 4
                np.testing.assert_allclose(SLAM.raw_to_det(mid, slot, D.FULL), c, atol=1.0)
                # the crop's half width spans k x size / 2 (angle at the centre ray)
                rays = self.geom.cam[slot].unproject(SLAM.det_to_raw(np.array([c, c + [D.SEED_K * 50.0, 0]]), slot, D.FULL))
                self.assertAlmostEqual(np.arctan(r) * 2, np.arccos(rays[0] @ rays[1]), places=9)

    def test_seed_inputs_are_unhinted(self):
        mk = MK.MercuryKeypoint(None, Rig(), SLAM.slot_cams, 640, backend=object(), source=SLAM)
        X, = self.points(1)
        m = D.Match("right", X, 0.0, self.answer(0, "right", X), self.answer(1, "right", X))
        frames = {s: self.rng.integers(0, 255, (1024, 1056), np.uint8) for s in (0, 1)}
        prep = D.seed_inputs(mk, self.geom, m, frames)
        self.assertEqual([p[0] for p in prep], [0, 1])
        for slot, img, q, r, flip, f in prep:
            self.assertTrue(flip)
            self.assertEqual(f["inputImg"].shape, (1, 1, 128, 128))
            self.assertFalse(f["useLastKeypoints"].any() or f["lastKeypoints"].any())
        blank = {s: np.zeros((1024, 1056), np.uint8) for s in (0, 1)}
        self.assertIsNone(D.seed_inputs(mk, self.geom, m, blank))

    def test_seed_rejects_an_implausible_hand_size(self):
        g = self.geom
        for size, ok in ((0.05, False), (0.065, True), (0.12, True), (0.15, False)):
            X, = self.points(1)
            P = hand(self.rng, X, size)
            dec = []
            for s in (0, 1):
                R, t = g.pose[s]
                d = (P - t) @ R
                dec.append({"rays": d / np.linalg.norm(d, axis=1, keepdims=True)})
            m = D.Match("left", X, 0.0, self.answer(0, "left", X), self.answer(1, "left", X))
            self.assertEqual(D.seed_joints(g, m, *dec) is not None, ok, size)

    def test_seed_joints_triangulate_both_views(self):
        g = self.geom
        for X in self.points(10):
            P = hand(self.rng, X)
            dec = []
            for s in (0, 1):
                R, t = g.pose[s]
                d = (P - t) @ R
                dec.append({"rays": d / np.linalg.norm(d, axis=1, keepdims=True)})
            # the detector's point is poor in depth: a match 8 cm off along camera 0's ray still seeds
            o0, r0 = g.rays(0, det_px(g, 0, X))
            Xd = X + r0[0] * 0.08
            m = D.Match("left", Xd, 0.0, self.answer(0, "left", X), self.answer(1, "left", X))
            np.testing.assert_allclose(D.seed_joints(g, m, *dec), P, atol=1e-9)
            # views that disagree across the epipolar direction (their rays miss) seed nothing
            bad = {"rays": dec[1]["rays"] @ rot((1, 0, 0), 8).T}
            self.assertIsNone(D.seed_joints(g, m, dec[0], bad))
            # nor a hand away from either camera's answer (more than 0.75 hand size in its image)
            a1 = self.answer(1, "left", X)
            far = D.Match("left", X, 0.0, self.answer(0, "left", X), a1._replace(c=a1.c + [0.8 * a1.size, 0]))
            self.assertIsNone(D.seed_joints(g, far, *dec))
            near = D.Match("left", X, 0.0, self.answer(0, "left", X), a1._replace(c=a1.c + [0.6 * a1.size, 0]))
            self.assertIsNotNone(D.seed_joints(g, near, *dec))


if __name__ == "__main__":
    unittest.main()
