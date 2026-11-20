"""Bridge wire format: v4 adds per-hand posAgeMs / rotAgeMs; v2 / v3 bodies stay byte-identical."""
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtrack import palm


class FakeSock:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((addr, data))


def bridge_with(subs):
    b = object.__new__(palm.Bridge)
    b.sock, b.seq, b.enabled_mask = FakeSock(), 0, 3
    import threading
    import time
    b.lock = threading.Lock()
    b.splay = False
    b.splay_filt = {sd: palm.OneEuro(1.0, 1.0) for sd in ("left", "right")}
    b.subs = {addr: (time.monotonic(), ver) for addr, ver in subs.items()}
    return b


POSE = (np.array([0.1, -0.2, -0.4]), np.array([1.0, 0, 0, 0]), 0.9, None, np.linspace(0, 1, 5))


class BridgeTests(unittest.TestCase):
    def test_sizes_match_driver_struct(self):
        # pcdriver/src/framehands_proto.h: FhHand v4 = 88 bytes, FhHandPacket v4 = 208 bytes
        self.assertEqual(palm._HAND4.size, 88)
        self.assertEqual(palm._PKT_HEAD[4].size + 2 * palm._HAND4.size, 208)
        self.assertEqual(palm._PKT_HEAD[3].size + 2 * palm._HAND.size, 192)   # v3 driver struct

    def test_v4_carries_ages_and_v3_is_unchanged(self):
        b = bridge_with({("a", 1): 3, ("b", 2): 4})
        b.send({"left": None, "right": POSE}, 123456789, {"right": (40.0, 85.5)})
        sent = dict(b.sock.sent)
        v3, v4 = sent[("a", 1)], sent[("b", 2)]
        self.assertEqual(len(v3), 192)
        self.assertEqual(len(v4), 208)
        self.assertEqual(struct.unpack_from("<I", v4, 4)[0], 4)
        right4 = palm._HAND4.unpack_from(v4, palm._PKT_HEAD[4].size + palm._HAND4.size)
        self.assertEqual(right4[0], 1)
        self.assertAlmostEqual(right4[-2], 40.0, places=4)
        self.assertAlmostEqual(right4[-1], 85.5, places=4)
        # the v4 hand without its two ages is exactly the v3 hand
        right3 = palm._HAND.unpack_from(v3, palm._PKT_HEAD[3].size + palm._HAND.size)
        self.assertEqual(right3, right4[:-2])

    def test_negative_or_missing_ages_are_zero(self):
        b = bridge_with({("b", 2): 4})
        b.send({"left": POSE, "right": POSE}, 1, {"left": (-5.0, -1.0)})
        v4 = b.sock.sent[0][1]
        h = palm._PKT_HEAD[4].size
        left = palm._HAND4.unpack_from(v4, h)
        right = palm._HAND4.unpack_from(v4, h + palm._HAND4.size)
        self.assertEqual(left[-2:], (0.0, 0.0))
        self.assertEqual(right[-2:], (0.0, 0.0))


    def test_splay_off_sends_zeros_and_on_sends_joint_splay(self):
        P = open_hand()
        pose = POSE[:3] + (P,) + POSE[4:]
        b = bridge_with({("b", 2): 4})
        b.send({"left": None, "right": pose}, 10_000_000, {})
        right = palm._HAND4.unpack_from(b.sock.sent[0][1], palm._PKT_HEAD[4].size + palm._HAND4.size)
        self.assertEqual(right[15:20], (0.0,) * 5)               # splay fields: 0 unless --splay
        b.splay = True
        b.send({"left": None, "right": pose}, 20_000_000, {})
        right = palm._HAND4.unpack_from(b.sock.sent[1][1], palm._PKT_HEAD[4].size + palm._HAND4.size)
        np.testing.assert_allclose(right[15:20], palm.finger_splays(P), atol=1e-6)


def open_hand(spread_deg=(15.0, -6.0, -13.0, -30.0)):
    """Flat hand in the x-y plane, wrist at the origin, fingers along +y, thumb side +x; each finger's
    proximal bone at spread_deg from +y (+ = toward the thumb)."""
    P = np.zeros((21, 3))
    P[1:5] = [[0.03, 0.02, 0], [0.045, 0.04, 0], [0.055, 0.06, 0], [0.06, 0.08, 0]]
    for k, (x, sp) in enumerate(zip((0.025, 0.0, -0.02, -0.04), spread_deg)):
        m = 5 + 4 * k
        P[m] = [x, 0.09, 0]
        d = np.array([np.sin(np.radians(sp)), np.cos(np.radians(sp)), 0])
        for j in range(1, 4):
            P[m + j] = P[m] + d * 0.025 * j
    return P


class SplayTests(unittest.TestCase):
    def test_relaxed_hand_is_zero_and_spread_is_positive_index_negative_pinky(self):
        np.testing.assert_allclose(palm.finger_splays(open_hand((0.0, -6.0, -13.0, -15.0))), np.zeros(5), atol=1e-6)
        s = palm.finger_splays(open_hand((15.0, -6.0, -13.0, -30.0)))
        self.assertAlmostEqual(s[1], 1.0, places=5)              # index 15 deg toward the thumb
        self.assertAlmostEqual(s[4], -1.0, places=5)             # pinky 15 deg further out
        self.assertAlmostEqual(s[2], 0.0, places=5)

    def test_mirrored_hand_gives_the_same_splay(self):
        P = open_hand((10.0, -6.0, -20.0, -25.0))
        M = P * np.array([-1.0, 1.0, 1.0])                       # left/right mirror
        np.testing.assert_allclose(palm.finger_splays(M), palm.finger_splays(P), atol=1e-9)

    def test_finger_curled_out_of_the_palm_plane_fades_to_zero(self):
        P = open_hand((15.0, -6.0, -13.0, -15.0))
        P[6:9] = P[5] + np.array([[0.002, 0.001, 0.025], [0.004, 0.002, 0.05], [0.006, 0.003, 0.075]])
        self.assertEqual(palm.finger_splays(P)[1], 0.0)


if __name__ == "__main__":
    unittest.main()
