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


if __name__ == "__main__":
    unittest.main()
