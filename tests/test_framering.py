"""framering: frame pairs reach a spawned worker bit-identical; slots are held until the job is done."""
import multiprocessing
import os
import sys
import unittest
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.framering import FrameRing, frames  # noqa: E402

SHAPE = (64, 66)


def checksum(imgs):
    f = frames(imgs)
    return {c: (f[c].shape, f[c].tobytes()) for c in sorted(f)}


class FrameRingTests(unittest.TestCase):
    def setUp(self):
        self.ring = FrameRing(SHAPE, slots=2)
        self.addCleanup(self.ring.close)
        rng = np.random.default_rng(1)
        self.pairs = [{c: rng.integers(0, 255, SHAPE, np.uint8) for c in (0, 1)} for _ in range(3)]

    def test_worker_sees_the_same_frames(self):
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
            for pair in self.pairs:
                ref = self.ring.put(pair)
                fut = ex.submit(checksum, ref)
                self.ring.release_when_done(ref, fut)
                self.assertEqual(fut.result(), checksum(pair))

    def test_slots_are_held_until_the_job_is_done(self):
        futs = [Future(), Future()]
        refs = [self.ring.put(p) for p in self.pairs[:2]]
        for r, f in zip(refs, futs):
            self.ring.release_when_done(r, f)
        self.assertIsNone(self.ring.put(self.pairs[2]))          # full: the caller pickles instead
        self.assertEqual(self.ring.stats["full"], 1)
        futs[0].set_result(None)
        ref = self.ring.put(self.pairs[2])
        self.assertEqual(ref.k, refs[0].k)                        # the finished job's slot, reused
        # the unfinished job's frames were not touched
        np.testing.assert_array_equal(frames(refs[1])[1], self.pairs[1][1])

    def test_other_shapes_and_plain_frames(self):
        self.assertIsNone(self.ring.put({0: np.zeros((8, 8), np.uint8), 1: np.zeros((8, 8), np.uint8)}))
        self.assertIs(frames(self.pairs[0]), self.pairs[0])


if __name__ == "__main__":
    unittest.main()
