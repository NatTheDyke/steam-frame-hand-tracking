import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtrack.inference import CPUBackend, FallbackBackend, OUTPUTS, SHAPES, make_backend
from handtrack.mercury_kp import MercuryKeypoint
from handtrack import mpworker


def outputs(value=0):
    return {n: np.full(s, value, np.float32) for n, s in zip(OUTPUTS, SHAPES)}


class BackendTests(unittest.TestCase):
    def test_init_failure_keeps_cpu_and_does_not_retry(self):
        cpu, factory = Mock(), Mock(side_effect=PermissionError("no device access"))
        with self.assertLogs("handtrack.inference", "WARNING"):
            backend = FallbackBackend(cpu, factory)
        backend.infer([1])
        backend.infer([2, 3])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(cpu.infer.call_args_list[0].args[0], [1])
        self.assertEqual(backend.name, "cpu")

    def test_inference_failure_replays_entire_batch_on_cpu_and_latches(self):
        cpu = Mock()
        accelerator = Mock()
        accelerator.infer.side_effect = RuntimeError("completion uncertain")
        accelerator.close.side_effect = RuntimeError("state retained")
        backend = FallbackBackend(cpu, lambda: accelerator)
        feeds = [object(), object(), object()]
        with self.assertLogs("handtrack.inference", "WARNING"):
            backend.infer(feeds)
        self.assertIs(cpu.infer.call_args.args[0], feeds)
        backend.infer([object()])
        accelerator.infer.assert_called_once_with(feeds)
        accelerator.close.assert_called_once()
        backend.idle()
        accelerator.idle.assert_not_called()
        backend.close()
        cpu.close.assert_called_once()

    def test_invalid_accelerator_outputs_fall_back(self):
        for result in ([], [outputs(float("nan"))], [{"heatmap_xy": np.zeros(1)}]):
            with self.subTest(result=len(result)):
                cpu, accelerator = Mock(), Mock()
                accelerator.infer.return_value = result
                backend = FallbackBackend(cpu, lambda: accelerator)
                with self.assertLogs("handtrack.inference", "WARNING"):
                    backend.infer([object()])
                cpu.infer.assert_called_once()
                self.assertEqual(backend.name, "cpu")

    def test_valid_batches_are_not_padded_and_idle_releases_accelerator(self):
        cpu, accelerator = Mock(), Mock()
        backend = FallbackBackend(cpu, lambda: accelerator)
        for count in (1, 2, 3, 1):
            accelerator.infer.return_value = [outputs(i) for i in range(count)]
            self.assertEqual(len(backend.infer([object() for _ in range(count)])), count)
        self.assertEqual([len(c.args[0]) for c in accelerator.infer.call_args_list], [1, 2, 3, 1])
        cpu.infer.assert_not_called()
        backend.idle()
        accelerator.idle.assert_called_once()
        for count in (0, 4):
            with self.assertRaises(ValueError):
                backend.infer([None] * count)

    def test_idle_failure_disables_accelerator_before_next_observation(self):
        cpu, accelerator = Mock(), Mock()
        accelerator.idle.side_effect = RuntimeError("power verification failed")
        backend = FallbackBackend(cpu, lambda: accelerator)
        with self.assertLogs("handtrack.inference", "WARNING"):
            backend.idle()
        backend.infer([object()])
        accelerator.close.assert_called_once()
        accelerator.infer.assert_not_called()
        cpu.infer.assert_called_once()

    def test_backend_timing_disappears_on_cpu_fallback(self):
        cpu, accelerator = Mock(), Mock()
        accelerator.last_timing = {"crops": 2, "body_ms": 10.0}
        backend = FallbackBackend(cpu, lambda: accelerator)
        self.assertEqual(backend.last_timing, accelerator.last_timing)
        accelerator.idle.side_effect = RuntimeError("cannot verify power")
        with self.assertLogs("handtrack.inference", "WARNING"):
            backend.idle()
        self.assertIsNone(backend.last_timing)

    def test_cpu_default_does_not_construct_accelerator(self):
        with patch("handtrack.inference.CPUBackend") as cpu, patch("handtrack.inference.FrameNPUBackend") as npu:
            self.assertIs(make_backend("model"), cpu.return_value)
            npu.assert_not_called()


class CropBatchTests(unittest.TestCase):
    def wrapper(self):
        mk = object.__new__(MercuryKeypoint)
        mk.backend = Mock(name="backend")
        mk.backend.name = "test"
        mk.backend.last_timing = None
        mk.backend.infer.side_effect = lambda fs: [{"value": f} for f in fs]
        mk._prepare = Mock(side_effect=lambda value: None if value is None else ("context", value))
        mk._decode = Mock(side_effect=lambda p, out: out["value"])
        return mk

    def test_small_calls_submit_before_return_and_never_wait_for_third(self):
        mk = self.wrapper()
        for values in ([1], [2, 3], [4, 5, 6], [7]):
            self.assertEqual(mk.run_many([{"value": i} for i in values]), values)
            self.assertEqual(mk.backend.infer.call_args.args[0], values)
        self.assertEqual(mk.backend.infer.call_count, 4)

    def test_four_observations_submit_three_then_one_in_order(self):
        mk = self.wrapper()
        self.assertEqual(mk.run_many([{"value": i} for i in range(4)]), [0, 1, 2, 3])
        self.assertEqual([c.args[0] for c in mk.backend.infer.call_args_list], [[0, 1, 2], [3]])

    def test_blank_crop_keeps_its_slot_and_does_not_infer_or_pad(self):
        mk = self.wrapper()
        self.assertEqual(mk.run_many([{"value": i} for i in (1, None, 3)]), [1, None, 3])
        mk.backend.infer.assert_called_once_with([1, 3])
        mk.backend.infer.reset_mock()
        self.assertEqual(mk.run_many([{"value": None}]), [None])
        self.assertEqual(mk.run_many([]), [])
        mk.backend.infer.assert_not_called()

    def test_worker_preserves_identity_hint_and_order_across_chunks(self):
        mk = self.wrapper()
        mk.run_many = Mock(side_effect=lambda obs: [dict(crop=np.zeros(1), is_hand=i) for i in range(len(obs))])
        jobs = [(i % 2, "left" if i % 2 == 0 else "right", np.zeros((21, 3)), 2.0, i % 2 == 0)
                for i in range(4)]
        with patch.object(mpworker, "init_mercury", return_value=mk):
            out = mpworker.run_mercury(jobs, {0: np.zeros(1), 1: np.zeros(1)})
        self.assertEqual([len(c.args[0]) for c in mk.run_many.call_args_list], [3, 1])
        self.assertEqual([(r[0], r[1]) for r in out], [(j[0], j[1]) for j in jobs])
        self.assertEqual([r[4]["hint"] for r in out], [True, False, True, False])
        self.assertTrue(all("crop" not in r[2] for r in out))
        self.assertEqual([r[4]["batch_crops"] for r in out], [3, 3, 3, 1])
        self.assertEqual([r[4]["request_crops"] for r in out], [3, 3, 3, 1])
        self.assertEqual([r[4]["dsp_body_ms"] for r in out], [0.0] * 4)


@unittest.skipUnless(os.environ.get("FNPU_MERCURY_MODEL"), "pinned local model required")
class CPUParityTests(unittest.TestCase):
    def test_cpu_outputs_are_exactly_the_existing_onnx_session(self):
        backend = CPUBackend(os.environ["FNPU_MERCURY_MODEL"])
        fs = [{"inputImg": np.random.default_rng(i).uniform(-1, 1, (1, 1, 128, 128)).astype(np.float32),
               "lastKeypoints": np.zeros((1, 42), np.float32),
               "useLastKeypoints": np.zeros(1, np.float32)} for i in range(3)]
        for feeds, out in zip(fs, backend.infer(fs)):
            old = backend.sess.run(list(OUTPUTS), feeds)
            for name, value in zip(OUTPUTS, old):
                np.testing.assert_array_equal(out[name], value)


class SchedulerTests(unittest.TestCase):
    def test_one_hand_submits_now_and_idle_is_sent_once(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState

        pool, worker = Mock(), Mock()
        pending = Future()
        pool.submit.return_value = pending
        palm = SimpleNamespace(pose={0: (np.eye(3), np.zeros(3)), 1: (np.eye(3), np.zeros(3))})
        tracker = MercuryTracker(palm, pool, worker)
        joints = np.random.default_rng(9).normal(0, .01, (21, 3)) + [0, 0, .5]
        tracker.hands["left"] = HandState("left", joints, 1.0)
        self.assertTrue(tracker.submit({0: np.zeros(1), 1: np.zeros(1)}, 1000000000, {}))
        self.assertEqual(len(pool.submit.call_args.args[1]), 1)
        self.assertFalse(tracker.submit({0: np.zeros(1), 1: np.zeros(1)}, 1000000001, {}))
        self.assertEqual(pool.submit.call_count, 1)
        tracker.idle()
        self.assertEqual(pool.submit.call_count, 1)  # outstanding observations must finish first
        tracker.job = None
        tracker.hands.clear()
        tracker.idle()
        tracker.idle()
        self.assertEqual(pool.submit.call_count, 2)
        self.assertIs(pool.submit.call_args.args[0], worker.idle_mercury)


@unittest.skipUnless(os.environ.get("FNPU_MERCURY_MODEL"), "pinned local model required")
class GeometryParityTests(unittest.TestCase):
    def test_batch_preserves_crop_decode_and_observation_order(self):
        from types import SimpleNamespace

        class Camera:
            def project(self, rays):
                return rays[:, :2] / rays[:, 2:] * 1000 + 1231.5

        cam = Camera()
        mk = MercuryKeypoint(os.environ["FNPU_MERCURY_MODEL"],
                             SimpleNamespace(cams={"a": cam, "b": cam}), {0: "a", 1: "b"}, 640)
        rng = np.random.default_rng(63)
        img = rng.integers(0, 256, (640, 640), dtype=np.uint8)
        observations = [dict(slot=i % 2, det_img=img,
                             joints_cv=rng.normal(0, .02, (21, 3)) + [0, 0, .5],
                             flip=i % 2 == 0, use_hint=i < 2, expand=1.7 + i * .1)
                        for i in range(4)]
        singles = [mk.run(**o) for o in observations]
        batched = mk.run_many(observations)
        for single, batch in zip(singles, batched):
            for name in single:
                np.testing.assert_array_equal(single[name], batch[name])
        mk.close()
