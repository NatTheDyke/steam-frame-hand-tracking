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

    def test_records_carry_submit_path_sequence_and_stage_times(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState

        pool, done = Mock(), Future()
        pool.submit.return_value = done
        palm = SimpleNamespace(pose={0: (np.eye(3), np.zeros(3)), 1: (np.eye(3), np.zeros(3))})
        tracker = MercuryTracker(palm, pool, Mock())
        joints = np.random.default_rng(3).normal(0, .01, (21, 3)) + [0, 0, .5]
        tracker.hands["left"] = HandState("left", joints, 1.0)
        info = {"path": "early", "seq": 41, "t_read": 5, "t_ready": 7}
        self.assertTrue(tracker.submit({0: np.zeros(1), 1: np.zeros(1)}, 10**9, {}, info=info))
        done.set_result([(1, "left", None, 0.01, {})])
        tracker.collect()
        rec = tracker.records[0]
        self.assertEqual((rec["path"], rec["seq"], rec["t_read"], rec["t_ready"]), ("early", 41, 5, 7))
        self.assertEqual(tracker.job_info, {})


class AdmissionTests(unittest.TestCase):
    PERIOD = 1 / 60

    def test_a_capture_is_submitted_at_most_once(self):
        from handtrack.mercsched import Admission
        adm = Admission()
        self.assertFalse(adm.admissible(0.0, self.PERIOD, False))          # no frame yet
        adm.frame(1, 0.0)
        self.assertTrue(adm.admissible(0.001, self.PERIOD, False))
        adm.submitted(0.001)
        for between in (False, True):
            self.assertFalse(adm.admissible(0.002, self.PERIOD, between))
        adm.frame(2, 0.017)
        self.assertTrue(adm.admissible(0.018, self.PERIOD, True))

    def test_stalled_cameras_stop_admission_and_resume_on_a_new_frame(self):
        from handtrack.mercsched import Admission, STALL_S
        adm = Admission()
        adm.frame(1, 0.0)
        self.assertFalse(adm.stalled(STALL_S))
        self.assertTrue(adm.stalled(STALL_S + 0.001))
        self.assertFalse(adm.admissible(STALL_S + 0.001, self.PERIOD, False))
        adm.frame(2, 5.0)
        self.assertTrue(adm.admissible(5.001, self.PERIOD, False))

    def test_fresh_submits_between_frames_only_within_a_third_of_a_period(self):
        from handtrack.mercsched import Admission
        fresh, legacy = Admission(fresh=True), Admission()
        for adm in (fresh, legacy):
            adm.frame(1, 1.0)
        self.assertTrue(fresh.admissible(1.005, self.PERIOD, True))
        self.assertFalse(fresh.admissible(1.006, self.PERIOD, True))       # wait for the next frame
        self.assertTrue(fresh.admissible(1.012, self.PERIOD, False))       # the frame path is unaffected
        self.assertTrue(legacy.admissible(1.012, self.PERIOD, True))       # --merc-early behaviour kept

    def test_power_is_released_once_after_the_grace_period_not_per_gap(self):
        from handtrack.mercsched import Admission, IDLE_S
        adm = Admission()
        self.assertFalse(adm.should_release(10.0, False))                  # nothing ever held
        self.assertFalse(adm.should_release(0.0, True))                    # in flight counts as active
        self.assertFalse(adm.should_release(IDLE_S * 0.9, False))          # short gap keeps power
        adm.active(IDLE_S * 0.9)
        self.assertFalse(adm.should_release(IDLE_S * 1.5, False))
        self.assertTrue(adm.should_release(IDLE_S * 2, False))
        self.assertFalse(adm.should_release(IDLE_S * 3, False))            # once per idle period
        adm.submitted(5.0)
        self.assertTrue(adm.should_release(5.0 + IDLE_S, False))           # resume, then idle again

    def test_frozen_cameras_release_power_without_reinferring_the_last_frame(self):
        """Simulated 60 Hz capture that freezes: the coordinator's admit/power sequence."""
        from handtrack.mercsched import Admission, IDLE_S, STALL_S
        adm, submits, releases, busy_until = Admission(), [], [], -1.0
        t, seq = 0.0, 0
        while t < 1.0:
            if t < 0.3 and t >= seq * self.PERIOD:                         # cameras freeze at 0.3 s
                seq += 1
                adm.frame(seq, t)
            busy = t < busy_until
            if busy:
                adm.active(t)
            elif not adm.stalled(t):
                if adm.admissible(t, self.PERIOD, True):
                    submits.append(adm.seq)
                    adm.submitted(t)
                    busy_until = t + 0.028
                else:
                    adm.active(t)
            if adm.should_release(t, t < busy_until):
                releases.append(t)
            t += 0.002
        self.assertEqual(len(submits), len(set(submits)))
        self.assertEqual(len(releases), 1)
        self.assertGreater(releases[0], 0.3 + STALL_S)
        self.assertLess(releases[0], 0.3 + STALL_S + IDLE_S + 0.01)


@unittest.skipUnless(sys.platform == "linux" and os.path.exists("/proc/self/sched"), "Linux only")
class UclampTests(unittest.TestCase):
    def test_clamp_applies_to_threads_and_is_inherited(self):
        import subprocess
        from handtrack import uclamp
        if uclamp.current_max() is None:
            self.skipTest("kernel without uclamp reporting")
        root = str(Path(__file__).resolve().parents[1])
        child = "from handtrack import uclamp; print(uclamp.current_max())"
        parent = ("import subprocess, sys, threading; from handtrack import uclamp; "
                  "threading.Thread(target=lambda: None).start(); uclamp.set_process_max(512); "
                  "print(uclamp.current_max()); "
                  "print(subprocess.run([sys.executable, '-c', %r], capture_output=True, text=True).stdout.strip())"
                  % child)
        out = subprocess.run([sys.executable, "-c", parent], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONPATH=root))
        if out.returncode and "Operation not permitted" in out.stderr:
            self.skipTest("uclamp not permitted here")
        self.assertEqual(out.stdout.split(), ["512", "512"], out.stderr)
        with self.assertRaises(ValueError):
            uclamp.set_process_max(2000)


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
