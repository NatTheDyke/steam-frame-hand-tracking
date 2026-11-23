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


class FrameRequestTests(unittest.TestCase):
    def backend(self, client):
        accelerator = Mock()
        accelerator.client = client
        accelerator.infer.return_value = [outputs(1)]
        return FallbackBackend(Mock(), lambda: accelerator), accelerator

    def test_frame_request_carries_the_extra_crops(self):
        client = Mock(max_frame=2, models=[0, 2])
        client.infer_frame.return_value = [[outputs(2)], ["det"]]
        backend, _ = self.backend(client)
        kp, ex = backend.infer_frame(["a"], (2, ["d"]))
        self.assertEqual(client.infer_frame.call_args.args[0], [(0, ["a"]), (2, ["d"])])
        self.assertEqual((float(kp[0][OUTPUTS[0]].flat[0]), ex), (2.0, ["det"]))

    def test_no_frame_support_runs_keypoints_alone(self):
        for client in (None, Mock(max_frame=0, models=[0, 2]), Mock(max_frame=2, models=[0])):
            backend, accelerator = self.backend(client)
            kp, ex = backend.infer_frame(["a"], (2, ["d"]))
            self.assertIsNone(ex)
            self.assertEqual(accelerator.infer.call_args.args[0], ["a"])

    def test_frame_failure_latches_to_cpu(self):
        client = Mock(max_frame=2, models=[0, 2])
        client.infer_frame.side_effect = RuntimeError("completion uncertain")
        backend, _ = self.backend(client)
        with self.assertLogs("handtrack.inference", "WARNING"):
            kp, ex = backend.infer_frame(["a"], (2, ["d"]))
        self.assertIsNone(ex)
        self.assertEqual(backend.cpu.infer.call_args.args[0], ["a"])
        self.assertEqual(backend.name, "cpu")
        client.infer_frame.side_effect = None
        client.infer_frame.return_value = [[outputs()], ["d"], ["x"]]
        backend2, _ = self.backend(client)
        client.infer_frame.return_value = [[outputs()], []]               # wrong extra count
        with self.assertLogs("handtrack.inference", "WARNING"):
            self.assertIsNone(backend2.infer_frame(["a"], (2, ["d"]))[1])


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

    def test_discovery_rides_in_the_next_tracking_job(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState

        pool, done = Mock(), Future()
        pool.submit.return_value = done
        palm = SimpleNamespace(pose={0: (np.eye(3), np.zeros(3)), 1: (np.eye(3), np.zeros(3))})
        tracker = MercuryTracker(palm, pool, Mock())
        joints = np.random.default_rng(4).normal(0, .01, (21, 3)) + [0, 0, .5]
        tracker.hands["left"] = HandState("left", joints, 1.0)
        want = Future()
        tracker.discover_req = dict(future=want, pick=1, ncrop=2, det_size=640, t_req=0.0)
        imgs, raw = {0: "d0", 1: "d1"}, {0: "r0", 1: "r1"}
        self.assertTrue(tracker.submit(imgs, 10**9, {}, raw_imgs=raw))
        args = pool.submit.call_args.args
        self.assertEqual(args[2], raw)                     # both cameras, though one crop is tracked
        self.assertEqual((args[3], args[4]["pick"], args[4]["ncrop"]), (True, 1, 2))
        np.testing.assert_allclose(args[4]["tracked"]["left"], joints[[0, 5, 9, 17]].mean(0))
        self.assertIsNone(tracker.discover_req)
        self.assertEqual((tracker.job_disc["t_cap"], tracker.job_disc["det_imgs"]), (10**9, imgs))
        done.set_result({"obs": [(1, "left", None, 0.01, {})], "discover": {0: ([], .01), 1: ([], .01)}})
        tracker.collect()
        self.assertEqual(want.result(0), {0: ([], .01), 1: ([], .01)})
        self.assertEqual(len(tracker.records), 1)
        # a failed job fails its discovery too; without raw frames nothing rides
        failed, want2 = Future(), Future()
        pool.submit.return_value = failed
        tracker.discover_req = dict(future=want2, pick=0, ncrop=2, det_size=640, t_req=0.0)
        tracker.submit(imgs, 2 * 10**9, {}, raw_imgs=raw)
        failed.set_exception(RuntimeError("worker died"))
        tracker.collect()
        self.assertIsInstance(want2.exception(0), RuntimeError)
        req = dict(future=Future(), pick=0, ncrop=2, det_size=640, t_req=0.0)
        tracker.discover_req = req
        pool.submit.return_value = Future()
        tracker.submit(imgs, 3 * 10**9, {})
        self.assertEqual(len(pool.submit.call_args.args), 3)     # run_mercury, jobs, imgs: nothing rides
        self.assertIs(tracker.discover_req, req)

    def test_unplaceable_hand_expires_and_submit_time_is_kept(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState, LOST_AFTER_S

        pool = Mock()
        pool.submit.return_value = Future()
        palm = SimpleNamespace(pose={0: (np.eye(3), np.zeros(3)), 1: (np.eye(3), np.zeros(3))})
        tracker = MercuryTracker(palm, pool, Mock())
        behind = np.random.default_rng(5).normal(0, .01, (21, 3)) + [0, 0, -.3]
        tracker.hands["right"] = HandState("right", behind, 1.0)
        self.assertFalse(tracker.submit({0: np.zeros(1), 1: np.zeros(1)}, 10**9, {}))
        self.assertIsNone(tracker.last_submit_t)
        self.assertEqual(tracker.expire(1.0 + LOST_AFTER_S / 2), [])
        self.assertEqual(tracker.expire(1.0 + 2 * LOST_AFTER_S), ["right"])
        self.assertEqual(tracker.tracked(), set())
        tracker.hands["left"] = HandState("left", behind * [1, 1, -1], 1.0)
        self.assertTrue(tracker.submit({0: np.zeros(1), 1: np.zeros(1)}, 2 * 10**9, {}))
        self.assertIsNotNone(tracker.last_submit_t)

    def test_preflow_job_waits_for_its_images_and_anchor(self):
        """--merc-preflow: submitted before the capture's detector images / flow exist; collect() holds the
        finished job until complete() fills them in, then applies it with this capture's anchor."""
        from concurrent.futures import Future
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState

        pool, done = Mock(), Future()
        pool.submit.return_value = done
        palm = SimpleNamespace(pose={0: (np.eye(3), np.zeros(3)), 1: (np.eye(3), np.zeros(3))})
        tracker = MercuryTracker(palm, pool, Mock())
        joints = np.random.default_rng(6).normal(0, .01, (21, 3)) + [0, 0, .5]
        tracker.hands["left"] = HandState("left", joints, 1.0)
        want = Future()
        tracker.discover_req = dict(future=want, pick=0, ncrop=2, det_size=640, t_req=0.0)
        raw = {0: "r0", 1: "r1"}
        old = {"left": (np.array([0.0, 0, .5]), None)}
        self.assertTrue(tracker.submit(None, 10**9, old, anchor_pos=old, info={"t_read": 5}, raw_imgs=raw))
        self.assertTrue(tracker.pending())
        self.assertEqual(pool.submit.call_args.args[2], raw)
        self.assertIsNone(tracker.job_disc["det_imgs"])
        done.set_result({"obs": [(1, "left", None, 0.01, {})], "discover": {0: ([], .01), 1: ([], .01)}})
        self.assertEqual(tracker.collect(), {})                   # finished, but not completed yet
        self.assertFalse(want.done())
        imgs, new = {0: "d0", 1: "d1"}, {"left": (np.array([0.01, 0, .5]), None)}
        self.assertTrue(tracker.complete(imgs, new, t_ready=7))
        self.assertFalse(tracker.pending())
        self.assertIs(tracker.job[2], new)
        self.assertEqual(tracker.job[3], {s: imgs[s] for s in tracker.job_used})
        self.assertEqual(tracker.job_disc["det_imgs"], imgs)
        tracker.collect()
        self.assertEqual(want.result(0), {0: ([], .01), 1: ([], .01)})
        self.assertEqual((tracker.records[0]["t_read"], tracker.records[0]["t_ready"]), (5, 7))
        self.assertFalse(tracker.complete(imgs, new))              # nothing pending

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


class MercuryViewTests(unittest.TestCase):
    """Both views of one frame -> stereo from one instant; an older view is moved with the palm."""
    O = (np.array([0.0527, -0.0186, -0.0368]), np.array([-0.0528, -0.0186, -0.037]))   # SLAM cameras

    def setUp(self):
        from types import SimpleNamespace
        from handtrack.merctrack import MercuryTracker, HandState
        self.HandState = HandState
        self.poses = []
        self.R = np.diag([1.0, -1.0, -1.0])                  # camera -> head: cameras look forward (-z)
        palm = SimpleNamespace(pose={s: (self.R, self.O[s]) for s in (0, 1)},
                               raw_quat=lambda side, P: (1.0, 0.0, 0.0, 0.0),
                               curl_filt={sd: (lambda c, t: c) for sd in ("left", "right")},
                               _pose=lambda side, P, t, conf: self.poses.append((side, P.copy(), t)) or
                               (P[0], (1.0, 0.0, 0.0, 0.0), conf, P))
        self.pool = Mock()
        self.mt = MercuryTracker(palm, self.pool, Mock())
        rng = np.random.default_rng(4)
        self.J = rng.normal(0, 0.03, (21, 3)) + [0.0, -0.2, -0.5]          # a hand 0.5 m ahead, below

    def result(self, slot, P, ok=True):
        d = P - self.O[slot]
        return {"is_hand": 0.99 if ok else 0.5, "rays": (d / np.linalg.norm(d, axis=1)[:, None]) @ self.R,
                "rel_depth": np.zeros(21), "curls": np.zeros(5), "curl_var": np.ones(5), "det_px": None}

    def run_job(self, results, t_ns, fast_pos):
        from concurrent.futures import Future
        f = Future()
        f.set_result([(slot, side, r, 0.01, {"inference": "framenpu"}) for slot, side, r in results])
        self.mt.job = (f, t_ns, fast_pos, {0: None, 1: None}, 0, len(results))
        return self.mt.collect()

    def test_both_views_of_one_frame_triangulate_once(self):
        self.mt.views = "both"
        self.mt.hands["right"] = self.HandState("right", self.J, 0.9)
        self.assertTrue(self.mt.submit({0: 0, 1: 0}, 10**9, {}))
        jobs = self.pool.submit.call_args.args[1]
        self.assertEqual(sorted(j[0] for j in jobs), [0, 1])
        out = self.run_job([(0, "right", self.result(0, self.J)), (1, "right", self.result(1, self.J))], 10**9, {})
        self.assertEqual(len(self.poses), 1)                    # one pose per hand and frame
        self.assertEqual([r["outcome"] for r in self.mt.records], ["pair", "stereo"])
        np.testing.assert_allclose(self.poses[0][1], self.J, atol=1e-9)
        self.assertIn("right", out)

    def test_pair_with_a_rejected_second_view_still_poses(self):
        self.mt.hands["right"] = self.HandState("right", self.J, 0.9)
        self.run_job([(0, "right", self.result(0, self.J)), (1, "right", self.result(1, self.J, ok=False))],
                     10**9, {})
        self.assertEqual(len(self.poses), 1)
        self.assertEqual(self.mt.records[0]["outcome"], "mono")

    def test_second_views_take_turns_between_two_hands(self):
        self.mt.views = "both"
        self.mt.hands["right"] = self.HandState("right", self.J, 0.9)
        self.mt.hands["left"] = self.HandState("left", self.J + [-0.3, 0, 0], 0.9)
        doubled = []
        for k in range(4):
            self.mt.job = None
            self.assertTrue(self.mt.submit({0: 0, 1: 0}, 10**9 + k, {}))
            jobs = self.pool.submit.call_args.args[1]
            self.assertEqual(len(jobs), 3)                      # at most three crops per job
            doubled.append([sd for sd in ("left", "right") if sum(j[1] == sd for j in jobs) == 2][0])
        self.assertEqual(len(set(doubled[0::2])), 1)
        self.assertNotEqual(doubled[0], doubled[1])

    def test_cpu_inference_alternates_in_auto(self):
        self.mt.hands["right"] = self.HandState("right", self.J, 0.9)
        self.mt.last_inference = "cpu"
        self.assertTrue(self.mt.submit({0: 0, 1: 0}, 10**9, {}))
        self.assertEqual(len(self.pool.submit.call_args.args[1]), 1)

    def test_older_view_is_moved_with_the_palm(self):
        self.mt.views = "alternate"
        move = np.array([0.01, 0.0, 0.0])                       # 1 cm between the two captures
        errs = {}
        for comp in (False, True):
            self.poses.clear()
            self.mt.stereo_comp = comp
            self.mt.hands["right"] = self.HandState("right", self.J, 0.9)
            c0 = self.J[[0, 5, 9, 17]].mean(0)
            self.run_job([(0, "right", self.result(0, self.J))], 10**9, {"right": (c0,)})
            self.run_job([(1, "right", self.result(1, self.J + move))], 10**9 + 33 * 10**6, {"right": (c0 + move,)})
            P = self.poses[-1][1]
            errs[comp] = np.linalg.norm(P[[0, 5, 9, 17]].mean(0) - (c0 + move))
        self.assertLess(errs[True], 1e-6)
        self.assertGreater(errs[False], 0.005)


    def test_single_view_curls_replace_only_the_chosen_fingers(self):
        from handtrack.merctrack import view_joints
        from handtrack.palm import finger_curls
        J = self.J
        o = self.O[0]
        d = J - o
        rays = d / np.linalg.norm(d, axis=1)[:, None]
        hs = np.linalg.norm(J[0] - J[9])
        rel = (np.linalg.norm(d, axis=1) - np.linalg.norm(d[9])) / hs
        np.testing.assert_allclose(view_joints(J, hs, o, rays, rel), J, atol=1e-12)   # exact rays + depths -> same hand
        self.mt.views = "alternate"
        self.mt.curl_geo = [1]
        self.mt.hands["right"] = self.HandState("right", J, 0.9)
        res = self.result(0, J)
        res["curls"] = np.full(5, -2.0)                       # Mercury curl output: 1.0 thumb, 0.5 fingers
        res["rel_depth"] = rel
        out = self.run_job([(0, "right", res)], 10**9, {})
        curls = out["right"][0][4]
        self.assertAlmostEqual(curls[1], finger_curls(J)[1], places=9)   # index from the view's joints
        np.testing.assert_allclose(curls[[0, 2, 3, 4]], [1.0, 0.5, 0.5, 0.5])


class AdmissionTests(unittest.TestCase):
    PERIOD = 1 / 60

    def test_rate_cap_tolerates_frame_jitter(self):
        from handtrack.mercsched import rate_cap_due
        period = 1 / 30.0
        submit = 10.0
        next_t = submit + 1 / 30.0                 # --merc-hz 30 at a 30 Hz camera
        # the next frame ready 1 ms "early" (jitter) is still due; half a period early is not
        self.assertTrue(rate_cap_due(submit + period - 0.001, next_t, period))
        self.assertFalse(rate_cap_due(submit + period / 2, next_t, period))
        # 60 Hz cameras with a 30 Hz cap still skip the frame in between
        self.assertFalse(rate_cap_due(submit + 1 / 60.0, next_t, 1 / 60.0))
        self.assertTrue(rate_cap_due(submit + 2 / 60.0 - 0.001, next_t, 1 / 60.0))

    def test_capture_cadence_keeps_a_fixed_frame_gap(self):
        from handtrack.mercsched import capture_cap_due, capture_cap_frames
        iv = 1 / 30.0
        for hz, k in ((114.4, 4), (30.0, 1), (60.0, 2), (90.0, 3)):
            period_ns = 1e9 / hz
            self.assertEqual(capture_cap_frames(iv, period_ns / 1e9), k)
            rng = np.random.default_rng(int(hz))
            last, subs = None, []
            for i in range(400):
                t = int(i * period_ns + rng.normal(0, 0.3e6))   # capture-time jitter 0.3 ms
                if capture_cap_due(t, last, iv, period_ns):
                    subs.append(i)
                    last = t
            self.assertEqual(set(np.diff(subs)), {k}, hz)
        # a frame the loop never processed: the next processed one is taken
        p = 1e9 / 114.4
        self.assertFalse(capture_cap_due(int(3 * p), 0, iv, p))
        self.assertTrue(capture_cap_due(int(5 * p), 0, iv, p))
        self.assertTrue(capture_cap_due(0, None, iv, p))

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


class FrameGridTests(unittest.TestCase):
    @staticmethod
    def old_grid(raws, nominal):
        """the inline grid live.py had before FrameGrid"""
        out, t, period = [], None, nominal
        for r in raws:
            if t is None or abs(r - t) > 1e9:
                t = r
            else:
                nfr = max(1, round((r - t) / period))
                pred = t + nfr * period
                resid = r - pred
                t = pred + 0.05 * resid
                period += 0.002 * resid / nfr
                period = min(max(period, 0.9 * nominal), 1.11 * nominal)
            out.append(int(t))
        return out

    def frames(self, period, n, skip=(1,), seed=0):
        rng = np.random.default_rng(seed)
        k, out, true = 0, [], []
        for _ in range(n):
            k += int(rng.choice(skip))
            true.append(1e12 + k * period)
            out.append(int(true[-1] + rng.normal(0, 0.4e6)))
        return out, np.array(true)

    def test_30hz_is_the_old_grid(self):
        from handtrack.mercsched import FrameGrid
        raws, true = self.frames(1e9 / 30, 600, skip=(1, 1, 1, 2))
        g = FrameGrid(1e9 / 30)
        self.assertEqual([g.update(r) for r in raws], self.old_grid(raws, 1e9 / 30))
        self.assertIsNone(g.adopted)

    def test_120hz_mode_adopts_the_measured_period(self):
        from handtrack.mercsched import FrameGrid
        period = 1e9 / 114
        for skip in ((1,), (1, 2, 2, 3), (2, 3), (3, 4)):   # every frame, or a loop that skips some / always
            raws, true = self.frames(period, 900, skip=skip, seed=len(skip))
            g = FrameGrid(1e9 / 30)
            caps = np.array([g.update(r) for r in raws])
            self.assertAlmostEqual(g.adopted / period, 1.0, delta=0.12)
            err = np.abs(caps[100:] - true[100:])
            self.assertLess(np.percentile(err, 95), 1.5e6, skip)
            self.assertLess(abs(g.period - period) / period, 0.01, skip)
        # the old grid ran ahead of real time at this rate
        raws, true = self.frames(period, 300)
        self.assertGreater(np.max(np.array(self.old_grid(raws, 1e9 / 30)) - true), 0.1e9)
