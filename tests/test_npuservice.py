"""NPU service: wire format, one-client ownership, session lifetime and error handling, with a
stand-in session (no DSP needed)."""
from pathlib import Path
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtrack import npuservice as ns
from handtrack.inference import FallbackBackend, ServiceBackend


class FakeSession:
    """Deterministic outputs derived from the inputs, so the round trip can be checked."""
    opened = closed = 0

    def __init__(self, fail_on=None):
        FakeSession.opened += 1
        self.fail_on, self.idles, self.last_timing = fail_on, 0, None

    def infer(self, feeds):
        out = []
        for f in feeds:
            assert f["inputImg"].shape == (1, 1, 128, 128) and f["inputImg"].dtype == np.float32
            s = float(f["inputImg"].sum() + f["lastKeypoints"].sum() + f["useLastKeypoints"][0])
            if self.fail_on is not None and s == self.fail_on:
                raise RuntimeError("device fault")
            out.append({name: np.full(shape, s, np.float32) for name, shape in ns.OUTPUTS})
        self.last_timing = {"crops": len(feeds), "body_ms": np.float64(4.9)}
        return out

    def idle(self):
        self.idles += 1

    def close(self):
        FakeSession.closed += 1


def feed(v):
    return {"inputImg": np.full((1, 1, 128, 128), v, np.float32), "lastKeypoints": np.zeros((1, 42), np.float32),
            "useLastKeypoints": np.array([1.0], np.float32)}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "npu.sock")
        FakeSession.opened = FakeSession.closed = 0
        self.fail_on = None
        self.srv = ns.Server("model", "jobs", self.path, os.getuid(),
                             session_factory=lambda: FakeSession(self.fail_on))
        self.t = threading.Thread(target=self.srv.run, daemon=True)
        self.t.start()
        for _ in range(100):
            if os.path.exists(self.path):
                break
            time.sleep(0.01)

    def tearDown(self):
        self.srv.stop = True
        self.t.join(5)
        os.rmdir(self.dir)

    def wait_closed(self, n):
        for _ in range(200):
            if FakeSession.closed >= n:
                return
            time.sleep(0.01)

    def test_round_trip_one_to_three_crops_in_order(self):
        c = ns.Client(self.path)
        for k in (1, 2, 3):
            outs = c.infer([feed(i + 1) for i in range(k)])
            self.assertEqual(len(outs), k)
            for i, o in enumerate(outs):
                want = (i + 1) * 128 * 128 + 1
                for name, shape in ns.OUTPUTS:
                    self.assertEqual(o[name].shape, shape)
                    self.assertTrue(np.all(o[name] == want))
        self.assertEqual(c.last_timing, {"crops": 3, "body_ms": 4.9})
        c.close()

    def test_socket_is_private(self):
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_session_opens_on_first_infer_and_closes_on_disconnect(self):
        c = ns.Client(self.path)
        self.assertEqual(FakeSession.opened, 0)            # ping alone does not touch the DSP
        c.infer([feed(1)])
        c.idle()
        self.assertEqual(FakeSession.opened, 1)
        c.close()
        self.wait_closed(1)
        self.assertEqual(FakeSession.closed, 1)

    def test_second_client_is_turned_away_and_falls_back_to_cpu(self):
        c1 = ns.Client(self.path)
        with self.assertRaises(RuntimeError):
            ns.Client(self.path)
        c1.close()

    def test_device_error_closes_session_and_reports(self):
        self.fail_on = 1 * 128 * 128 + 1
        c = ns.Client(self.path)
        with self.assertRaises(RuntimeError):
            c.infer([feed(1)])
        self.wait_closed(1)
        self.assertEqual(FakeSession.closed, 1)
        c.close()

    def test_malformed_requests_are_refused(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(self.path)
        ns.send_msg(s, {"op": "infer", "n": 4})               # more than three crops: dropped
        self.assertEqual(s.recv(16), b"")
        s.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(self.path)
        s.sendall(struct.pack("<I", 1 << 30))                 # absurd header length: dropped
        self.assertEqual(s.recv(16), b"")
        s.close()
        c = ns.Client(self.path)                              # the service is still fine
        self.assertEqual(len(c.infer([feed(2)])), 1)
        c.close()

    def test_fallback_backend_uses_cpu_without_a_service(self):
        class Cpu:
            name = "cpu"

            def infer(self, feeds):
                return [{n: np.zeros(s, np.float32) for n, s in ns.OUTPUTS} for _ in feeds]

            def close(self):
                pass
        b = FallbackBackend(Cpu(), lambda: ServiceBackend(os.path.join(self.dir, "missing.sock")))
        self.assertEqual(b.name, "cpu")
        self.assertEqual(len(b.infer([feed(1)])), 1)
        b2 = FallbackBackend(Cpu(), lambda: ServiceBackend(self.path))
        self.assertEqual(b2.name, "framenpu")
        self.assertTrue(np.all(b2.infer([feed(1)])[0]["curls"] == 128 * 128 + 1))
        b2.close()


if __name__ == "__main__":
    unittest.main()
