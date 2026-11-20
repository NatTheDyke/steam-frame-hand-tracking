"""FrameNPU Mercury inference as a small privileged service, so the tracker can run as the user.

The DSP needs /dev/mem (root). Instead of running the whole tracker as root, this service owns
the FrameNPU MercurySession and answers inference requests over a Unix socket that only the
tracking user (and root) may use. The tracker's `--merc-inference service` backend
(handtrack/inference.py ServiceBackend) is the only client; any failure there falls back to CPU.

Server (root, normally via systemd: headset/install_npu_service.sh):
    python -I npuservice.py --model M.onnx --jobs DIR --user steamos [--socket PATH]

Design rules
- Standard library + numpy + framenpu only, so the service runs from its own root-owned copy
  without importing the tracker package.
- No pickle: requests are a small JSON header plus raw float32 arrays of fixed, validated
  shapes, so a client can never make root run code. MercurySession validates values again.
- The model and job image are the service's own (root-owned, hash-verified by FrameNPU);
  a client cannot choose them.
- One client at a time (the DSP has a single owner). A second client is told "busy" and
  closed, and falls back to CPU. The session opens on the client's first request and closes
  when it disconnects, so a stopped tracker leaves the DSP exactly as before.
- Any FrameNPU error closes the session and is reported; the client latches to CPU.

Wire format (little endian), both directions:  u32 header length, JSON header, payload bytes.
  request  {"op": "infer", "n": k}  payload = k x (inputImg, lastKeypoints, useLastKeypoints)
           {"op": "idle"} | {"op": "ping"}
  response {"ok": true, "timing": {...}}  payload = k x (heatmap_xy, heatmap_depth,
           scalar_extras, curls);  {"ok": false, "error": "..."} on failure
"""
import argparse
import json
import logging
import os
import pwd
import signal
import socket
import struct

import numpy as np

SOCKET = "/run/mrhand-npu/npu.sock"
INPUTS = (("inputImg", (1, 1, 128, 128)), ("lastKeypoints", (1, 42)), ("useLastKeypoints", (1,)))
OUTPUTS = (("heatmap_xy", (1, 21, 22, 22)), ("heatmap_depth", (1, 21, 22)), ("scalar_extras", (1, 8)),
           ("curls", (1, 10)))
MAX_HEADER = 1 << 16
LOG = logging.getLogger("npuservice")


def _nbytes(spec):
    return sum(4 * int(np.prod(shape)) for _, shape in spec)


IN_BYTES, OUT_BYTES = _nbytes(INPUTS), _nbytes(OUTPUTS)


# ---------------------------------------------------------------- framing
def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


def send_msg(sock, header, payload=b""):
    h = json.dumps(header, separators=(",", ":"), default=float).encode()
    sock.sendall(struct.pack("<I", len(h)) + h + payload)


def recv_msg(sock, payload_bytes):
    """payload_bytes(header) -> how many payload bytes follow (validated by the caller)."""
    (hl,) = struct.unpack("<I", _recv_exact(sock, 4))
    if not 0 < hl <= MAX_HEADER:
        raise ValueError("bad header length")
    header = json.loads(_recv_exact(sock, hl))
    if not isinstance(header, dict):
        raise ValueError("bad header")
    n = payload_bytes(header)
    return header, (_recv_exact(sock, n) if n else b"")


def pack_arrays(items, spec):
    return b"".join(np.ascontiguousarray(np.asarray(item[name], np.float32).reshape(shape)).tobytes()
                    for item in items for name, shape in spec)


def unpack_arrays(payload, count, spec):
    out, at = [], 0
    for _ in range(count):
        item = {}
        for name, shape in spec:
            n = 4 * int(np.prod(shape))
            item[name] = np.frombuffer(payload, np.float32, n // 4, at).reshape(shape).copy()
            at += n
        out.append(item)
    if at != len(payload):
        raise ValueError("payload size mismatch")
    return out


def _infer_count(header):
    n = header.get("n", 0) if header.get("op") == "infer" else 0
    if header.get("op") == "infer" and (not isinstance(n, int) or not 1 <= n <= 3):
        raise ValueError("infer carries one to three crops")
    return n


# ---------------------------------------------------------------- client
class Client:
    """Tracker side: same interface as the in-process FrameNPU backend (infer / idle / close)."""

    def __init__(self, path=SOCKET, timeout=2.0, open_timeout=30.0):
        """timeout: per request; open_timeout: the first inference, which opens the DSP session
        (parameter upload and image check) in the service."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(path)
            header, _ = self._call({"op": "ping"})
        except Exception:
            self.sock.close()
            raise
        self.info = header
        self.last_timing = None
        self.timeout, self.open_timeout, self.opened = timeout, open_timeout, False

    def _call(self, header, payload=b"", out_count=0):
        send_msg(self.sock, header, payload)
        resp, data = recv_msg(self.sock, lambda h: out_count * OUT_BYTES if h.get("ok") else 0)
        if not resp.get("ok"):
            raise RuntimeError("NPU service: %s" % resp.get("error", "failed"))
        return resp, data

    def infer(self, feeds):
        if not 1 <= len(feeds) <= 3:
            raise ValueError("inference calls carry one to three crops")
        self.sock.settimeout(self.timeout if self.opened else self.open_timeout)
        resp, data = self._call({"op": "infer", "n": len(feeds)}, pack_arrays(feeds, INPUTS), len(feeds))
        self.opened = True
        self.sock.settimeout(self.timeout)
        self.last_timing = resp.get("timing")
        return unpack_arrays(data, len(feeds), OUTPUTS)

    def idle(self):
        self._call({"op": "idle"})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, model, jobs, path, uid, session_factory=None, require_pin=True):
        self.model, self.jobs, self.path, self.uid = model, jobs, path, uid
        if session_factory is None:
            def session_factory():
                from framenpu.mercury_runtime import MercurySession
                return MercurySession(self.model, self.jobs, require_pin=require_pin)
        self.session_factory = session_factory
        self.session = None
        self.stop = False

    def listen(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        # bind under a temporary name and rename once listening: a client never finds a socket
        # file that still refuses connections
        tmp = self.path + ".new"
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        self.lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            self.lsock.bind(tmp)
        finally:
            os.umask(old)
        if os.geteuid() == 0:
            os.chown(tmp, self.uid, -1)
        self.lsock.listen(2)
        os.rename(tmp, self.path)
        self.lsock.settimeout(1.0)
        LOG.info("listening on %s (user uid %d)", self.path, self.uid)

    def _peer_ok(self, conn):
        pid, uid, gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        return uid in (0, self.uid), (pid, uid)

    def _close_session(self, why):
        s, self.session = self.session, None
        if s is not None:
            try:
                s.close()
                LOG.info("session closed (%s)", why)
            except Exception as exc:          # poisoned sessions keep their refusal record
                LOG.warning("session close (%s) retained device state: %s", why, exc)

    def _handle(self, header, payload):
        op = header.get("op")
        if op == "ping":
            return {"ok": True, "service": "mrhand-npu", "version": 1}, b""
        if op == "idle":
            if self.session is not None:
                self.session.idle()
            return {"ok": True}, b""
        if op == "infer":
            n = _infer_count(header)
            feeds = unpack_arrays(payload, n, INPUTS)
            if self.session is None:
                self.session = self.session_factory()
                LOG.info("session opened")
            outs = self.session.infer(feeds)
            if len(outs) != n:
                raise ValueError("FrameNPU returned the wrong crop count")
            timing = getattr(self.session, "last_timing", None)
            return {"ok": True, "timing": json.loads(json.dumps(timing, default=float)) if timing else None}, \
                pack_arrays(outs, OUTPUTS)
        raise ValueError("unknown op")

    def serve_client(self, conn):
        conn.settimeout(None)
        try:
            while not self.stop:
                try:
                    header, payload = recv_msg(conn, lambda h: _infer_count(h) * IN_BYTES)
                except (ConnectionError, OSError):
                    return
                except ValueError as exc:     # malformed framing: drop this client
                    LOG.warning("bad request: %s", exc)
                    return
                try:
                    resp, data = self._handle(header, payload)
                except Exception as exc:
                    LOG.warning("request failed, session closed: %s", exc)
                    self._close_session("error")
                    resp, data = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, b""
                send_msg(conn, resp, data)
        finally:
            conn.close()
            self._close_session("client disconnected")

    def run(self):
        self.listen()
        try:
            while not self.stop:
                try:
                    conn, _ = self.lsock.accept()
                except socket.timeout:
                    continue
                ok, peer = self._peer_ok(conn)
                if not ok:
                    LOG.warning("refused client pid %d uid %d", *peer)
                    conn.close()
                    continue
                LOG.info("client pid %d connected", peer[0])
                self.lsock.settimeout(0.0)    # while serving, turn away a second client at once
                self._serve_one(conn)
                self.lsock.settimeout(1.0)
        finally:
            self._close_session("service stopping")
            self.lsock.close()
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def _serve_one(self, conn):
        import threading
        t = threading.Thread(target=self.serve_client, args=(conn,), daemon=True)
        t.start()
        while t.is_alive() and not self.stop:
            try:
                extra, _ = self.lsock.accept()
            except (BlockingIOError, socket.timeout):
                t.join(0.2)
                continue
            try:
                send_msg(extra, {"ok": False, "error": "busy: another tracker owns the NPU"})
            except OSError:
                pass
            extra.close()
        if t.is_alive():                      # stopping: unblock the client thread
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            t.join(5.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--user", required=True, help="the only non-root user allowed to connect")
    ap.add_argument("--socket", default=SOCKET)
    ap.add_argument("--unpinned", action="store_true",
                    help="accept a retrained keypoint model (same graph) instead of the pinned Mercury hash")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    srv = Server(a.model, a.jobs, a.socket, pwd.getpwnam(a.user).pw_uid, require_pin=not a.unpinned)

    def stop(*_):
        srv.stop = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    srv.run()


if __name__ == "__main__":
    main()
