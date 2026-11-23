"""FrameNPU Mercury inference as a small privileged service, so the tracker can run as the user.

The DSP needs /dev/mem (root). Instead of running the whole tracker as root, this service owns
the FrameNPU MercurySession and answers inference requests over a Unix socket that only the
tracking user (and root) may use. The tracker's `--merc-inference service` backend
(handtrack/inference.py ServiceBackend) is the only client; any failure there falls back to CPU.

Server (root, normally via systemd: headset/install_npu_service.sh):
    python -I npuservice.py --model M.onnx --jobs DIR --user steamos [--socket PATH]
                            [--s3d-model S3D.onnx]   (a --res-slots image: the 3D stereo student too)
                            [--det-model DET.onnx]   (the packed-bias image: Mercury's hand detector too)

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
           {"op": "infer", "n": k, "model": 1}  (k <= 2) payload = k x (inputImg [1,2,128,128],
               lastKeypoints [1,84], useLastKeypoints, geom [1,16]): the 3D stereo student
           {"op": "infer_many", "reqs": [{"model": m, "n": k}, ...]} (1..MAX_SEQ requests): the
               requests' payloads back to back; the service overlaps each request's host work
               with the DSP running its neighbour (MercurySession.infer_many)
           {"op": "infer", "n": k, "model": 2}  (k <= 2) payload = k x inputImg [1,1,160,160]: Mercury's
               hand detector (a --det-model service)
           {"op": "infer_frame", "reqs": [...]} as infer_many, 1..MAX_FRAME requests of distinct models:
               one DSP job on a --multi image (streamed on a --stream one: the second request is
               prepared, and the first finished, while the DSP runs), else as infer_many
           {"op": "idle"} | {"op": "ping"}
  response {"ok": true, "timing": {...}}  payload = k x (heatmap_xy, heatmap_depth,
           scalar_extras, curls), for model 1 k x (joints3d [1,63], scalar_extras), for model 2
           k x (hand_exists, cx, cy, size, each [1,2]: left, right); infer_many / infer_frame:
           each request's outputs back to back, "timing": {"requests": [...]};
           {"ok": false, "error": "..."} on failure.  ping lists the served "models".
The served models run in one FrameNPU session, each with its own resident parameters (no reload
when the tracker alternates between them). The 3D student and the detector do not fit together.
"""
import argparse
import json
import logging
import os
import pwd
import signal
import socket
import struct

# One BLAS thread: the host heads are small matmuls, and OpenBLAS worker threads spin between them.
# Measured on the Frame (2026-11-20): 11.8 -> 4.4 ms of service CPU per crop.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

SOCKET = "/run/mrhand-npu/npu.sock"
INPUTS = (("inputImg", (1, 1, 128, 128)), ("lastKeypoints", (1, 42)), ("useLastKeypoints", (1,)))
OUTPUTS = (("heatmap_xy", (1, 21, 22, 22)), ("heatmap_depth", (1, 21, 22)), ("scalar_extras", (1, 8)),
           ("curls", (1, 10)))
S3D_INPUTS = (("inputImg", (1, 2, 128, 128)), ("lastKeypoints", (1, 84)), ("useLastKeypoints", (1,)),
              ("geom", (1, 16)))
S3D_OUTPUTS = (("joints3d", (1, 63)), ("scalar_extras", (1, 8)))
DET_INPUTS = (("inputImg", (1, 1, 160, 160)),)
DET_OUTPUTS = (("hand_exists", (1, 2)), ("cx", (1, 2)), ("cy", (1, 2)), ("size", (1, 2)))
SPECS = {0: (INPUTS, OUTPUTS, 3), 1: (S3D_INPUTS, S3D_OUTPUTS, 2),   # model -> (inputs, outputs, most crops)
         2: (DET_INPUTS, DET_OUTPUTS, 2)}
MAX_FRAME = 2                 # requests per infer_frame (FrameNPU MULTI_SEGMENTS)
MAX_SEQ = 4                   # requests per infer_many (two hands: ft7 3 + 1 crops, the 3D pairs)
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


def _infer_model(header):
    m = header.get("model", 0)
    if m not in SPECS:
        raise ValueError("unknown model")
    return m


def _infer_count(header):
    n = header.get("n", 0) if header.get("op") == "infer" else 0
    if header.get("op") == "infer":
        most = SPECS[_infer_model(header)][2]
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= most:
            raise ValueError("infer carries one to %d crops" % most)
    return n


def _seq(header):
    """infer_many's / infer_frame's [(model, n), ...], validated"""
    frame = header.get("op") == "infer_frame"
    most = MAX_FRAME if frame else MAX_SEQ
    reqs = header.get("reqs")
    if not isinstance(reqs, list) or not 1 <= len(reqs) <= most or not all(isinstance(r, dict) for r in reqs):
        raise ValueError("%s carries one to %d requests" % (header.get("op"), most))
    seq = [(_infer_model(r), _infer_count(dict(r, op="infer"))) for r in reqs]
    if frame and len({m for m, _ in seq}) != len(seq):
        raise ValueError("infer_frame's requests are of distinct models")
    return seq


def _in_bytes(header):
    if header.get("op") in ("infer_many", "infer_frame"):
        return sum(n * _nbytes(SPECS[m][0]) for m, n in _seq(header))
    return _infer_count(header) * _nbytes(SPECS[_infer_model(header)][0]) if header.get("op") == "infer" else 0


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

    @property
    def models(self):
        """served model numbers: 0 the keypoint model, 1 the 3D stereo student, 2 the detector"""
        return tuple(self.info.get("models", (0,)))

    def _call(self, header, payload=b"", out_count=0, out_spec=OUTPUTS):
        send_msg(self.sock, header, payload)
        resp, data = recv_msg(self.sock, lambda h: out_count * _nbytes(out_spec) if h.get("ok") else 0)
        if not resp.get("ok"):
            raise RuntimeError("NPU service: %s" % resp.get("error", "failed"))
        return resp, data

    @property
    def max_seq(self):
        """requests per infer_many (0: the service predates it)"""
        return int(self.info.get("infer_many", 0))

    @property
    def max_frame(self):
        """requests per infer_frame (0: the service predates it)"""
        return int(self.info.get("infer_frame", 0))

    def infer_frame(self, requests):
        """[(model, feeds), ...] of distinct models (up to max_frame) -> each request's outputs: one
        round trip and, on a --multi FrameNPU image, one DSP job."""
        if not 1 <= len(requests) <= self.max_frame:
            raise ValueError("infer_frame carries one to %d requests" % self.max_frame)
        if len({m for m, _ in requests}) != len(requests):
            raise ValueError("infer_frame's requests are of distinct models")
        return self.infer_many(requests, op="infer_frame")

    def infer_many(self, requests, op="infer_many"):
        """[(model, feeds), ...] -> each request's outputs; one round trip, the service overlaps
        the requests' host work with the DSP."""
        if op == "infer_many" and not 1 <= len(requests) <= self.max_seq:
            raise ValueError("infer_many carries one to %d requests" % self.max_seq)
        reqs, payload, out_n = [], [], 0
        for model, feeds in requests:
            ins, outs, most = SPECS[model]
            if not 1 <= len(feeds) <= most:
                raise ValueError("inference calls carry one to %d crops" % most)
            if model not in self.models:
                raise RuntimeError("NPU service does not serve model %d" % model)
            reqs.append({"model": model, "n": len(feeds)})
            payload.append(pack_arrays(feeds, ins))
            out_n += len(feeds) * _nbytes(outs)
        self.sock.settimeout(self.timeout if self.opened else self.open_timeout)
        send_msg(self.sock, {"op": op, "reqs": reqs}, b"".join(payload))
        resp, data = recv_msg(self.sock, lambda h: out_n if h.get("ok") else 0)
        if not resp.get("ok"):
            raise RuntimeError("NPU service: %s" % resp.get("error"))
        self.opened = True
        self.sock.settimeout(self.timeout)
        self.last_timing = resp.get("timing")
        res, at = [], 0
        for model, feeds in requests:
            size = len(feeds) * _nbytes(SPECS[model][1])
            res.append(unpack_arrays(data[at:at + size], len(feeds), SPECS[model][1]))
            at += size
        return res

    def infer(self, feeds, model=0):
        ins, outs, most = SPECS[model]
        if not 1 <= len(feeds) <= most:
            raise ValueError("inference calls carry one to %d crops" % most)
        if model not in self.models:
            raise RuntimeError("NPU service does not serve model %d" % model)
        self.sock.settimeout(self.timeout if self.opened else self.open_timeout)
        header = {"op": "infer", "n": len(feeds)} | ({"model": model} if model else {})
        resp, data = self._call(header, pack_arrays(feeds, ins), len(feeds), outs)
        self.opened = True
        self.sock.settimeout(self.timeout)
        self.last_timing = resp.get("timing")
        return unpack_arrays(data, len(feeds), outs)

    def idle(self):
        self._call({"op": "idle"})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, model, jobs, path, uid, session_factory=None, require_pin=True, s3d_model=None,
                 det_model=None):
        self.model, self.jobs, self.path, self.uid = model, jobs, path, uid
        if s3d_model and det_model:
            raise ValueError("the 3D student and the detector do not fit one FrameNPU image together")
        extra = [(m, p) for m, p in ((1, s3d_model), (2, det_model)) if p]
        self.models = (0,) + tuple(m for m, _ in extra)
        # service model number -> the session's model index (extra models follow the keypoint model)
        self.index = {0: 0} | {m: k + 1 for k, (m, _) in enumerate(extra)}
        if session_factory is None:
            def session_factory():
                from framenpu.mercury_runtime import MercurySession
                return MercurySession(self.model, self.jobs, require_pin=require_pin,
                                      extra_models=[(p, SPECS[m][2]) for m, p in extra])
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
            return {"ok": True, "service": "mrhand-npu", "version": 4, "models": list(self.models),
                    "infer_many": MAX_SEQ, "infer_frame": MAX_FRAME}, b""
        if op == "idle":
            if self.session is not None:
                self.session.idle()
            return {"ok": True}, b""
        if op == "infer":
            n, model = _infer_count(header), _infer_model(header)
            if model not in self.models:
                raise ValueError("model %d is not served" % model)
            ins, outspec, _ = SPECS[model]
            feeds = unpack_arrays(payload, n, ins)
            if self.session is None:
                self.session = self.session_factory()
                LOG.info("session opened")
            outs = self.session.infer(feeds, model=self.index[model]) if model else self.session.infer(feeds)
            if len(outs) != n:
                raise ValueError("FrameNPU returned the wrong crop count")
            timing = getattr(self.session, "last_timing", None)
            return {"ok": True, "timing": json.loads(json.dumps(timing, default=float)) if timing else None}, \
                pack_arrays(outs, outspec)
        if op in ("infer_many", "infer_frame"):
            seq, at, reqs = _seq(header), 0, []
            for model, n in seq:
                if model not in self.models:
                    raise ValueError("model %d is not served" % model)
                size = n * _nbytes(SPECS[model][0])
                reqs.append((self.index[model], unpack_arrays(payload[at:at + size], n, SPECS[model][0])))
                at += size
            if self.session is None:
                self.session = self.session_factory()
                LOG.info("session opened")
            outs = (self.session.infer_frame if op == "infer_frame" else self.session.infer_many)(reqs)
            if len(outs) != len(seq) or any(len(o) != n for o, (_, n) in zip(outs, seq)):
                raise ValueError("FrameNPU returned the wrong crop count")
            timing = getattr(self.session, "last_timing", None)
            return {"ok": True, "timing": json.loads(json.dumps(timing, default=float)) if timing else None}, \
                b"".join(pack_arrays(o, SPECS[m][1]) for o, (m, _) in zip(outs, seq))
        raise ValueError("unknown op")

    def serve_client(self, conn):
        conn.settimeout(None)
        try:
            while not self.stop:
                try:
                    header, payload = recv_msg(conn, _in_bytes)
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
    ap.add_argument("--s3d-model", default=None,
                    help="also serve this 3D stereo student (needs the --res-slots FrameNPU image)")
    ap.add_argument("--det-model", default=None,
                    help="also serve Mercury's hand detector (needs the packed-bias FrameNPU image)")
    ap.add_argument("--unpinned", action="store_true",
                    help="accept a retrained keypoint model (same graph) instead of the pinned Mercury hash")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    srv = Server(a.model, a.jobs, a.socket, pwd.getpwnam(a.user).pw_uid, require_pin=not a.unpinned,
                 s3d_model=a.s3d_model, det_model=a.det_model)

    def stop(*_):
        srv.stop = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    srv.run()


if __name__ == "__main__":
    main()
