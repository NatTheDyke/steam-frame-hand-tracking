"""Pluggable Mercury inference over normalized crops; CPU remains the default.

Backends implement infer(feeds) -> one output dict per feed, plus idle()/close().
A call contains 1..max_crops available crops (3, or 4 on a 4-crop FrameNPU image). There is no
batching timer or crop queue.
"""
import logging

import numpy as np

OUTPUTS = ("heatmap_xy", "heatmap_depth", "scalar_extras", "curls")
SHAPES = ((1, 21, 22, 22), (1, 21, 22), (1, 8), (1, 10))
LOG = logging.getLogger(__name__)


def crops_per_call(backend):
    """Crops one backend.infer call takes: its max_crops (3, or 4 on a 4-crop FrameNPU image)."""
    m = getattr(backend, "max_crops", 3)
    return m if isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= 4 else 3


class CPUBackend:
    name = "cpu"

    def __init__(self, model_path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

    max_crops = 4                 # one ONNX run per crop; any group size

    def infer(self, feeds):
        return [dict(zip(OUTPUTS, self.sess.run(list(OUTPUTS), f))) for f in feeds]

    def idle(self):
        pass

    def close(self):
        pass


class FrameNPUBackend:
    name = "framenpu"

    def __init__(self, model_path, jobs):
        # Optional dependency; CPU-only installations never import FrameNPU.
        from framenpu.mercury_runtime import MercurySession
        self.session = MercurySession(model_path, jobs)

    @property
    def last_timing(self):
        return self.session.last_timing

    @property
    def max_crops(self):
        return getattr(self.session, "max_crops", 3)

    def infer(self, feeds):
        return self.session.infer(feeds)

    def idle(self):
        self.session.idle()

    def close(self):
        self.session.close()


class ServiceBackend:
    """FrameNPU through the privileged NPU service (handtrack/npuservice.py), so the tracker itself
    runs as the user. The service owns the DSP session; this process only sends normalized crops."""
    name = "framenpu"

    def __init__(self, socket_path=None):
        from handtrack.npuservice import SOCKET, Client
        self.client = Client(socket_path or SOCKET)

    @property
    def last_timing(self):
        return self.client.last_timing

    @property
    def max_crops(self):
        return self.client.max_crops

    def infer(self, feeds):
        return self.client.infer(feeds)

    def idle(self):
        self.client.idle()

    def close(self):
        self.client.close()


class FallbackBackend:
    """Latch an accelerator failure to CPU for this worker's remaining lifetime.

    RecoveryRequired is never retried. The session's close leaves poisoned
    device bytes/refusal state intact; CPU inference uses no DSP resources.
    """
    def __init__(self, cpu, accelerator_factory):
        self.cpu, self.accelerator = cpu, None
        try:
            self.accelerator = accelerator_factory()
        except Exception as exc:
            LOG.warning("FrameNPU unavailable; using CPU: %s", exc)

    @property
    def name(self):
        return "framenpu" if self.accelerator is not None else "cpu"

    @property
    def last_timing(self):
        return getattr(self.accelerator, "last_timing", None)

    @property
    def max_crops(self):
        """crops per infer call: the accelerator's (3, or 4 on a 4-crop FrameNPU image); after a
        latch to the CPU, the CPU's"""
        return crops_per_call(self.accelerator if self.accelerator is not None else self.cpu)

    def _disable(self, exc):
        backend, self.accelerator = self.accelerator, None
        LOG.warning("FrameNPU disabled for this worker; using CPU: %s", exc)
        try:
            backend.close()
        except Exception as cleanup:
            LOG.warning("FrameNPU cleanup retained device state: %s", cleanup)

    @staticmethod
    def _check(out, n):
        if len(out) != n:
            raise ValueError("FrameNPU returned the wrong crop count")
        for crop in out:
            for name, shape in zip(OUTPUTS, SHAPES):
                v = np.asarray(crop[name])
                if v.shape != shape or not np.isfinite(v).all():
                    raise ValueError("FrameNPU returned invalid " + name)
        return out

    def infer(self, feeds):
        if not 1 <= len(feeds) <= self.max_crops:
            raise ValueError("inference calls carry one to %d crops" % self.max_crops)
        if self.accelerator is not None:
            try:
                return self._check(self.accelerator.infer(feeds), len(feeds))
            except Exception as exc:
                self._disable(exc)
        return self.cpu.infer(feeds)

    def infer_frame(self, feeds, extra):
        """Keypoint crops (1..max_crops) and extra = (service model, crops) in one service frame request (one
        DSP job) -> (keypoint outputs, the extra request's outputs). The extra outputs are None
        when the accelerator cannot take them along (no service, a service without infer_frame or
        that model, or after a failure, which latches this worker to the CPU as infer does): the
        caller runs that request itself."""
        if not 1 <= len(feeds) <= self.max_crops:
            raise ValueError("inference calls carry one to %d crops" % self.max_crops)
        client = getattr(self.accelerator, "client", None)
        if client is None or getattr(client, "max_frame", 0) < 2 or extra[0] not in client.models:
            return self.infer(feeds), None
        try:
            out, ex = client.infer_frame([(0, feeds), extra])
            if len(ex) != len(extra[1]):
                raise ValueError("FrameNPU returned the wrong crop count")
            return self._check(out, len(feeds)), ex
        except Exception as exc:
            self._disable(exc)
        return self.cpu.infer(feeds), None

    def idle(self):
        if self.accelerator is not None:
            try:
                self.accelerator.idle()
            except Exception as exc:
                self._disable(exc)

    def close(self):
        backend, self.accelerator = self.accelerator, None
        try:
            if backend is not None:
                backend.close()
        except Exception as exc:
            LOG.warning("FrameNPU cleanup retained device state: %s", exc)
        finally:
            self.cpu.close()


def make_backend(model_path, threads=1, inference="cpu", framenpu_jobs=None, npu_socket=None):
    """inference: cpu | framenpu (in-process, needs root) | service (the NPU service, as the user)."""
    if inference not in ("cpu", "framenpu", "service"):
        raise ValueError("unknown Mercury inference backend: " + inference)
    cpu = CPUBackend(model_path, threads)
    if inference == "cpu":
        return cpu
    if inference == "service":
        return FallbackBackend(cpu, lambda: ServiceBackend(npu_socket))
    return FallbackBackend(cpu, lambda: FrameNPUBackend(model_path, framenpu_jobs))
