"""Pluggable Mercury inference over normalized crops; CPU remains the default.

Backends implement infer(feeds) -> one output dict per feed, plus idle()/close().
A call contains 1..3 available crops. There is no batching timer or crop queue.
"""
import logging

import numpy as np

OUTPUTS = ("heatmap_xy", "heatmap_depth", "scalar_extras", "curls")
SHAPES = ((1, 21, 22, 22), (1, 21, 22), (1, 8), (1, 10))
LOG = logging.getLogger(__name__)


class CPUBackend:
    name = "cpu"

    def __init__(self, model_path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

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

    def infer(self, feeds):
        return self.session.infer(feeds)

    def idle(self):
        self.session.idle()

    def close(self):
        self.session.close()


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

    def _disable(self, exc):
        backend, self.accelerator = self.accelerator, None
        LOG.warning("FrameNPU disabled for this worker; using CPU: %s", exc)
        try:
            backend.close()
        except Exception as cleanup:
            LOG.warning("FrameNPU cleanup retained device state: %s", cleanup)

    def infer(self, feeds):
        if not 1 <= len(feeds) <= 3:
            raise ValueError("inference calls carry one to three crops")
        if self.accelerator is not None:
            try:
                out = self.accelerator.infer(feeds)
                if len(out) != len(feeds):
                    raise ValueError("FrameNPU returned the wrong crop count")
                for crop in out:
                    for name, shape in zip(OUTPUTS, SHAPES):
                        v = np.asarray(crop[name])
                        if v.shape != shape or not np.isfinite(v).all():
                            raise ValueError("FrameNPU returned invalid " + name)
                return out
            except Exception as exc:
                self._disable(exc)
        return self.cpu.infer(feeds)

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


def make_backend(model_path, threads=1, inference="cpu", framenpu_jobs=None):
    if inference not in ("cpu", "framenpu"):
        raise ValueError("unknown Mercury inference backend: " + inference)
    cpu = CPUBackend(model_path, threads)
    if inference == "cpu":
        return cpu
    return FallbackBackend(cpu, lambda: FrameNPUBackend(model_path, framenpu_jobs))
