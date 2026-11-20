"""MediaPipe in its own process.

Running MediaPipe (45-55 ms per run) in a thread of the main process stalled the
60 Hz tracking loop (Python GIL), which showed up as bursty, uneven motion.  The
worker process owns one HandLandmarker per camera; the main process sends it
640x640 gray frames and gets back plain Hand records.
"""
import os

def cpus_from_env():
    """CPU cores for our workload: HANDTRACK_CPUS like "0-2" or "0,1,2" (default 0-6, never the prime core 7)."""
    spec = os.environ.get("HANDTRACK_CPUS", "0-6")
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.add(int(part))
    return out or set(range(7))


_dets = {}


def init(gpu=False, mirrored=True):
    """ProcessPoolExecutor initializer: lower priority, off the prime core, build the detectors."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    from handtrack.detect import HandDetector
    for slot in (0, 1):
        _dets[slot] = HandDetector(gpu=gpu, mirrored=mirrored)


_merc = None
_merc_config = {}
_merc_model = "~/handtrack/models/mercury/grayscale_keypoint_jan18.onnx"


def init_mercury():
    """Lazily load Monado Mercury's keypoint model (license UNVERIFIED: local measurement only)."""
    global _merc
    if _merc is None:
        from handtrack import source as camsource
        from handtrack.calib import Rig
        from handtrack.mercury_kp import MercuryKeypoint
        src = camsource.active()          # HANDTRACK_SOURCE, inherited from live.py
        _merc = MercuryKeypoint(os.path.expanduser(_merc_model),
                                Rig(), src.slot_cams, 640, source=src, **_merc_config)
    return _merc


def init_mercury_proc(inference="cpu", framenpu_jobs=None, npu_socket=None, model=None):
    """Initializer for the dedicated Mercury worker process (no MediaPipe in it)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    global _merc_config, _merc_model
    if model:
        _merc_model = model
    _merc_config = dict(inference=inference, framenpu_jobs=framenpu_jobs, npu_socket=npu_socket)
    init_mercury()
    from multiprocessing.util import Finalize
    Finalize(None, close_mercury, exitpriority=10)


def idle_mercury():
    if _merc is not None:
        _merc.idle()


def close_mercury():
    global _merc
    if _merc is not None:
        current, _merc = _merc, None
        current.close()


def _schedstat():
    """(on-CPU ns, runqueue-wait ns) of the calling thread, or (0, 0) if unavailable."""
    try:
        with open("/proc/thread-self/schedstat") as f:
            c, w = f.read().split()[:2]
        return int(c), int(w)
    except (OSError, ValueError):
        return 0, 0


def run_mercury(jobs, imgs, raw=False):
    """jobs: [(slot, side, joints_cv (21,3))]; ROI from those joints, same images MediaPipe saw
    (raw=True: imgs are the published camera frames, sampled at full resolution).
    Returns [(slot, side, result-without-crop or None, seconds, timing)] where timing =
    {t0, t1: CLOCK_MONOTONIC ns around this crop (same clock as the tracker), cpu_ns: thread CPU
    time, wait_ns: runqueue wait (schedstat) during the crop group}."""
    import time
    mk = init_mercury()
    out = []
    for start in range(0, len(jobs), 3):
        group = jobs[start:start + 3]
        observations = [dict(slot=j[0], det_img=imgs[j[0]], joints_cv=j[2],
                             flip=(j[1] == "right"), expand=j[3] if len(j) > 3 else None,
                             use_hint=j[4] if len(j) > 4 else True, raw=raw) for j in group]
        s0, t0, td = _schedstat(), time.monotonic_ns(), time.perf_counter()
        results = mk.run_many(observations)
        dsec, t1, s1 = time.perf_counter() - td, time.monotonic_ns(), _schedstat()
        timing = (getattr(mk.backend, "last_timing", None) or {}) if any(r is not None for r in results) else {}
        for job, observation, r in zip(group, observations, results):
            if r is not None:
                r.pop("crop", None)
            # Group wall time is the latency of each observation, not wall/count.
            out.append((job[0], job[1], r, dsec,
                        {"t0": t0, "t1": t1, "cpu_ns": s1[0] - s0[0],
                         "wait_ns": s1[1] - s0[1], "hint": observation["use_hint"],
                         "batch_crops": len(group), "inference": mk.backend.name,
                         "request_crops": timing.get("crops", sum(r is not None for r in results)),
                         "dsp_body_ms": timing.get("body_ms", 0.0)}))
    return out


def run(use, imgs, t_s):
    """Detect hands in imgs[slot] for each slot in `use`; returns {slot: (hands, seconds)}."""
    import time
    out = {}
    for sl in use:
        td = time.perf_counter()
        hs = _dets[sl].detect(imgs[sl], t_s)
        out[sl] = (hs, time.perf_counter() - td)
    return out
