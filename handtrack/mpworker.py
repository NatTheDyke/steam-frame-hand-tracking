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


def _open_session(inference="cpu", framenpu_jobs=None, npu_socket=None):
    """One framenpu session (framenpu.open) for every model this worker runs: the keypoint model,
    the detector (when its file is here) and the 3D stereo student (when configured). On the NPU
    service each runs there when the service serves it, else on the CPU (loaded on first use)."""
    import framenpu
    from handtrack.mercury_kp import INFERENCE
    models = [(os.path.expanduser(_merc_model), "mercury-ft7")]
    names = {"keypoint": 0}
    det = os.path.expanduser(_det_model) if _det_model else None
    if det and os.path.exists(det):
        names["detector"] = len(models)
        models.append((det, "mercury-detector-160"))
    if _s3d_model:
        names["stereo3d"] = len(models)
        models.append((os.path.expanduser(_s3d_model), "mercury-stereo3d-tpc1"))
    s = framenpu.open(models, INFERENCE[inference], jobs=framenpu_jobs, socket=npu_socket)
    return s, {k: s.models[i] for k, i in names.items()}


_names = {}               # kind -> the session's model name (keypoint, detector, stereo3d)


def init_mercury():
    """Lazily load Monado Mercury's keypoint model (license UNVERIFIED: local measurement only)."""
    global _merc, _names
    if _merc is None:
        from handtrack import source as camsource
        from handtrack.calib import Rig
        from handtrack.mercury_kp import MercuryKeypoint
        src = camsource.active()          # HANDTRACK_SOURCE, inherited from live.py
        cfg = dict(_merc_config)
        session, _names = _open_session(cfg.pop("inference", "cpu"), cfg.pop("framenpu_jobs", None),
                                        cfg.pop("npu_socket", None))
        _merc = MercuryKeypoint(os.path.expanduser(_merc_model), Rig(), src.slot_cams, 640, source=src,
                                session=session, model=_names["keypoint"], **cfg)
    return _merc


def _session_model(mk, kind):
    """(session, model name) of `kind` in the keypoint model's session, or (None, None)"""
    s = getattr(mk, "session", None)
    name = _names.get(kind)
    return (s, name) if s is not None and name in getattr(s, "models", ()) else (None, None)


def init_mercury_proc(inference="cpu", framenpu_jobs=None, npu_socket=None, model=None, s3d_model=None,
                      cell_offset=0.5, det_model=None):
    """Initializer for the dedicated Mercury worker process (no MediaPipe in it). det_model: Mercury's
    hand detector for --discover mercury (run_discover), used on the CPU when the NPU service does
    not serve it."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    global _merc_config, _merc_model, _s3d_model, _det_model
    if model:
        _merc_model = model
    if det_model:
        _det_model = det_model
    if s3d_model:
        _s3d_model = s3d_model
    _merc_config = dict(inference=inference, framenpu_jobs=framenpu_jobs, npu_socket=npu_socket, cell_offset=cell_offset)
    init_mercury()
    from multiprocessing.util import Finalize
    Finalize(None, close_mercury, exitpriority=10)


_s3d = None
_s3d_model = None
_s3d_geo = None


def init_stereo3d():
    """the 3D stereo student (views = stereo3d) and the camera-to-camera transform, lazily"""
    global _s3d, _s3d_geo
    if _s3d is None:
        if not _s3d_model:
            raise RuntimeError("views stereo3d needs a 3D stereo student (--merc-s3d-model)")
        from handtrack import stereo3d
        from handtrack.calib import Rig
        from handtrack import source as camsource
        session, name = _session_model(init_mercury(), "stereo3d")
        _s3d = (stereo3d.SessionStereo3D(session, name) if session is not None
                else stereo3d.Stereo3D(os.path.expanduser(_s3d_model)))
        rig, src = Rig(), camsource.active()
        slot_of = {n: s for s, n in src.slot_cams.items()}
        v = [slot_of[n] for n in stereo3d.VIEW_CAMS]
        (R0, o0), (R1, o1) = (rig.head_from_cam(n) for n in stereo3d.VIEW_CAMS)
        _s3d_geo = (v, R1.T @ R0, R1.T @ (o0 - o1))
    return _s3d, _s3d_geo


def _many_session(mk, s3d):
    """the session when the keypoint model and the 3D student both run on its accelerator"""
    session, name = _session_model(mk, "stereo3d")
    if session is None or getattr(s3d, "session", None) is not session:
        return None
    on = session.backend_of
    return session if on(_names["keypoint"]) != "cpu" and on(name) != "cpu" else None


def _chunks(xs, k):
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def _many(session, ft7_feeds, pairs):
    """one round trip: the 3D pairs first (requests of <= 2), then the keypoint crops (requests of
    <= 3). The service runs each request's host work inside the DSP wait of its neighbour when it
    fits, so only the first prepare and the last finish are exposed. Ending on a cheap keypoint
    finish, and giving the 3D head's finish the wait of a 3-crop request, measured 15.3 ms for a
    two-hand frame vs 19.5 one request at a time (Frame, cores 5-6, 2026-11-21)."""
    import numpy as np
    from handtrack.inference import OUTPUTS, SHAPES
    kp_name, s3d_name = _names["keypoint"], _names["stereo3d"]
    reqs = [(s3d_name, c) for c in _chunks(pairs, 2)] + [(kp_name, c) for c in _chunks(ft7_feeds, 3)]
    res = session.infer_many(reqs)
    flat = [o for r in res for o in r]
    j3, kp = flat[:len(pairs)], flat[len(pairs):]
    if len(kp) != len(ft7_feeds) or len(j3) != len(pairs):
        raise ValueError("FrameNPU returned the wrong crop count")
    for crop in kp:
        for name, shape in zip(OUTPUTS, SHAPES):
            v = np.asarray(crop[name])
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError("FrameNPU returned invalid " + name)
    for o in j3:
        if np.asarray(o["joints3d"]).size != 63 or not np.isfinite(o["joints3d"]).all():
            raise ValueError("FrameNPU returned invalid joints3d")
    return kp, j3, session.last_timing or {}


def run_stereo3d(jobs, imgs, raw=False):
    """jobs: [(side, {slot: joints_cv (21,3)} for both cameras, expand, use_hint)]: both crops of each
    hand from the same capture -> the keypoint model on both crops and the 3D stereo student on the
    pair. On the NPU service the whole frame (every hand) is one infer_many round trip; otherwise
    each hand makes its own requests. Returns [(side, {slot: decoded result or None}, joints3d in
    view 0's camera (OpenCV) or None, is_hand_3d, view0_slot, seconds, timing)]."""
    import logging
    import time
    from handtrack import stereo3d
    global _many_off
    LOG = logging.getLogger(__name__)
    mk = init_mercury()
    s3d, (views, R10, t10) = init_stereo3d()
    hands = []
    for side, jcs, expand, use_hint in jobs:
        right = side == "right"
        preps = {sl: mk._prepare(sl, imgs[sl], jcs[sl], right, use_hint=use_hint, expand=expand, raw=raw) for sl in views}
        valid = [sl for sl in views if preps[sl] is not None]
        pair = None
        if len(valid) == len(views):
            pair = stereo3d.feeds(preps[views[0]], preps[views[1]], jcs[views[0]], right, R10, t10)
        hands.append((side, jcs, right, use_hint, preps, valid, pair))
    session = None if _many_off else _many_session(mk, s3d)
    if session is not None and any(h[5] for h in hands):
        s0, t0, td = _schedstat(), time.monotonic_ns(), time.perf_counter()
        try:
            kp, j3, timing = _many(session, [h[4][sl][-1] for h in hands for sl in h[5]],
                                   [h[6][0] for h in hands if h[6] is not None])
        except Exception as exc:
            # (the session latches its own fallback; this is a malformed answer) per hand from here on
            LOG.warning("NPU frame request failed; tracking continues per hand: %s", exc)
            _many_off = True
        else:
            kp_on, s3d_on = mk.backend.name, s3d.name
            dsec, t1, s1 = time.perf_counter() - td, time.monotonic_ns(), _schedstat()
            reqs = timing.get("requests") or [{}]
            out, ik, ij = [], 0, 0
            for n, (side, jcs, right, use_hint, preps, valid, pair) in enumerate(hands):
                res = {sl: None for sl in views}
                for sl in valid:
                    r = mk._decode(preps[sl], kp[ik])
                    ik += 1
                    r.pop("crop", None)
                    res[sl] = r
                J3, p3 = None, 0.0
                if pair is not None:
                    J3, p3 = stereo3d.finish(j3[ij], pair[1], pair[2])
                    ij += 1
                first = n == 0          # the call's CPU and wait are booked on the first hand
                out.append((side, res, J3, p3, views[0], dsec,
                            {"t0": t0, "t1": t1, "cpu_ns": (s1[0] - s0[0]) if first else 0,
                             "wait_ns": (s1[1] - s0[1]) if first else 0, "hint": use_hint,
                             "batch_crops": len(valid), "inference": kp_on, "s3d_inference": s3d_on,
                             "request_crops": len(valid), "frame_requests": len(reqs),
                             "dsp_body_ms": timing.get("body_ms", 0.0) if first else 0.0}))
            return out
    out = []
    for side, jcs, right, use_hint, preps, valid, pair in hands:
        s0, t0, td = _schedstat(), time.monotonic_ns(), time.perf_counter()
        outputs = mk.backend.infer([preps[sl][-1] for sl in valid]) if valid else []
        res = {sl: None for sl in views}
        for sl, o in zip(valid, outputs):
            r = mk._decode(preps[sl], o)
            r.pop("crop", None)
            res[sl] = r
        timing = (getattr(mk.backend, "last_timing", None) or {}) if valid else {}
        J3, p3 = None, 0.0
        if pair is not None:
            J3, p3 = s3d.run(preps[views[0]], preps[views[1]], jcs[views[0]], right, R10, t10)
        dsec, t1, s1 = time.perf_counter() - td, time.monotonic_ns(), _schedstat()
        out.append((side, res, J3, p3, views[0], dsec,
                    {"t0": t0, "t1": t1, "cpu_ns": s1[0] - s0[0], "wait_ns": s1[1] - s0[1], "hint": use_hint,
                     "batch_crops": len(valid), "inference": mk.backend.name, "s3d_inference": s3d.name,
                     "request_crops": timing.get("crops", len(valid)), "dsp_body_ms": timing.get("body_ms", 0.0)}))
    return out


_many_off = False


# ---------------------------------------------------------------- Mercury-detector discovery
_det = None               # ("session", (session, model name)) | ("cpu", onnxruntime session)
_det_geo = None
_det_model = "~/handtrack/models/mercury/grayscale_detection_160x160.onnx"


def init_detector():
    """The detector backend (the detector model of the keypoint model's framenpu session: the NPU
    service when it serves it, else its CPU; or ONNX Runtime on one thread without a session) and
    the head-frame camera geometry, lazily."""
    global _det, _det_geo
    if _det is None:
        session, name = _session_model(init_mercury(), "detector")
        _det = ("session", (session, name)) if session is not None else _det_cpu()
    if _det_geo is None:
        from handtrack import mercdet
        from handtrack import source as camsource
        from handtrack.calib import Rig
        _det_geo = mercdet.Geometry(camsource.active(), Rig())
    return _det, _det_geo


def _det_cpu():
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    return ("cpu", ort.InferenceSession(os.path.expanduser(_det_model), o, providers=["CPUExecutionProvider"]))


def _det_kind():
    """where the detector runs now (telemetry): framenpu | cpu"""
    kind, b = _det
    return b[0].backend_of(b[1]) if kind == "session" else kind


def _detect(feeds):
    """Detector outputs of the crops, in order: two crops per request, all requests in one round
    trip (infer_many). A failure latches the session to its CPU (framenpu.open's fallback)."""
    (kind, b), _ = init_detector()
    if kind == "session":
        session, name = b
        reqs = [(name, feeds[i:i + 2]) for i in range(0, len(feeds), 2)]
        outs = session.infer_many(reqs) if len(reqs) > 1 else [session.infer(name, reqs[0][1])]
        return [o for r in outs for o in r]
    sess = _det[1]
    names = [o.name for o in sess.get_outputs()]
    return [dict(zip(names, sess.run(None, f))) for f in feeds]


def _det_items(frames, layout, index=None):
    """[(slot, crop index, feed)] of both cameras' detector crops"""
    from handtrack import mercdet
    (_, _), geo = init_detector()
    return [(sl, i, f) for sl in (0, 1) for i, f in mercdet.feeds(geo.src, sl, frames[sl], layout, index)]


def _finish_discovery(frames, items, outs, layout, det_size, tracked, t_start):
    """Detector outputs of `items` -> run_discover's {slot: (hands, seconds)}: answers kept when the
    cameras agree, those not on a `tracked` hand (side -> head-frame palm centre) seeded by unhinted
    keypoint crops in both cameras (one more inference round trip, only for a new hand) and
    triangulated."""
    import time
    import types
    import numpy as np
    from handtrack import mercdet
    mk = init_mercury()
    _, geo = init_detector()
    kind = _det_kind()
    answers = []
    for (sl, i, _), o in zip(items, outs):
        answers += mercdet.decode(sl, [i], [o], layout)
    matches = mercdet.confirm(answers, geo)
    new = mercdet.untracked(matches, geo, tracked or {})
    preps, owners = [], []
    for side, m in new.items():
        p = mercdet.seed_inputs(mk, geo, m, frames)
        if p is not None:
            preps += p
            owners.append((side, m))
    decs = []
    from handtrack.mercury_kp import crops_per_call
    most = crops_per_call(mk.backend)
    for k in range(0, len(preps), most):
        group = preps[k:k + most]
        res = mk.backend.infer([g[-1] for g in group])
        decs += [mk._decode(g, r) for g, r in zip(group, res)]
    hands = {0: [], 1: []}
    for j, (side, m) in enumerate(owners):
        d0, d1 = decs[2 * j], decs[2 * j + 1]
        P = mercdet.seed_joints(geo, m, d0, d1)
        if P is None:
            continue
        for sl, a, d in ((0, m.a0, d0), (1, m.a1, d1)):
            R, t = geo.pose[sl]
            px = geo.src.raw_to_det(geo.cam[sl].project((P - t) @ R), sl, det_size)
            hands[sl].append(types.SimpleNamespace(
                side=side, side_score=a.p, px=px, rel_z=np.zeros(21), world=P.copy(), presence=1.0,
                extra={"source": "mercury-det", "detector": kind, "is_hand": d["is_hand"], "gap": m.gap,
                       "answers": len(answers), "matches": len(matches), "on_tracked": len(matches) - len(new)}))
    dt = time.perf_counter() - t_start
    return {sl: (hands[sl], dt) for sl in (0, 1)}


def run_discover(frames, det_size, ncrop=2, tracked=None):
    """Discovery with Mercury's hand detector (handtrack/mercdet.py) on the published frames of both
    cameras of one capture: the detector on `ncrop` fixed upright crops per camera, answers kept
    when both cameras agree, each kept answer not on a `tracked` hand seeded by unhinted keypoint
    crops in both cameras and triangulated. Returns MediaPipe's run() shape, {slot: (hands,
    seconds)}: per camera, hand records with side, side_score (that camera's detector
    probability), px (the seeded joints in the det_size detector image), rel_z, world (the
    joints, head frame) and extra (source, is_hand, gap), so live.py's stereo fusion seeds them
    like MediaPipe's. (While hands are tracked, discovery rides in run_mercury instead.)"""
    import time
    from handtrack import mercdet
    from handtrack.framering import frames as _frames
    frames = _frames(frames)          # a FrameRef (--frame-ring): views on the tracker's shared memory
    td = time.perf_counter()
    init_mercury()
    lay = mercdet.LAYOUTS[ncrop]
    items = _det_items(frames, lay)
    outs = _detect([f for _, _, f in items]) if items else []
    return _finish_discovery(frames, items, outs, lay, det_size, tracked, td)


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


def _ride(mk, observations, frames, disc):
    """The first crop group of a tracking job with discovery's detector crops along (one service
    frame request): (keypoint results, discovery result)."""
    import time
    from handtrack import mercdet
    td = time.perf_counter()
    lay = mercdet.LAYOUTS[disc.get("ncrop", 2)]
    items = _det_items(frames, lay, [disc.get("pick", 0) % len(lay)])
    prepared = [mk._prepare(**o) for o in observations]
    valid = [p for p in prepared if p is not None]
    det = None
    (kind, b), _ = init_detector()
    if valid and items and kind == "session":   # one frame request (one DSP job on a --multi image)
        session, name = b
        kp, det = session.infer_frame([(_names["keypoint"], [p[-1] for p in valid]),
                                       (name, [f for _, _, f in items])])
    else:
        kp = mk.backend.infer([p[-1] for p in valid]) if valid else []
    it = iter(kp)
    results = [None if p is None else mk._decode(p, next(it)) for p in prepared]
    if items and det is None:
        det = _detect([f for _, _, f in items])
    return results, (lambda: _finish_discovery(frames, items, det or [], lay, disc.get("det_size", 640),
                                               disc.get("tracked"), td))


def run_mercury(jobs, imgs, raw=False, discover=None):
    """jobs: [(slot, side, joints_cv (21,3))]; ROI from those joints, same images MediaPipe saw
    (raw=True: imgs are the published camera frames, sampled at full resolution).
    Returns [(slot, side, result-without-crop or None, seconds, timing)] where timing =
    {t0, t1: CLOCK_MONOTONIC ns around this crop (same clock as the tracker), cpu_ns: thread CPU
    time, wait_ns: runqueue wait (schedstat) during the crop group}.
    discover = {pick, ncrop, det_size, tracked} (raw=True, imgs of both cameras): Mercury-detector
    discovery on this capture rides along -- crop `pick` of the ncrop layout in both cameras, in
    the first group's service request -- and the return is {"obs": the list above, "discover":
    run_discover's result}."""
    import time
    from handtrack.framering import frames
    imgs = frames(imgs)               # a FrameRef (--frame-ring): views on the tracker's shared memory
    mk = init_mercury()
    out = []
    finish = None
    from handtrack.mercury_kp import crops_per_call
    most = crops_per_call(mk.backend)
    for start in range(0, len(jobs), most):
        group = jobs[start:start + most]
        observations = [dict(slot=j[0], det_img=imgs[j[0]], joints_cv=j[2],
                             flip=(j[1] == "right"), expand=j[3] if len(j) > 3 else None,
                             use_hint=j[4] if len(j) > 4 else True, raw=raw) for j in group]
        s0, t0, td = _schedstat(), time.monotonic_ns(), time.perf_counter()
        if discover is not None and start == 0:
            results, finish = _ride(mk, observations, imgs, discover)
        else:
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
                         "dsp_body_ms": timing.get("body_ms", 0.0), "discover": finish is not None}))
    if discover is not None:
        return {"obs": out, "discover": finish() if finish is not None else
                run_discover(imgs, discover.get("det_size", 640), discover.get("ncrop", 2), discover.get("tracked"))}
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
