"""Cross-repository performance: does one set of libraries cost more CPU than another?

    perf.py bench [--only NAME ...]               one round in this environment, JSON lines
    perf.py compare --base PY [--cand PY] [--rounds N] [--base-tracker DIR] [--base-cdsp-tests DIR]

compare runs both environments' benches alternately on this machine (order swapped every round) and
compares CPU time per operation, round by round. CPU time of a shared machine still drifts by more than
a small regression, so the verdict is the paired ratio's 95% interval: a bench fails when the whole
interval is above --limit (default 1.10). What was not measurable is reported as such, never as a pass.

Needs FI_MODELS, and FI_CDSP_TESTS for the device host path. Not a model of the headset: x86 runs
framenpu's NumPy host path, an aarch64 machine its native one; the DSP's own time is not in here
(headset.py measures the real thing)."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
os.environ.setdefault("FRAMEHANDS_CORE", "native")   # as headset/start.sh
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")                     # as the tracker and the service set them
KEYPOINT, DETECTOR = "student_ft7.onnx", "grayscale_detection_160x160.onnx"


def per_op(f, n, k=3):
    """CPU ms per call of f (all threads of this process): the best of k runs of n calls"""
    f()
    best = math.inf
    for _ in range(k):
        t = time.process_time()
        for _ in range(n):
            f()
        best = min(best, time.process_time() - t)
    return 1e3 * best / n


def models():
    return [str(Path(os.environ["FI_MODELS"]) / n) for n in (KEYPOINT, DETECTOR)]


def feeds(n):
    import numpy as np
    rng = np.random.default_rng(1)
    return [{"inputImg": rng.uniform(-1, 2, (1, 1, 128, 128)).astype(np.float32),
             "lastKeypoints": rng.uniform(0, 1, (1, 42)).astype(np.float32), "useLastKeypoints": np.ones(1, np.float32)}
            for _ in range(n)]


def wire():
    """One 2-crop request through the socket, both ends (no model)."""
    import tempfile
    import numpy as np
    from framenpu import service

    class Zero:
        last_timing = {"crops": 2, "body_ms": 1.0}

        def infer(self, crops, model=0):
            return [{n: np.zeros(s, np.float32) for n, s in service.SPECS[model][1]} for _ in crops]

        idle = close = lambda self: None
    with tempfile.TemporaryDirectory(prefix="fh") as d:
        path = os.path.join(d, "s")
        srv = service.Server(models()[0], None, path, os.getuid(), session_factory=Zero, crops=4)
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        while not os.path.exists(path):
            time.sleep(0.01)
        c, f = service.Client(path), feeds(2)
        try:
            return per_op(lambda: c.infer(f), 1500, 7)      # two threads hand over: noisy
        finally:
            c.close()
            srv.stop = True
            t.join(3)


def device():
    """One request's host side on the device path (staging, layout, readback, host heads); the simulated
    DSP does nothing. Two keypoint crops, and a frame with two detector crops too."""
    sys.path.insert(0, os.environ["FI_CDSP_TESTS"])
    os.environ["FRAMENPU_DEVICES"] = "cdsp=sim_plugin:SimPlugin"
    import framenpu
    import numpy as np
    import teacher_resident_layout as oracle
    oracle.simulate = lambda *a, **k: None
    m = models()
    s = framenpu.open([(m[0], "mercury-ft7"), (m[1], "mercury-detector-160")], "device")
    try:
        kp, det = feeds(2), [{"inputImg": np.random.default_rng(i).uniform(-1, 2, (1, 1, 160, 160)).astype(np.float32)} for i in (0, 1)]
        out = {"device 2 crops": per_op(lambda: s.infer("mercury-ft7", kp), 40),
               "device frame 2+2": per_op(lambda: s.infer_frame([("mercury-ft7", kp), ("mercury-detector-160", det)]), 20)}
        if [s.backend_of(x) for x in s.models] != ["framenpu"] * 2:
            raise RuntimeError("the device path fell back to the CPU")
        return out
    finally:
        s.close()


class _Canned:
    """A framenpu session that answers at once."""
    last_timing = None
    name, models, max = "framenpu", ("mercury-ft7",), 4

    def __init__(self, out):
        self.out = out

    def infer(self, model, crops):
        return [self.out] * len(crops)

    def max_crops(self, model):
        return 4

    def backend_of(self, model):
        return "framenpu"

    def idle(self):
        pass

    close = idle


def _rig(d):
    sys.path.insert(0, str(HERE))
    import cv2
    cv2.setNumThreads(1)                              # as the tracker's processes
    import world
    from framehands.calib import Rig
    from framehands import source
    p = os.path.join(d, "c.json")
    world.write_calibration(p)
    rig, src = Rig(p), source.SOURCES["slam"]
    src._angles = {k: round(source.upright_angle(rig, n), 1) for k, n in src.slot_cams.items()}
    return world, rig, src


def keypoint():
    """framehands around the model: two crops sampled from full camera frames, decoded to joints."""
    import tempfile
    import numpy as np
    from framehands.mercury_kp import MercuryKeypoint
    with tempfile.TemporaryDirectory() as d:
        world, rig, src = _rig(d)
        hands = world.truth(0)
        imgs = world.render(hands)
        mk = MercuryKeypoint(None, rig, src.slot_cams, 640, source=src, session=_Canned(world.keypoints(np.zeros((128, 128), np.float32))[0]),
                             model="mercury-ft7", cell_offset=0)
        obs = []
        for k, (side, P) in enumerate(hands.items()):
            R, t = rig.head_from_cam(src.slot_cams[k])
            obs.append(dict(slot=k, det_img=imgs[k], raw=True, joints_cv=(P - t) @ R, flip=side == "right", use_hint=True, expand=1.7))
        return per_op(lambda: mk.run_many(obs), 100)


def detector():
    """Detector crops of one camera frame (two crops) and their decoding."""
    import tempfile
    from framehands import mercdet
    with tempfile.TemporaryDirectory() as d:
        world, _, src = _rig(d)
        img = world.render(world.truth(0))[0]
        lay = mercdet.LAYOUTS[2]

        def once():
            f = mercdet.feeds(src, 0, img, lay)
            mercdet.decode(0, [i for i, _ in f], [canned] * len(f), lay)
        canned = world.detect(mercdet.feeds(src, 0, img, lay)[0][1]["inputImg"][0, 0])
        return per_op(once, 60)


def loop(seconds=8.0):
    """The whole tracker: CPU ms per camera frame of handtrack.live and its worker, two hands tracked."""
    sys.path.insert(0, str(HERE))
    import stack
    why = stack.blocked()
    if why:
        raise RuntimeError(why)
    with stack.Stack(models()) as s:
        if not s.wait(s.tracked, 40):
            raise RuntimeError("no hands tracked\n" + s.log("live")[-2000:])
        time.sleep(2)
        n, c = len(s.driver.packets), s.cpu_s("live")
        time.sleep(seconds)
        got, used = s.driver.packets[n:], s.cpu_s("live") - c
        if "using CPU" in s.log("live"):
            raise RuntimeError("the tracker fell back to the CPU")
        if sum(all(h["valid"] for h in p[3].values()) for p in got) < 0.95 * len(got):
            raise RuntimeError("the hands were not tracked all through")
        if len(got) < 0.9 * seconds * s.fps:
            raise RuntimeError("only %d of %d frames" % (len(got), seconds * s.fps))
        age = sorted(p[0] - p[2] for p in got)
        return {"loop cpu/frame": 1e3 * used / len(got), "loop latency p50 (wall)": age[len(age) // 2] / 1e6}


BENCHES = dict(wire=wire, device=device, keypoint=keypoint, detector=detector, loop=loop)
WALL = "(wall)"


def bench(only):
    for name in only or BENCHES:
        try:
            v = BENCHES[name]()
        except Exception as e:
            print(json.dumps({"bench": name, "error": "%s: %s" % (type(e).__name__, e)}), flush=True)
            continue
        for k, x in (v.items() if isinstance(v, dict) else [(name, v)]):
            print(json.dumps({"bench": k, "ms": x}), flush=True)


def round_of(python, env, only):
    p = subprocess.run([python, str(HERE / "perf.py"), "bench", *(["--only", *only] if only else [])], env=env,
                       capture_output=True, text=True, timeout=600)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            out[d["bench"]] = d.get("ms", d.get("error"))
    if p.returncode:
        raise SystemExit("bench failed:\n" + p.stderr[-3000:])
    return out


def interval(logs, rng):
    """Median of the paired ratios (one disturbed round must not decide) and its 95% bootstrap interval."""
    boot = sorted(statistics.median(rng.choices(logs, k=len(logs))) for _ in range(4000))
    return [math.exp(v) for v in (statistics.median(logs), boot[100], boot[-101])]


def compare(a):
    base, cand = dict(os.environ), dict(os.environ)
    if a.base_tracker:
        base["FI_TRACKER"] = str(Path(a.base_tracker).resolve())
    if a.base_cdsp_tests:
        base["FI_CDSP_TESTS"] = str(Path(a.base_cdsp_tests).resolve())
    arms = {"base": (a.base, base), "cand": (a.cand, cand)}
    rows = {"base": [], "cand": []}
    for k in range(a.rounds):
        for arm in (("base", "cand") if k % 2 == 0 else ("cand", "base")):
            rows[arm].append(round_of(*arms[arm], a.only))
        print("round %d of %d" % (k + 1, a.rounds), file=sys.stderr, flush=True)
    if a.cpus:
        os.sched_setaffinity(0, [int(c) for c in a.cpus.split(",")])      # inherited by every bench
    if a.rounds < 6:
        print("fewer than 6 rounds: the interval is not one", file=sys.stderr)
    rng, result, bad = random.Random(1), {}, []
    names = list(dict.fromkeys(n for r in rows["base"] + rows["cand"] for n in r))
    print("| benchmark | base ms | candidate ms | ratio | 95% interval | |\n|---|---:|---:|---:|---|---|")
    for n in names:
        errs = [r.get(n) for arm in rows for r in rows[arm] if not isinstance(r.get(n), (int, float))]
        if errs:
            print("| %s | | | | | NOT MEASURED: %s |" % (n, str(errs[0])[:160]))
            result[n] = {"error": str(errs[0])}
            bad.append(n)
            continue
        b, c = ([r[n] for r in rows[arm]] for arm in ("base", "cand"))
        ratio, lo, hi = interval([math.log(y / x) for x, y in zip(b, c)], rng)
        gated = WALL not in n
        verdict = ("REGRESSION" if gated and lo > a.limit else "faster" if hi < 1 / a.limit else
                   "ok" if hi <= a.limit else "unresolved") + ("" if gated else " (reported only)")
        if verdict == "REGRESSION":
            bad.append(n)
        mid = lambda v: sorted(v)[len(v) // 2]
        print("| %s | %.3f | %.3f | %.3f | %.3f-%.3f | %s |" % (n, mid(b), mid(c), ratio, lo, hi, verdict))
        result[n] = dict(base=b, cand=c, ratio=ratio, low=lo, high=hi, verdict=verdict)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--only", nargs="+", choices=BENCHES)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, help="python of the environment to compare against")
    c.add_argument("--cand", default=sys.executable)
    c.add_argument("--base-tracker", help="the base's tracker checkout (default: this one)")
    c.add_argument("--base-cdsp-tests", help="the base's framenpu-cdsp tests directory (default: FI_CDSP_TESTS)")
    c.add_argument("--cpus", help="run on these CPUs only (0,1,2): needed where cores differ in speed")
    c.add_argument("--rounds", type=int, default=10)
    c.add_argument("--limit", type=float, default=1.10)
    c.add_argument("--only", nargs="+", choices=BENCHES)
    c.add_argument("--json")
    a = ap.parse_args()
    return bench(a.only) if a.cmd == "bench" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
