#!/usr/bin/env python3
"""Replay a tools/record_frames.py recording through the Mercury tracking path, offline (CPU).

The live tracker's own telemetry, recorded with the frames, supplies what the rest of the pipeline
decided at each capture: the 60 Hz palm position (F rows: the flow tracker's position that steers
the Mercury crop and anchors single-view depth) and which frames got a Mercury crop from which
camera (R rows).  So the replay runs exactly the part under study, MercuryTracker + Mercury +
the palm orientation, on the frames the live run saw, without MediaPipe or the flow tracker.
The hand starts from a generic template at the live palm position (the first crop runs without a
keypoint hint), and is re-seeded the same way whenever the replay loses it.

  tools/replay_recording.py REC_DIR --model M.onnx [--side right] [--free-slots]

Library use: replay(rec_dir, model, side="right", setup=None) -> list of result dicts; setup(mt,
palm, mk) may change the tracker before the run (the experiments).
"""
import argparse
import os
import sys
from concurrent.futures import Future

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HANDTRACK_SOURCE", "slam")

from handtrack import mpworker                               # noqa: E402
from handtrack import source as camsource                    # noqa: E402
from handtrack.calib import Rig                              # noqa: E402
from handtrack.merctrack import MercuryTracker, HandState    # noqa: E402
from handtrack.mercury_kp import MercuryKeypoint             # noqa: E402
from handtrack.palm import PalmTracker                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_frames import read_frames                        # noqa: E402

DET = 640


class SyncPool:
    """ProcessPoolExecutor stand-in: runs the job now, returns a finished Future."""

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:                                # surfaced by MercuryTracker.collect
            f.set_exception(e)
        return f


def template_hand(centre):
    """Generic open right hand (21 joints, head frame, metres): palm facing down, fingers pointing
    away from the head, palm centre (joints 0, 5, 9, 17) at `centre`.  Only a crop seed."""
    P = np.zeros((21, 3))
    xs = (0.025, 0.008, -0.009, -0.025)                      # index, middle, ring, pinky MCP across
    for f, x in enumerate(xs):
        for k in range(4):
            P[5 + 4 * f + k] = (x, 0.0, -0.085 - 0.028 * k)
    for k in range(4):                                       # thumb out to the side
        P[1 + k] = (0.03 + 0.017 * k, 0.0, -0.025 - 0.02 * k)
    return P - P[[0, 5, 9, 17]].mean(0) + centre


def load_telemetry(path, side):
    """F rows -> {t_cap: (pos, pos_src)}; R rows -> {t_cap: [(slot, outcome, quat)]} for one hand."""
    F, R = {}, {}
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split(",")
            if c[0] == "F" and len(c) >= 18 and c[3] == side:
                F[int(c[1])] = (np.array([float(x) for x in c[11:14]]), c[4])
            elif c[0] == "R" and len(c) >= 17 and c[2] == side:
                R.setdefault(int(c[4]), []).append((int(c[3]), c[12], np.array([float(x) for x in c[13:17]])))
    return F, R


def nearest(table, tol_ns=1_000_000):
    """Lookup by capture time: telemetry times differ from the shared-memory ones by a few
    microseconds (frames are 33 ms apart), so take the nearest entry within tol_ns."""
    keys = np.array(sorted(table), dtype=np.int64)

    def get(t):
        if not len(keys):
            return None
        j = int(np.searchsorted(keys, t))
        best = min((k for k in (j - 1, j) if 0 <= k < len(keys)), key=lambda k: abs(int(keys[k]) - t))
        return table[int(keys[best])] if abs(int(keys[best]) - t) <= tol_ns else None
    return get


def replay(rec, model, side="right", setup=None, free_slots=False, progress=False):
    rig = Rig(os.path.join(rec, "calib", "sensorRigCalibration.json"))
    src = camsource.SOURCES["slam"]
    # detector-image rotation from the recording's calibration (Source.angle would read ~/handtrack)
    src._angles = {s: round(camsource.upright_angle(rig, n), 1) for s, n in src.slot_cams.items()}
    palm = PalmTracker(DET, rig=rig, source=src)
    mk = MercuryKeypoint(model, rig, src.slot_cams, DET, source=src)
    mpworker._merc = mk                                       # run_mercury uses this instance
    mt = MercuryTracker(palm, SyncPool(), mpworker)
    mt.views = "alternate"
    F, R = load_telemetry(os.path.join(rec, "telemetry.csv"), side)
    near_F, near_R = nearest(F), nearest(R)
    if setup is not None:
        setup(mt, palm, mk)
    out = []
    for idx, _seq, t0, _t1, img0, img1 in read_frames(rec):
        live = near_R(t0)
        if not live:                                          # the live run gave this frame no crop
            continue
        fp = near_F(t0)
        fast_pos = {side: (fp[0],)} if fp is not None and fp[1] == "stereo" else {}
        if side not in mt.hands:
            if fp is None:
                continue
            mt.hands[side] = HandState(side, template_hand(fp[0]), t0 / 1e9 - 0.05)
            mt.hands[side].misses = 1                         # recovery crop: no keypoint hint
            hint_reset, mt.hint_reset = mt.hint_reset, True
        else:
            hint_reset = None
        hs = mt.hands[side]
        if not free_slots:
            hs.next_slot = live[0][0]                         # the camera the live run used
        imgs = {s: src.det_image(img, s, DET) for s, img in ((0, img0), (1, img1))}
        mt.records.clear()
        mt.submit(imgs, t0, fast_pos, raw_imgs={0: img0, 1: img1})
        res = mt.collect()
        if hint_reset is not None:
            mt.hint_reset = hint_reset
        for rc in mt.records:
            P = mt.hands[side].P.copy() if side in mt.hands and rc["outcome"] in ("stereo", "mono") else None
            out.append({"idx": idx, "t": t0, "slot": rc["slot"], "outcome": rc["outcome"],
                        "rq": None if rc["rq"] is None else np.asarray(rc["rq"], float), "P": P,
                        "pose_q": np.asarray(res[side][0][1], float) if side in res else None,
                        "live": live[0], "fast": None if fp is None else fp[0]})
        if progress and idx % 500 == 0:
            print("frame %d, results %d" % (idx, len(out)), file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rec")
    ap.add_argument("--model", required=True)
    ap.add_argument("--side", default="right")
    ap.add_argument("--free-slots", action="store_true", help="alternate cameras freely instead of as live")
    a = ap.parse_args()
    out = replay(a.rec, a.model, a.side, free_slots=a.free_slots, progress=True)
    from collections import Counter
    print("results %d, outcomes %s" % (len(out), dict(Counter(r["outcome"] for r in out))))
    live = Counter(r["live"][1] for r in out)
    print("live outcomes on the same frames %s" % dict(live))


if __name__ == "__main__":
    main()
