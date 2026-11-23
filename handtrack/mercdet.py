"""Hand discovery with Monado Mercury's own hand detector (grayscale_detection_160x160.onnx), as an
alternative to MediaPipe discovery (live.py --discover mercury; MediaPipe stays the default).

The detector answers, per 160x160 crop, a probability, a centre and a size for each side (outputs
hand_exists, cx, cy, size; index 0 = left, 1 = right).  It only works on upright, tight crops: the
crops here are fixed squares of the 1024 upright detector image (handtrack.source.det_image), chosen
offline on Frametop takes (distill det_tile.py / det_tile_report.py): one 640 crop finds 65% of
hands, two (640 + 384) 81%, four 85%, against MediaPipe's 14% on the same images.

Its "is_hand" style confidence does not separate hands from clutter (distill det_verify.py), so an
answer is only used when the other SLAM camera agrees: a same-side answer whose ray passes within
GAP_M and triangulates to DIST_M from the head (distill det_stereo.py: hand-free answers ~5x down,
none of the eye-checked survivors truly false).

A confirmed answer seeds tracking through the keypoint model: an upright, unhinted crop centred on
each camera's answer, SEED_K x the size estimate wide (det_verify.py: ft7 median joint error 0.08
hand size), and the two same-instant views triangulated joint by joint -> 21 head-frame joints for
MercuryTracker.seed.

Pure geometry and array work (NumPy + OpenCV); inference is the caller's (the NPU service's model 2,
or ONNX Runtime).
"""
from collections import namedtuple

import cv2
import numpy as np

from handtrack.calib import triangulate

FULL = 1024               # the detector image the crops are cut from (source.det_image)
DET = 160                 # the detector's input size
# (x, y, w) squares of the FULL upright image, best first (det_tile_e3_report)
LAYOUTS = {
    1: ((256, 128, 640),),
    2: ((256, 128, 640), (128, 384, 384)),
    4: ((256, 128, 640), (128, 384, 384), (384, 512, 384), (128, 640, 384)),
}
THR = 0.5                 # detector probability for an answer
GAP_M = 0.05              # two-camera agreement: closest approach of the two rays
DIST_M = (0.12, 0.9)      # ... and the triangulated point's distance from the head origin
COLLAPSE_M = 0.07         # a left and a right answer closer than this are one physical hand
SIZE_GAIN = 0.8           # detector size output x crop width x this = hand size (det px), fitted offline
SEED_K = 1.7              # seed crop full width / hand size
SEED_GAP_M = 0.03         # seed joints: median two-view gap at most this
SEED_HAND_M = (0.06, 0.13)  # ... a wrist -> middle knuckle length in this range (views that disagree
                          # along the epipolar direction still meet, at a wrong depth and scale)
SEED_NEAR = 0.75          # ... and their palm centre within this many hand sizes of each camera's answer
                          # (in the image: the confirmed point's depth is poor -- 10.5 cm baseline, centre
                          # error ~0.2 hand size -> 6.5 cm median -- so it is not compared in 3D)
SIDES = ("left", "right")
PALM = (0, 5, 9, 17)

Answer = namedtuple("Answer", "slot crop side p c size")      # c (2,), size: FULL det px
Match = namedtuple("Match", "side X gap a0 a1")               # X (3,) head frame; a0/a1: slot 0/1 answers


# ---------------------------------------------------------------- detector crops
def crop(up, x, y, w):
    """Square crop of side w at (x, y) of the upright image (zero outside), resized to DET (area)."""
    x, y, w = int(round(x)), int(round(y)), int(round(w))
    n = up.shape[0]
    pad = np.zeros((w, w), np.uint8)
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, n), min(y + w, n)
    if xb > xa and yb > ya:
        pad[ya - y:yb - y, xa - x:xb - x] = up[ya:yb, xa:xb]
    return cv2.resize(pad, (DET, DET), interpolation=cv2.INTER_AREA) if w != DET else pad


def normalise(img):
    """uint8 crop -> the detector's input (std 0.25, mean 0.5), or None for a flat crop."""
    f = img.astype(np.float32) / 255.0
    sd = float(f.std())
    if sd == 0:
        return None
    f = f * (0.25 / sd)
    return (f + (0.5 - f.mean())).astype(np.float32)


def upright_region(src, slot, frame, x, y, w):
    """The (x, y, w) square of src.det_image(frame, slot, FULL), computed for that square alone:
    the same warpAffine with its output origin moved, bit-identical to cropping the full upright
    image (a 640 crop rotates 39% of its pixels). Other sources / sizes crop the full image."""
    f = frame.shape[1] / src.raw_w
    x0, y0, n = (int(round(v * f)) for v in (src.x0, src.y0, src.n))
    a = src.angle(slot)
    if n != FULL or a % 90 == 0 or x < 0 or y < 0 or x + w > FULL or y + w > FULL:
        return crop(src.det_image(frame, slot, FULL), x, y, w)
    sq = frame[y0:y0 + n, x0:x0 + n]
    c = (FULL - 1) / 2.0
    M = src._rot(slot)
    A = np.hstack([M, (np.array([c, c]) - M @ np.array([c, c]))[:, None]])
    A[:, 2] -= (x, y)
    up = cv2.warpAffine(sq, A, (w, w), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(up, (DET, DET), interpolation=cv2.INTER_AREA) if w != DET else up


def feeds(src, slot, frame, layout, index=None):
    """The camera frame (as published) -> [(crop index, feed)] of the layout's crops `index`
    (default all). Flat crops are left out (they hold no hand and the detector's normalisation is
    undefined)."""
    out = []
    for i in (range(len(layout)) if index is None else index):
        v = normalise(upright_region(src, slot, frame, *layout[i]))
        if v is not None:
            out.append((i, {"inputImg": v[None, None]}))
    return out


def decode(slot, index, outs, layout, thr=THR):
    """Detector outputs of the crops `index` -> answers above thr (FULL det px)."""
    ans = []
    for i, o in zip(index, outs):
        x, y, w = layout[i]
        p = np.asarray(o["hand_exists"], float).reshape(-1)
        cx, cy = (np.asarray(o[k], float).reshape(-1) for k in ("cx", "cy"))
        size = np.asarray(o["size"], float).reshape(-1)
        for sd in (0, 1):
            if p[sd] > thr:
                c = np.array([x + (cx[sd] + 1) * 0.5 * w, y + (cy[sd] + 1) * 0.5 * w])
                ans.append(Answer(slot, i, SIDES[sd], float(p[sd]), c, float(SIZE_GAIN * size[sd] * w)))
    return ans


# ---------------------------------------------------------------- geometry
class Geometry:
    """Rays of FULL det pixels in the head frame (rig: calib.Rig or a stand-in with cams and
    head_from_cam)."""

    def __init__(self, src, rig):
        self.src = src
        self.cam = {s: rig.cams[n] for s, n in src.slot_cams.items()}
        self.pose = {s: rig.head_from_cam(n) for s, n in src.slot_cams.items()}   # (R, t): head = R cam + t

    def rays(self, slot, det_px):
        """FULL det pixels (N, 2) -> (origin (3,), unit head-frame rays (N, 3))."""
        R, t = self.pose[slot]
        d = self.cam[slot].unproject(self.src.det_to_raw(np.atleast_2d(det_px), slot, FULL))
        return np.asarray(t, float), d @ np.asarray(R).T


def confirm(answers, geom, gap_m=GAP_M, dist_m=DIST_M):
    """answers: all answers of both cameras of one capture -> {side: Match}, the best (smallest gap)
    two-camera agreement per side. A left and a right match on one physical hand keep the one
    with the higher joint probability."""
    by = {(s, sd): [a for a in answers if a.slot == s and a.side == sd] for s in (0, 1) for sd in SIDES}
    best = {}
    for sd in SIDES:
        for a0 in by[(0, sd)]:
            o0, r0 = geom.rays(0, a0.c)
            for a1 in by[(1, sd)]:
                o1, r1 = geom.rays(1, a1.c)
                X, g, s0, s1 = triangulate(o0, r0[0], o1, r1[0])
                X, g = X[0], float(g[0])
                if s0[0] <= 0 or s1[0] <= 0 or not g < gap_m or not dist_m[0] < np.linalg.norm(X) < dist_m[1]:
                    continue
                if sd not in best or g < best[sd].gap:
                    best[sd] = Match(sd, X, g, a0, a1)
    if len(best) == 2 and np.linalg.norm(best["left"].X - best["right"].X) < COLLAPSE_M:
        keep = max(best.values(), key=lambda m: m.a0.p * m.a1.p)
        best = {keep.side: keep}
    return best


# ---------------------------------------------------------------- seeding
def untracked(matches, geom, tracked, near=1.0):
    """The matches that are not on a tracked hand: a tracked palm centre (head frame, any side)
    within `near` x the answer's hand size of the answer in both cameras' images is that hand."""
    def on(m, X):
        for a in (m.a0, m.a1):
            R, t = geom.pose[a.slot]
            d = (np.asarray(X, float) - t) @ np.asarray(R)
            if d[2] <= 0:
                return False
            px = geom.src.raw_to_det(geom.cam[a.slot].project(d[None]), a.slot, FULL)[0]
            if not np.linalg.norm(px - a.c) <= near * a.size:
                return False
        return True
    return {sd: m for sd, m in matches.items() if not any(on(m, X) for X in tracked.values())}


def seed_crop(geom, slot, c, size, k=SEED_K):
    """(q, r) of the keypoint crop centred on FULL det pixel c, k x size det pixels wide, upright
    in the camera (twist 0), in mercury_kp's crop convention."""
    from handtrack.mercury_kp import _direction

    pts = np.array([c, np.asarray(c, float) + [k * size / 2.0, 0.0]])
    rays = geom.cam[slot].unproject(geom.src.det_to_raw(pts, slot, FULL))
    alpha = float(np.arccos(np.clip(rays[0] @ rays[1], -1.0, 1.0)))
    return _direction(rays[0] * np.array([1, -1, -1]), 0.0), float(np.tan(alpha / 2))


def seed_inputs(mk, geom, match, frames):
    """The two unhinted keypoint inputs of a match, as MercuryKeypoint prepared tuples (slot, crop,
    q, r, flip, feeds) for mk._decode, cropped from the published frames (raw=True); None for a
    flat crop."""
    flip = match.side == "right"
    out = []
    for a in (match.a0, match.a1):
        q, r = seed_crop(geom, a.slot, a.c, a.size)
        img = mk.crop(a.slot, frames[a.slot], q, r, flip, raw=True)
        x = mk.normalise(img)
        if x is None:
            return None
        out.append((a.slot, img, q, r, flip, {"inputImg": x[None, None],
                                              "lastKeypoints": np.zeros((1, 42), np.float32),
                                              "useLastKeypoints": np.zeros(1, np.float32)}))
    return out


def seed_joints(geom, match, dec0, dec1, gap_m=SEED_GAP_M, near=SEED_NEAR, hand_m=SEED_HAND_M):
    """Two same-instant keypoint decodes (mk._decode of the slot 0 and slot 1 seed crops) -> 21
    head-frame joints triangulated joint by joint, or None when the views disagree (median gap
    over gap_m, a joint behind a camera, a hand size outside hand_m) or the hand is not the one the detector answered (its palm
    centre over `near` hand sizes from either camera's answer, in that camera's image)."""
    (R0, t0), (R1, t1) = geom.pose[0], geom.pose[1]
    r0 = np.asarray(dec0["rays"], float) @ np.asarray(R0).T
    r1 = np.asarray(dec1["rays"], float) @ np.asarray(R1).T
    P, g, s0, s1 = triangulate(np.broadcast_to(t0, r0.shape), r0, np.broadcast_to(t1, r1.shape), r1)
    if (s0 <= 0).any() or (s1 <= 0).any() or not np.median(g) <= gap_m:
        return None
    if not hand_m[0] <= np.linalg.norm(P[0] - P[9]) <= hand_m[1]:
        return None
    palm = P[list(PALM)].mean(0)
    for a in (match.a0, match.a1):
        R, t = geom.pose[a.slot]
        raw = geom.cam[a.slot].project(np.atleast_2d((palm - t) @ np.asarray(R)))
        if not np.linalg.norm(geom.src.raw_to_det(raw, a.slot, FULL)[0] - a.c) <= near * a.size:
            return None
    return P
