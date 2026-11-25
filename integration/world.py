"""A world the whole stack can be run in, with known answers and no headset, recording or DSP.

Two synthetic SLAM cameras (a sensorRigCalibration.json of the real layout: fisheye, rolled about
+-125 degrees) see two flat hands, each drawn as 21 discs on black. BlobSession stands in for the
models behind a real framenpu.service.Server: it finds the discs in the crops it is sent and answers
as the models do (ft7's heatmap convention), so crop sampling, the wire, decoding, triangulation,
flow tracking and the bridge must all agree for the tracker to report where the hands are. The
left-hand / right-hand discs differ by one large disc beside the right wrist.
"""
import json
import math
import os
import struct
import sys
import time

import cv2
import numpy as np

W, H = 1056, 1024
K = dict(fx=430.0, fy=430.0, cx=527.5, cy=511.5, k1=0.02, k2=-0.01, k3=0.0, k4=0.0)
# camera -> (head-frame position m, roll deg); right first (shm slot 0)
CAMS = {"slam_right": ((0.052, -0.031, -0.060), 124.0), "slam_left": ((-0.053, -0.030, -0.061), -125.0),
        "upper_left": ((-0.060, 0.041, -0.050), 10.0), "upper_right": ((0.063, 0.043, -0.052), -10.0)}
SLOTS = ("slam_right", "slam_left")
DISC_M, MARK_M = 0.0050, 0.0105      # disc radii (metres)
LEVELS = np.linspace(95, 255, 21).round().astype(int)    # joint 0 .. 20


def axes(roll):
    """Camera axes in the head frame as columns (x right, y down, z forward), rolled about z."""
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    base = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    return base @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def calibration():
    cams = []
    for name, (pos, roll) in CAMS.items():
        R = axes(roll)
        cams.append({"sourceCamera": name, "cameraModel": "kb", "width": W, "height": H, "intrinsics": dict(K),
                     "extrinsics": {"plus_x": R[:, 0].tolist(), "plus_z": R[:, 2].tolist(),
                                    "position": [1e3 * v for v in pos]}})
    return {"cameras": cams, "cad": {"__head": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]},
                                     "cameras": [{"position": [1e3 * v for v in p]} for p, _ in CAMS.values()]}}


def write_calibration(path):
    with open(path, "w") as f:
        json.dump(calibration(), f)


def project(slot, P):
    """Head-frame points -> raw pixels of the slot's camera (Kannala-Brandt)."""
    pos, roll = CAMS[SLOTS[slot]]
    d = (np.atleast_2d(P) - np.array(pos)) @ axes(roll)
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(r, d[:, 2])
    t2 = th * th
    k = th * (1 + t2 * (K["k1"] + t2 * (K["k2"] + t2 * (K["k3"] + t2 * K["k4"]))))
    s = np.where(r > 1e-12, k / np.maximum(r, 1e-12), 0)
    return np.stack([K["fx"] * d[:, 0] * s + K["cx"], K["fy"] * d[:, 1] * s + K["cy"]], -1)


def hand(side, centre):
    """21 joints (MediaPipe order) of an open hand facing the head, fingers up, palm centre (joints
    0, 5, 9, 17) at `centre`."""
    P = np.zeros((21, 3))
    for f, x in enumerate((0.026, 0.008, -0.010, -0.028)):
        for k in range(4):
            P[5 + 4 * f + k] = (x * (1 + 0.28 * k), 0.085 + 0.026 * k, 0.0)
    for k in range(4):
        P[1 + k] = (0.028 + 0.019 * k, 0.018 + 0.021 * k, 0.0)
    if side == "left":           # palm towards the face: the thumb points outwards
        P[:, 0] *= -1
    return P - P[[0, 5, 9, 17]].mean(0) + centre


def truth(t, moving=True):
    """side -> joints at time t (s). FI_HAND=left|right leaves only that hand."""
    a = 0.025 if moving else 0.0
    w = 2 * math.pi * 0.8          # 126 mm/s: a frame of wrong time is 4 mm
    hands = {"left": hand("left", np.array([-0.10 + a * math.sin(w * t), -0.02 + a * math.cos(w * t), -0.46])),
            "right": hand("right", np.array([0.16 + a * math.sin(w * t), -0.03, -0.46 + a * math.cos(w * t)]))}
    only = os.environ.get("FI_HAND")
    return {only: hands[only]} if only else hands


def render(hands):
    """The two cameras' images of the hands: [slot 0 (right), slot 1 (left)]."""
    out = []
    for slot in (0, 1):
        img = np.zeros((H, W), np.uint8)
        for side, P in hands.items():
            px = project(slot, P)
            edge = project(slot, P + np.array([DISC_M, 0, 0]))
            rad = np.linalg.norm(edge - px, axis=1)
            if side == "right":
                m = P[0] + np.array([-0.045, -0.02, 0.0])
                c = project(slot, m)[0]
                r = np.linalg.norm(project(slot, m + np.array([MARK_M, 0, 0]))[0] - c)
                _disc(img, c, r, 80)
            for j in range(21):
                _disc(img, px[j], rad[j], int(LEVELS[j]))
        out.append(img)
    return out


def _disc(img, c, r, level):
    cv2.circle(img, (int(round(c[0] * 256)), int(round(c[1] * 256))), int(round(r * 256)), level, -1, cv2.LINE_AA, 8)


# ---------------------------------------------------------------- the camera shared memory
HEADER = struct.Struct("<4I" + "QQIIQ" * 2 + "2Q")
MAGIC = 0x4D525054


class Camera:
    """Writes pairs as slamgrab does (a seqlock per camera)."""

    def __init__(self, path):
        import mmap
        n = HEADER.size + 2 * W * H
        with open(path, "wb+") as f:
            f.truncate(n)
            self.m = mmap.mmap(f.fileno(), n)
        struct.pack_into("<4I", self.m, 0, MAGIC, W, H, 2)
        self.seq = 0

    def put(self, images, t_ns):
        for c, img in enumerate(images):
            off = 16 + c * 32
            struct.pack_into("<Q", self.m, off, self.seq + 1)
            a = HEADER.size + c * W * H
            self.m[a:a + W * H] = img.tobytes()
            struct.pack_into("<QQ", self.m, off, self.seq + 1, t_ns)
            struct.pack_into("<Q", self.m, off, self.seq + 2)
        self.seq += 2


# ---------------------------------------------------------------- the stand-in models
def _discs(x, min_px):
    """Connected bright spots of a normalised crop, as (cx, cy, area, peak), background removed."""
    bg = np.median(x)
    top = float(x.max()) - bg
    if top <= 0:
        return []
    mask = (x - bg > 0.15 * top).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        w = np.minimum(x[ys, xs] - bg, 0.9 * (x[ys, xs].max() - bg))     # flat-topped: the disc's centre
        out.append(((w * xs).sum() / w.sum(), (w * ys).sum() / w.sum(), st[i, cv2.CC_STAT_AREA], x[ys, xs].max() - bg))
    return out


GRID = 22


def keypoints(crop, hint=None):
    """The keypoint model's outputs for one 128 x 128 crop, in ft7's convention (a heatmap peak at
    grid index = 22 x pixel / 128), and how far the hint is off (None without one). Like the model it
    only knows one handedness: a right hand's crop must arrive mirrored, or it is no hand."""
    found = _discs(crop, 3)
    if len(found) == 22:                               # the right hand's mark: the largest disc
        found.remove(max(found, key=lambda d: d[2]))
    found.sort(key=lambda d: d[3])                     # joint order: brightness
    off = None
    if ok := len(found) == 21:
        p = np.array([d[:2] for d in found])
        a, b = p[5] - p[0], p[17] - p[0]
        ok = bool(a[0] * b[1] - a[1] * b[0] > 0)
        if hint is not None:                           # -1 .. 1 over the crop
            off = float(np.abs(np.reshape(hint, (21, 2)) - (p / 64 - 1)).mean())
    hm = np.zeros((1, 21, GRID, GRID), np.float32)
    hd = np.zeros((1, 21, GRID), np.float32)
    hd[:, :, 10:12] = 0.5                              # every joint at the middle knuckle's depth
    if ok:
        g = np.arange(GRID)
        for j, (u, v, _, _) in enumerate(found):
            x, y = GRID * u / 128, GRID * v / 128
            hm[0, j] = np.exp(-((g[None, :] - x) ** 2 + (g[:, None] - y) ** 2) / (2 * 0.7 ** 2))
    return {"heatmap_xy": hm, "heatmap_depth": hd,
            "scalar_extras": np.full((1, 8), 8.0 if ok else -8.0, np.float32), "curls": np.zeros((1, 10), np.float32)}, off


def detect(crop):
    """The detector's outputs for one 160 x 160 crop."""
    found = _discs(crop, 2)
    out = {"hand_exists": np.zeros((1, 2), np.float32), "cx": np.zeros((1, 2), np.float32),
           "cy": np.zeros((1, 2), np.float32), "size": np.zeros((1, 2), np.float32)}
    groups = []                                        # discs of one hand are near each other
    for d in found:
        near = [g for g in groups if any(math.hypot(d[0] - e[0], d[1] - e[1]) < 18 for e in g)]
        groups = [g for g in groups if g not in near] + [sum(near, []) + [d]]
    for g in groups:
        pts = np.array([(d[0], d[1]) for d in g])
        if len(g) < 18 or pts.min() < 4 or pts.max() > 155:
            continue                                   # cut off by the crop
        area = np.array([d[2] for d in g])
        s = int(area.max() > 2 * np.median(area))      # 1 right: it has the mark
        if s:
            pts = np.delete(pts, int(np.argmax(area)), 0)
        lo, hi = pts.min(0), pts.max(0)
        c, size = (lo + hi) / 2, float((hi - lo).max())
        out["hand_exists"][0, s] = 0.95
        out["cx"][0, s], out["cy"][0, s] = c[0] / 80 - 1, c[1] / 80 - 1
        out["size"][0, s] = size / 160 / 0.8
    return out


class BlobSession:
    """What framenpu.service.Server opens instead of a DSP session. Faults, for the tests:
    FI_FAIL_AFTER=n raises RecoveryRequired on every request after the n-th of a session;
    FI_CLOSE_S=s makes closing take s seconds (the device's takes about 0.1)."""
    max_crops = 4

    def __init__(self, log=None):
        self.last_timing, self.log = None, log if log is not None else []
        self.left = int(os.environ["FI_FAIL_AFTER"]) if os.environ.get("FI_FAIL_AFTER") else None
        self.log.append(("open", []))

    def _count(self):
        if self.left is not None:
            self.left -= 1
            if self.left < 0:
                from framenpu.errors import RecoveryRequired
                raise RecoveryRequired("injected by the integration test")

    def _run(self, feeds, model):
        most = {0: 4, 1: 2}[model]
        if not 1 <= len(feeds) <= most:
            raise ValueError("%d crops in one request of model %d" % (len(feeds), model))
        if model == 1:
            return [detect(np.asarray(x["inputImg"])[0, 0]) for x in feeds]
        out = []
        for x in feeds:
            hinted = float(np.asarray(x["useLastKeypoints"]).reshape(-1)[0]) == 1.0
            o, off = keypoints(np.asarray(x["inputImg"])[0, 0], x["lastKeypoints"] if hinted else None)
            if off is not None:
                self.log.append(("hint", off))
            out.append(o)
        return out

    def infer(self, feeds, model=0):
        self.log.append(("infer", [(model, len(feeds))]))
        self._count()
        self.last_timing = {"crops": len(feeds), "body_ms": 0.1}
        return self._run(feeds, model)

    def _many(self, kind, reqs):
        self.log.append((kind, [(m, len(f)) for m, f in reqs]))
        self._count()
        out = [self._run(f, m) for m, f in reqs]
        self.last_timing = {"requests": [{"crops": len(f), "body_ms": 0.1} for _, f in reqs], "body_ms": 0.1}
        return out

    def infer_frame(self, reqs):
        if len({m for m, _ in reqs}) != len(reqs):
            raise ValueError("a frame's models are distinct")
        return self._many("frame", reqs)

    def infer_many(self, reqs):
        return self._many("many", reqs)

    def idle(self):
        self.log.append(("idle", []))

    def close(self):
        time.sleep(float(os.environ.get("FI_CLOSE_S", 0)))
        self.log.append(("close", []))


def serve(socket_path, keypoint, detector, log_path=None):
    """Run the real service on BlobSessions (python world.py serve SOCKET KEYPOINT DETECTOR [LOG])."""
    import logging
    import signal
    from framenpu.service import Server
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    class Log(list):
        def append(self, x):
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(x) + "\n")
    srv = Server(keypoint, None, socket_path, os.getuid(), session_factory=lambda: BlobSession(Log()),
                 det_model=detector, crops=4)
    signal.signal(signal.SIGTERM, lambda *_: setattr(srv, "stop", True))
    srv.run()


def film(path, seconds, fps=30.0, moving=True):
    """Publish the world on camera shared memory for `seconds` (python world.py film PATH SECONDS)."""
    cam, t0, n = Camera(path), time.monotonic(), 0
    while time.monotonic() - t0 < seconds:
        t = time.monotonic_ns()
        cam.put(render(truth(t / 1e9, moving)), t)
        n += 1
        time.sleep(max(0.0, t0 + n / fps - time.monotonic()))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(*sys.argv[2:])
    else:
        film(sys.argv[2], float(sys.argv[3]), *[float(v) for v in sys.argv[4:5]], moving=sys.argv[5:] != ["0"])
