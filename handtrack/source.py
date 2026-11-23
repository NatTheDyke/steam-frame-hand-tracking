"""Camera sources: where frames come from and how a raw camera frame becomes the square, upright
detector image the tracker works in.

passthrough  ptgrab -> /dev/shm/mrhand-pt.  Colour passthrough cameras (luma), 2464x2464 calibrated
             pixels (ptgrab publishes them downscaled, normally to 640), 60 Hz.  Paused by XRService
             whenever room view is off, i.e. in games.
slam         slamgrab -> /dev/shm/mrhand-slam.  Lower tracking (SLAM) cameras, 1056x1024 greyscale
             fisheye, 30 Hz scene frames (the LED exposures in between are dropped by slamgrab).
             Running whenever head tracking runs, including in games.

shm slot 0 is always the right camera, slot 1 the left.

Detector image (det_size S square):  raw calibrated pixels -> crop to the square (x0, y0, n) ->
scale to S -> rotate about the image centre by the slot's angle (degrees, clockwise on screen) so
that head "down" points down in the image.

Source.resample picks how a frame that needs scaling is scaled (a frame at S already is only rotated,
bit-identically in every mode):  "area" (the default) area-averages then rotates; "linear" scales
bilinearly then rotates (~2.5x faster at 1024 -> 640, mean 0.9 grey levels off "area"); "remap"
scales and rotates in one bilinear pass from a precomputed map (~4x faster, 2.9 levels off: aliased).  Passthrough uses exact quarter turns (90 = CW, -90 =
CCW: the original behaviour); the SLAM cameras are also rolled ~35 deg, so their angle comes from
the factory calibration (upright_angle).
"""
import math
import os

import cv2
import numpy as np


def upright_angle(rig, cam_name):
    """Clockwise rotation (deg) that makes head-down point down in this camera's image."""
    R, _t = rig.head_from_cam(cam_name)
    d = R.T @ np.array([0.0, -1.0, 0.0])         # head down, camera coordinates (x right, y down)
    return math.degrees(math.atan2(d[0], d[1]))


class Source:
    def __init__(self, name, shm, slot_cams, raw_wh, crop, period_ns, angles=None, upright_from_calib=False):
        self.name, self.shm = name, shm
        self.slot_cams = dict(slot_cams)
        self.raw_w, self.raw_h = raw_wh
        self.x0, self.y0, self.n = crop
        self.period_ns = period_ns
        self._angles = dict(angles) if angles else None
        self._from_calib = upright_from_calib
        self.resample = "area"
        self._maps = {}

    # ---------------------------------------------------------------- rotation
    def angle(self, slot):
        if self._angles is None:
            from handtrack.calib import Rig
            rig = Rig()
            self._angles = {s: round(upright_angle(rig, n), 1) for s, n in self.slot_cams.items()}
        return self._angles[slot]

    def _rot(self, slot):
        a = self.angle(slot)
        if a % 90 == 0:                              # exact quarter turns: keep the integer formulas
            c, s = round(math.cos(math.radians(a))), round(math.sin(math.radians(a)))
        else:
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        return np.array([[c, -s], [s, c]], float)

    # ---------------------------------------------------------------- pixel mapping
    def raw_to_det(self, raw, slot, S):
        """Calibrated raw pixels (..., 2) -> detector-image pixels (..., 2)."""
        raw = np.asarray(raw, float)
        k = self.n / S
        sx = (raw[..., 0] - self.x0 + 0.5) / k - 0.5
        sy = (raw[..., 1] - self.y0 + 0.5) / k - 0.5
        c = (S - 1) / 2.0
        M = self._rot(slot)
        u, v = sx - c, sy - c
        return np.stack([M[0, 0] * u + M[0, 1] * v + c, M[1, 0] * u + M[1, 1] * v + c], -1)

    def det_to_raw(self, det, slot, S):
        """Detector-image pixels (..., 2) -> calibrated raw pixels (..., 2)."""
        det = np.asarray(det, float)
        c = (S - 1) / 2.0
        M = self._rot(slot)
        u, v = det[..., 0] - c, det[..., 1] - c
        sx = M[0, 0] * u + M[1, 0] * v + c            # M^T (rotation inverse)
        sy = M[0, 1] * u + M[1, 1] * v + c
        k = self.n / S
        return np.stack([(sx + 0.5) * k - 0.5 + self.x0, (sy + 0.5) * k - 0.5 + self.y0], -1)

    # ---------------------------------------------------------------- images
    def det_image(self, img, slot, S):
        """shm frame (any published size, same aspect as the raw camera) -> S x S upright detector image."""
        f = img.shape[1] / self.raw_w                 # published pixels per raw pixel
        x0, y0, n = (int(round(v * f)) for v in (self.x0, self.y0, self.n))
        sq = img[y0:y0 + n, x0:x0 + n]
        scale = sq.shape[0] != S or sq.shape[1] != S
        if scale and self.resample == "remap":
            return cv2.remap(img, *self._map(img.shape, slot, S, x0, y0, n), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if scale:
            if self.resample not in ("area", "linear"):
                raise ValueError("resample %r: expected area, linear or remap" % (self.resample,))
            sq = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA if self.resample == "area" else cv2.INTER_LINEAR)
        a = self.angle(slot)
        if a % 360 == 0:
            return np.ascontiguousarray(sq)
        if a % 90 == 0:
            code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a % 360]
            return cv2.rotate(sq, code)
        c = (S - 1) / 2.0
        M = self._rot(slot)
        A = np.hstack([M, (np.array([c, c]) - M @ np.array([c, c]))[:, None]])
        return cv2.warpAffine(sq, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _map(self, shape, slot, S, x0, y0, n):
        """remap tables taking each S x S detector pixel to its published-frame pixel, outside the
        crop square sampling the border (0) as the two-pass path does."""
        key = (shape, slot, S)
        if key not in self._maps:
            c = (S - 1) / 2.0
            M = self._rot(slot)
            yy, xx = np.mgrid[0:S, 0:S].astype(float)
            u, v = xx - c, yy - c                      # rotation inverse = M^T
            sx = M[0, 0] * u + M[1, 0] * v + c
            sy = M[0, 1] * u + M[1, 1] * v + c
            k = n / S
            px, py = (sx + 0.5) * k - 0.5, (sy + 0.5) * k - 0.5
            out = (px < -0.5) | (px > n - 0.5) | (py < -0.5) | (py > n - 0.5)
            mx = np.where(out, -10.0, px + x0).astype(np.float32)
            my = np.where(out, -10.0, py + y0).astype(np.float32)
            self._maps[key] = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        return self._maps[key]


SOURCES = {
    "passthrough": Source("passthrough", "/dev/shm/mrhand-pt",
                          {0: "passthrough_right", 1: "passthrough_left"}, (2464, 2464), (0, 0, 2464),
                          1e9 / 60.0, angles={0: 90, 1: -90}),
    # 1056x1024 sensor; centred 1024 square keeps the whole fisheye circle (cx ~522-526)
    "slam": Source("slam", "/dev/shm/mrhand-slam",
                   {0: "slam_right", 1: "slam_left"}, (1056, 1024), (16, 0, 1024),
                   1e9 / 30.0, upright_from_calib=True),
}

ENV = "HANDTRACK_SOURCE"


def active():
    """The source selected for this process tree (live.py --source sets HANDTRACK_SOURCE before
    spawning its workers, so MediaPipe / Mercury processes agree with the coordinator)."""
    name = os.environ.get(ENV, "passthrough")
    if name not in SOURCES:
        raise ValueError("%s=%r: expected one of %s" % (ENV, name, ", ".join(SOURCES)))
    return SOURCES[name]
