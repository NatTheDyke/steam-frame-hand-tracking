"""Monado Mercury's keypoint model as a per-hand ROI backend (Python port of the crop /
decode stage of src/xrt/tracking/hand/mercury/hg_model.cpp + hg_image_distorter.cpp,
Monado commit 13b5c134b16c86ff3e3be52e8fa6dc720a8247e0, BSL-1.0).

Model: grayscale_keypoint_jan18.onnx (sha256 40c0daa5...d1d5), from
gitlab.freedesktop.org/monado/utilities/hand-tracking-models @37a8a81b.  Its license is
UNVERIFIED (no license file in that repo): local benchmarking only, not a shipping dependency.

Per hand and camera view: a 128x128 stereographic reprojection of the fisheye image, centred on
the hand's predicted 3D keypoints (right hands mirrored), normalised, run through the model with
the predicted keypoints as a temporal hint; heatmaps are decoded back to 3D ray directions.

Frames: our camera frame is OpenCV (x right, y down, z forward); Mercury's is OpenXR-like
(x right, y up, -z forward) -> flip y and z when crossing.
"""
import math

import cv2
import numpy as np

WSIZE = 128
GRID = 22
EXPAND = 1.7          # Mercury tuneable dyn_radii_fac default
IS_HAND_MIN = 0.97    # Mercury: when the model is sure, it's really sure


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z)
def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_rot(q, v):
    """Rotate vectors v (N,3) by quaternion q."""
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.atleast_2d(v)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_matrix(q):
    """Rotation matrix of unit quaternion q: q_matrix(q) @ v == q_rot(q, v)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def q_from_two_vectors(a, b):
    """Minimal rotation taking direction a to direction b (Eigen setFromTwoVectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c < -1 + 1e-6:                      # opposite: rotate pi about any orthogonal axis
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    ax = np.cross(a, b)
    s = math.sqrt((1 + c) * 2)
    return np.array([s / 2, *(ax / s)])


def direction(dir3, twist):
    one = q_from_two_vectors(np.array([0.0, 0, -1]), dir3)
    h = twist / 2
    two = np.array([math.cos(h), 0, 0, -math.sin(h)])     # AngleAxis(twist, -UnitZ)
    return q_mul(one, two)


def _direction(v, twist):
    """direction(v, twist) in scalar arithmetic: q_from_two_vectors((0, 0, -1), v), then the twist."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    bx, by, bz = v[0] / n, v[1] / n, v[2] / n
    c = -bz
    if c < -1 + 1e-6:                      # opposite of -z: the general path
        return direction(np.asarray(v, float), twist)
    s = math.sqrt((1 + c) * 2)
    w1, x1, y1 = s / 2, by / s, -bx / s    # axis cross((0, 0, -1), b) = (by, -bx, 0)
    h = twist / 2
    w2, z2 = math.cos(h), -math.sin(h)
    return np.array([w1 * w2, x1 * w2 + y1 * z2, -x1 * z2 + y1 * w2, w1 * z2])


def sg_unproject(x, y):
    d = 1 + x * x + y * y
    return np.stack([2 * x / d, 2 * y / d, (-1 + x * x + y * y) / d], -1)


# ---------------------------------------------------------------- projection instructions
def project_unscaled(joints_m, q):
    d = joints_m / np.linalg.norm(joints_m, axis=1, keepdims=True)
    d = q_rot(q_conj(q), d)
    den = 1 - d[:, 2]
    return np.stack([d[:, 0] / den, d[:, 1] / den], 1)


def make_instructions(joints_m, flip, expand=EXPAND, twist=0.0):
    """joints_m: (21,3) in Mercury camera frame.  Returns rot_quat, radius, predicted keypoints
    (21,2) in the model's normalised input convention, relative depths (21,).

    Same iterations and formulas as Mercury; the joint directions are normalised once and each
    projection is one matrix product (project_unscaled and direction give the same values)."""
    dn = joints_m / np.linalg.norm(joints_m, axis=1, keepdims=True)

    def project(q):
        d = dn @ q_matrix(q)               # rows R^T d: rotation by conj(q)
        den = 1 - d[:, 2]
        return np.stack([d[:, 0] / den, d[:, 1] / den], 1)

    dir3 = joints_m[9] / np.linalg.norm(joints_m[9])
    q = _direction(dir3, twist)
    old = dir3
    for _ in range(8):
        sg = project(q)
        mn, mx = sg.min(0), sg.max(0)
        c0, c1 = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        r = max(c0 - mn[0], c1 - mn[1])
        dd = 1 + c0 * c0 + c1 * c1         # sg_unproject(c0, c1)
        new = q_matrix(q) @ np.array([2 * c0 / dd, 2 * c1 / dd, (-1 + c0 * c0 + c1 * c1) / dd])
        q = _direction(new, twist)
        if np.linalg.norm(old - new) < 1e-4:
            break
        old = new
    sg = project(q)
    fwd = np.linalg.norm(sg[0] - sg[9])
    side = np.linalg.norm(sg[5] - sg[17])
    palm_l = max(fwd, side)
    r = max(r * expand, palm_l * 0.5 * (2.2 / 1.65) * expand)
    kp = np.stack([sg[:, 0] / r * (-1 if flip else 1), -sg[:, 1] / r], 1)   # exactly as Mercury feeds it
    norms = np.linalg.norm(joints_m, axis=1)
    hand_size = np.linalg.norm(joints_m[0] - joints_m[9])
    rel = (norms - norms[9]) / max(hand_size, 1e-6)
    return q, r, kp.astype(np.float32), rel


# ---------------------------------------------------------------- model wrapper
class MercuryKeypoint:
    def __init__(self, model_path, rig, slot_cams, det_size, raw_size=2464, threads=1,
                 backend=None, inference="cpu", framenpu_jobs=None, source=None, npu_socket=None):
        from handtrack.inference import make_backend
        self.backend = backend if backend is not None else make_backend(
            model_path, threads, inference, framenpu_jobs, npu_socket)
        # source (handtrack/source.py) overrides slot_cams / raw_size and supplies the raw <-> det
        # mapping; without it the original passthrough geometry is used.
        self.src = source
        if source is not None:
            slot_cams = source.slot_cams
        self.cam = {s: rig.cams[n] for s, n in slot_cams.items()}
        self.det_size, self.raw_size = det_size, raw_size
        g = (np.arange(WSIZE, dtype=np.float32))
        self.gx, self.gy = np.meshgrid(g, g)
        self.axis = 1 - 2 * np.arange(WSIZE) / WSIZE   # sx = -r * axis[col] (unflipped), sy = r * axis[row]

    # det image (rotated, det_size square) <-> raw calibrated pixels
    def raw_to_det(self, raw, slot):
        if self.src is not None:
            return self.src.raw_to_det(raw, slot, self.det_size)
        k = self.raw_size / self.det_size
        x = (raw[..., 0] + 0.5) / k - 0.5
        y = (raw[..., 1] + 0.5) / k - 0.5
        S = self.det_size
        if slot == 0:            # rotated CW
            return np.stack([S - 1 - y, x], -1)
        return np.stack([y, S - 1 - x], -1)   # rotated CCW

    def crop(self, slot, det_img, q, r, flip, raw=False):
        """128x128 crop.  det_img is the det_size detector image, or with raw=True the camera frame
        as published in the shm (unrotated, raw aspect, any scale): sampling the full-resolution
        frame keeps detail for small (distant) hands that the detector image has already lost."""
        px = self.crop_pixels(slot, q, r, flip)
        if raw:
            raw_w = self.src.raw_w if self.src is not None else self.raw_size
            f = det_img.shape[1] / raw_w                       # published pixels per raw pixel
            m = ((px + 0.5) * f - 0.5).astype(np.float32)
        else:
            m = self.raw_to_det(px, slot).astype(np.float32)
        return cv2.remap(det_img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def crop_pixels(self, slot, q, r, flip):
        """Calibrated raw pixel (W, W, 2) sampled by each crop pixel.

        The stereographic ray of pixel (row, col) is (2 sx, 2 sy, sx^2 + sy^2 - 1) with sx from the
        column and sy from the row, so each rotated component is a row term plus a column term.
        Then Mercury -> OpenCV (y, z negated) and the camera's Kannala-Brandt projection."""
        cam = self.cam[slot]
        a = self.axis * (r if flip else -r)
        b = self.axis * r
        a2, b2 = a * a, b * b
        R = q_matrix(q)
        x = (2 * R[0, 1] * b + R[0, 2] * (b2 - 1))[:, None] + (2 * R[0, 0] * a + R[0, 2] * a2)[None, :]
        y = (-(2 * R[1, 1] * b + R[1, 2] * (b2 - 1)))[:, None] + (-(2 * R[1, 0] * a + R[1, 2] * a2))[None, :]
        z = (-(2 * R[2, 1] * b + R[2, 2] * (b2 - 1)))[:, None] + (-(2 * R[2, 0] * a + R[2, 2] * a2))[None, :]
        rr = np.hypot(x, y)
        d = cam._d(np.arctan2(rr, z))
        s = np.where(rr > 1e-12, d / np.maximum(rr, 1e-12), 0)
        return np.stack([cam.fx * x * s + cam.cx, cam.fy * y * s + cam.cy], -1)

    @staticmethod
    def normalise(img):
        f = img.astype(np.float32) / 255.0
        sd = f.std()
        if sd == 0:
            return None
        f *= 0.25 / sd
        f += 0.5 - f.mean()
        return f

    def run(self, slot, det_img, joints_cv, flip, use_hint=True, expand=None):
        """One available observation: submit immediately, never wait for more crops."""
        return self.run_many([dict(slot=slot, det_img=det_img, joints_cv=joints_cv,
                                   flip=flip, use_hint=use_hint, expand=expand)])[0]

    def run_many(self, observations):
        """Available observations, in order; synchronous chunks of at most three.

        Zero-contrast crops produce None in their original position and consume
        no inference slot. This method owns no queue and never pads a request.
        """
        results = []
        for start in range(0, len(observations), 3):
            prepared = [self._prepare(**o) for o in observations[start:start + 3]]
            valid = [p for p in prepared if p is not None]
            outputs = self.backend.infer([p[-1] for p in valid]) if valid else []
            if len(outputs) != len(valid):
                raise ValueError("inference returned the wrong crop count")
            it = iter(outputs)
            results.extend(None if p is None else self._decode(p, next(it)) for p in prepared)
        return results

    def idle(self):
        self.backend.idle()

    def close(self):
        self.backend.close()

    def _prepare(self, slot, det_img, joints_cv, flip, use_hint=True, expand=None, raw=False):
        jm = joints_cv * np.array([1, -1, -1])
        q, r, kp, _ = make_instructions(jm, flip, expand=EXPAND if expand is None else expand)
        img = self.crop(slot, det_img, q, r, flip, raw=raw)
        x = self.normalise(img)
        if x is None:
            return None
        feeds = {"inputImg": x[None, None], "lastKeypoints": kp.reshape(1, 42) if use_hint else np.zeros((1, 42), np.float32),
                 "useLastKeypoints": np.array([1.0 if use_hint else 0.0], np.float32)}
        return slot, img, q, r, flip, feeds

    def _decode(self, prepared, outputs):
        # Geometry and decoding are shared by CPU and FrameNPU.
        slot, img, q, r, flip, _ = prepared
        hm, hd, extras, curls = (outputs[n] for n in (
            "heatmap_xy", "heatmap_depth", "scalar_extras", "curls"))
        hm = hm[0].reshape(21, GRID * GRID)
        conf = hm.max(1)
        locs = self._refine_all(hm)
        px = (1 - locs[:, 0] / 11.0) if flip else (locs[:, 0] / 11.0 - 1)
        py = 1 - locs[:, 1] / 11.0
        dl = sg_unproject(px * r, py * r)
        dm = q_rot(q, dl)
        rays = dm * np.array([1, -1, -1])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        raw = self.cam[slot].project(rays)
        dep = np.clip(hd[0], 0, 1)
        s = dep.sum(1)
        com = (dep * np.arange(GRID)).sum(1) / np.maximum(s, 1e-5)
        rel = ((com + 0.5) / GRID - 0.5) * 3.0
        return {"rays": rays, "det_px": self.raw_to_det(raw, slot), "conf": conf, "rel_depth": rel,
                "is_hand": float(1 / (1 + math.exp(-float(extras[0][0])))),
                "curls": curls[0][:5].copy(), "curl_var": np.abs(curls[0][5:]) + 0.01, "crop": img}

    @staticmethod
    def _refine_all(hm, kmax=10):
        """_refine for every joint at once: hm (J, GRID*GRID) -> (J, 2) refined (x, y).

        Same peak (first argmax), the same clamped window and the same zero-sum fallback; the
        window is a row mask times a column mask, and the sums are taken in float64."""
        cy, cx = np.divmod(hm.argmax(1), GRID)
        g = np.arange(GRID)
        kx = np.maximum(0, np.minimum(cx, np.minimum(kmax, np.abs(cx - GRID) - 1)))
        ky = np.maximum(0, np.minimum(cy, np.minimum(kmax, np.abs(cy - GRID) - 1)))
        mx = np.abs(g - cx[:, None]) <= kx[:, None]             # (J, GRID) columns in the window
        my = np.abs(g - cy[:, None]) <= ky[:, None]             # (J, GRID) rows in the window
        h = hm.reshape(-1, GRID, GRID).astype(np.float64)
        rows = np.einsum("jyx,jx->jy", h, mx)                   # each row's sum over window columns
        s = np.einsum("jy,jy->j", rows, my)
        sx = np.einsum("jyx,jy,jx->j", h, my, mx * (g + 0.5))
        sy = np.einsum("jy,jy->j", rows, my * (g + 0.5))
        ok = s != 0
        safe = np.where(ok, s, 1)
        return np.stack([np.where(ok, sx / safe, cx), np.where(ok, sy / safe, cy)], 1).astype(float)

    @staticmethod
    def _refine(h, cx, cy, kmax=10):
        """Mercury's refine_center_of_distribution (centre of mass in a window around the peak)."""
        w = hgt = GRID
        kx = max(0, min(cx, min(kmax, abs(cx - w) - 1)))
        ky = max(0, min(cy, min(kmax, abs(cy - hgt) - 1)))
        win = h[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1]
        s = win.sum()
        if s == 0:
            return np.array([cx, cy], float)
        ys, xs = np.mgrid[cy - ky:cy + ky + 1, cx - kx:cx + kx + 1]
        return np.array([(win * (xs + 0.5)).sum() / s, (win * (ys + 0.5)).sum() / s])
