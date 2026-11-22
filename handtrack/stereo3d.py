"""Same-instant stereo hand estimate (MercuryTracker views = "stereo3d"), FeatureNN-style.

Both lower SLAM cameras expose together (identical capture times), so each tracked hand gets BOTH
cameras' crops from the same pair, placed from the hand's predicted joints, and:
  - the keypoint model (Mercury / a student) decodes each crop: two same-instant ray sets, which
    triangulate without the stale-view error of "alternate";
  - a 3D stereo student (distill/make_stereo3d.py graph: inputImg [1,2,128,128] = view 0 (slam_left)
    and view 1 (slam_right) crops as channels, lastKeypoints [1,84], useLastKeypoints [1], geom [1,16];
    output joints3d [1,63]) gives the whole hand in one inference.
The tracker combines them (hybrid): the 3D hand's shape moved onto the triangulated palm centre.
Measured offline (distill/jitlag.py, Frame test recording): still-hand palm jitter equal to ft7's,
fingertip depth jitter and palm-angle noise about half, prompt-shape AUC 0.99 vs 0.92.

Geometry (as distill/pairs3d.py): D = diag(1,-1,-1) OpenCV -> Mercury camera; q_k the crop
rotation; F flips x for a right hand (mirrored crop); A_k = F R(q_k)^T D maps camera-k vectors to
view k's crop frame; T_10 maps camera 0 to camera 1.  Target y = A_0 (joint - c0) / 0.1 with c0 the
placement's palm centre in camera 0.
"""
import numpy as np

PALM = [0, 5, 9, 17]
HYB_PALM = [0, 5, 9, 13, 17]
VIEW_CAMS = ("slam_left", "slam_right")        # the 3D student's view 0, view 1
D = np.diag([1.0, -1.0, -1.0])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def crop_frame(q, right):
    F = np.eye(3)
    if right:
        F[0, 0] = -1
    return F @ qmat(np.asarray(q, float)).T @ D


def pair_geometry(q0, q1, r0, r1, right, place0, R10, t10):
    """geom (16,) float32, A0 (3,3), c0 (3,) for one pair; place0 = placement joints in camera 0 (OpenCV)"""
    A0, A1 = crop_frame(q0, right), crop_frame(q1, right)
    c0 = place0[PALM].mean(0)
    c1 = R10 @ c0 + t10
    M = A1 @ R10 @ A0.T
    o1 = -R10.T @ t10
    geom = np.concatenate([M.reshape(9), A0 @ o1 / 0.1, [np.linalg.norm(c0) / 0.5, np.linalg.norm(c1) / 0.5, r0 * 5, r1 * 5]])
    return geom.astype(np.float32), A0, c0


def from_y(y, A0, c0):
    """network output (21,3) -> joints in camera 0 (OpenCV)"""
    return (y * 0.1) @ A0 + c0


def hybrid(P_tri, P_3d):
    """the 3D hand moved so its palm centre is the triangulated one (both (21,3), same frame)"""
    return P_3d - P_3d[HYB_PALM].mean(0) + P_tri[HYB_PALM].mean(0)


class Stereo3D:
    """CPU ONNX Runtime session for the 3D stereo student (one pair per run)."""

    def __init__(self, model_path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        if "geom" not in self.names:
            raise ValueError("%s has no geom input: not a 3D stereo student" % model_path)

    def run(self, prep0, prep1, place0, right, R10, t10):
        """prep_k = MercuryKeypoint._prepare() of view k (slot, img, q, r, flip, feeds); place0 the
        placement joints in camera 0 (OpenCV). Returns (joints (21,3) camera 0 OpenCV, is_hand)."""
        f0, f1 = prep0[5], prep1[5]
        x = np.concatenate([f0["inputImg"], f1["inputImg"]], 1).astype(np.float32)
        use = f0["useLastKeypoints"]
        hint = np.concatenate([f0["lastKeypoints"], f1["lastKeypoints"]], 1).astype(np.float32) * use[0]
        geom, A0, c0 = pair_geometry(prep0[2], prep1[2], float(prep0[3]), float(prep1[3]), right, place0, R10, t10)
        out = self.sess.run(None, {"inputImg": x, "lastKeypoints": hint, "useLastKeypoints": use.astype(np.float32),
                                   "geom": geom[None]})
        named = dict(zip([o.name for o in self.sess.get_outputs()], out))
        y = named["joints3d"].reshape(21, 3)
        ex = named.get("scalar_extras")
        is_hand = float(1 / (1 + np.exp(-float(ex[0][0])))) if ex is not None else 1.0
        return from_y(y, A0, c0), is_hand
