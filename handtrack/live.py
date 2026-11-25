#!/usr/bin/env python3
"""Live optical hand detection on the Steam Frame cameras.

--source passthrough (default): `ptgrab stream` -> /dev/shm/mrhand-pt (passthrough luma, 2464x2464
per camera, sensors mounted rotated ±90°; paused by XRService when room view is off, i.e. in games).
--source slam: `slamgrab` -> /dev/shm/mrhand-slam (lower tracking cameras, 1056x1024 fisheye, 30 Hz;
running in games too).  handtrack/source.py turns either into square upright detector images.
Each camera gets its own MediaPipe HandLandmarker in VIDEO mode.

Debug view: MJPEG on 127.0.0.1:8765 (camera imagery never leaves the headset
unless you tunnel it):  ssh -L 8765:127.0.0.1:8765 frame   then open
http://localhost:8765 on the PC.

usage: python -m handtrack.live [--size 640] [--no-http] [--log hands.jsonl]
"""
import argparse
import json
import math
import mmap
import os
import struct
import sys
import threading
import types
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


# Small per-frame work: thread pools only add spinning cores (heat), so one thread each.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from handtrack import source as camsource
from handtrack.calib import triangulate
from handtrack.fasttrack import FastTracker, plausible
from handtrack.mercsched import Admission, FrameGrid, capture_cap_due, rate_cap_due
from handtrack.palm import PALM_IDX, Bridge, OneEuro, PalmTracker

SHM = "/dev/shm/mrhand-pt"
# which hands to track, written by the "Hand Tracking" dashboard panel: "left", "right" or "left right"
# (missing file = both).  Re-read about once a second; a disabled hand gets no Mercury crops, no
# optical flow and is sent as invalid, so one-hand mode saves roughly half the per-hand work.
HANDS_FILE = os.path.expanduser("~/.config/mrhand/hands")


PROFILE_FILE = os.path.expanduser("~/.config/mrhand/profile")
# Latency/smoothing profile, editable live (re-read about once a second).  Missing file or missing key =
# the current smooth defaults below, so deleting the file is an instant rollback.
#   axis_filter   = <min_cutoff> <beta> | off   palm orientation axes One Euro (palm.py)
#   centre_filter = <min_cutoff> <beta> | off   palm centre One Euro inside the pose (palm.py)
#   curl_filter   = <min_cutoff> <beta> | off   finger curls One Euro
#   pos_filter    = <min_cutoff> <beta> | off   sent position One Euro (fast path)
#   rot_glide_ms  = <ms>  (0 = off)             exponential glide toward the newest orientation
#   merc_crop     = raw | det | auto            Mercury crop source (default: the --merc-crop option)
#   merc_views    = both | alternate | auto     Mercury views per hand and frame (default: --merc-views)
#   stereo_comp   = on | off                    move an older view by the palm motion before triangulating
#                                               (default: --merc-stereo-comp)
#   send_offset_ms = <ms> (default 0)           shift the capture time sent to the PC driver, which picks the
#                                               head pose for placing the hand in the world by that time
#                                               (diagnostic for hands that move with the head)
# Default = "A" from the 2026-10-11 latency pass: smoother than before when the hand is still, faster when
# it moves (orientation lag ~130 vs ~160 ms, same still-hand jitter).  The previous smooth profile is
# tools/profile_smooth_v1.conf (copy it to ~/.config/mrhand/profile to go back).
FINGERS = ("thumb", "index", "middle", "ring", "pinky")    # curl / splay order
PROFILE_DEFAULTS = {"axis_filter": (0.4, 1.2), "centre_filter": (1.5, 0.6), "curl_filter": (1.0, 1.0),
                    "pos_filter": (1.2, 1.5), "rot_glide_ms": 25.0, "send_offset_ms": 0.0, "merc_crop": None,
                    "merc_views": None, "stereo_comp": None}


class Profile:
    def __init__(self, path=PROFILE_FILE):
        self.path, self.mtime, self.next = path, None, 0.0
        self.values = dict(PROFILE_DEFAULTS)

    def _parse(self, text):
        v = dict(PROFILE_DEFAULTS)
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, val = (s.strip() for s in line.split("=", 1))
            if k not in v:
                continue
            if k in ("rot_glide_ms", "send_offset_ms"):
                v[k] = float(val)
            elif k == "merc_crop":
                v[k] = val if val in ("raw", "det", "auto") else None
            elif k == "merc_views":
                v[k] = val if val in ("both", "alternate", "auto", "stereo3d") else None
            elif k == "stereo_comp":
                v[k] = {"on": True, "off": False}.get(val)
            elif val == "off":
                v[k] = None
            else:
                a, b = val.split()[:2]
                v[k] = (float(a), float(b))
        return v

    def poll(self):
        """Returns the new values dict when the file changed, else None."""
        now = time.monotonic()
        if now < self.next:
            return None
        self.next = now + 1.0
        try:
            mt = os.stat(self.path).st_mtime
        except OSError:
            mt = None
        if mt == self.mtime:
            return None
        self.mtime = mt
        try:
            self.values = self._parse(open(self.path).read()) if mt is not None else dict(PROFILE_DEFAULTS)
        except (OSError, ValueError) as e:
            print("profile: ignoring %s (%r)" % (self.path, e), file=sys.stderr, flush=True)
            return None
        print("profile: %s" % " ".join("%s=%s" % kv for kv in sorted(self.values.items())), file=sys.stderr, flush=True)
        return self.values


class EnabledHands:
    def __init__(self, path=HANDS_FILE):
        self.path, self.mtime, self.sides, self.next = path, None, {"left", "right"}, 0.0

    def get(self):
        now = time.monotonic()
        if now >= self.next:
            self.next = now + 1.0
            try:
                mt = os.stat(self.path).st_mtime
                if mt != self.mtime:
                    self.mtime = mt
                    self.sides = set(open(self.path).read().split()) & {"left", "right"}
                    print("hands enabled: %s" % (" ".join(sorted(self.sides)) or "none"), file=sys.stderr, flush=True)
            except OSError:
                if self.mtime is not None:
                    self.mtime, self.sides = None, {"left", "right"}
        return self.sides
HDR = struct.Struct("<4I" + "QQIIQ" * 2 + "2Q")   # MrhandPtHeader (capture/handcam_proto.h)
MAGIC = 0x4D525054
# sensor mounting: rotate each camera's image upright
ROTATE = {0: cv2.ROTATE_90_CLOCKWISE, 1: cv2.ROTATE_90_COUNTERCLOCKWISE}


class PtShm:
    def __init__(self, path=SHM):
        fd = os.open(path, os.O_RDONLY)
        self.m = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
        os.close(fd)
        magic, self.w, self.h, self.cams = struct.unpack_from("<4I", self.m, 0)
        if magic != MAGIC:
            raise RuntimeError("bad shm magic; is ptgrab stream running?")
        self.frame = self.w * self.h
        self.data0 = HDR.size

    def slot(self, cam):
        off = 16 + cam * 32
        return struct.unpack_from("<QQIIQ", self.m, off)   # seq, tDetectNs, bufferId, pad, frameCount

    def read(self, cam, last_seq=0):
        """Return (seq, t_detect_ns, gray HxW) for the newest frame, or None if unchanged."""
        for _ in range(5):
            seq, t, buf, _pad, n = self.slot(cam)
            if seq == last_seq or seq & 1 or seq == 0:
                if seq & 1:
                    time.sleep(0.0005)
                    continue
                return None
            img = np.frombuffer(self.m, np.uint8, self.frame, self.data0 + cam * self.frame).reshape(self.h, self.w).copy()
            if self.slot(cam)[0] == seq:
                return seq, t, img
        return None


class Telemetry:
    """Bounded per-frame / per-detection telemetry (CSV, rotated at max_bytes, kept on the headset).
    F rows (one per hand per sent frame):
      F,t_cap_ns,t_send_ns,side,pos_src,pos_obs_age_ms,q_src,q_age_ms,curl_age_ms,npts0,npts1,x,y,z,qw,qx,qy,qz
      pos_src: stereo | mono_ray0/1 (flow in one camera, depth held) | hold (frozen) | mediapipe
      *_age_ms: t_send minus the capture time of the image the component was measured on
    U rows (curl-only re-send of the last frame packet when a Mercury result arrives between frames):
      U,t_cap_ns,t_send_ns,side,curl_age_ms
    M rows (one per camera per MediaPipe run):
      M,t_det_ns,t_done_ns,kind,slot,dur_ms,nhands
    Z rows (held Mercury inference power released after admission was blocked, see mercsched.py):
      Z,t_ns,reason   reason: stall (no new camera pair) | untracked (no tracked or enabled hand) |
                      thermal (thermal guard paused Mercury) | unplaceable (no crop in front of a camera)
    """

    def __init__(self, path, max_bytes=20_000_000):
        self.path, self.max_bytes = path, max_bytes
        self.f = open(path, "a", buffering=1 << 16)
        self.n = 0

    def write(self, line):
        self.f.write(line + "\n")
        self.n += 1
        if self.n % 2000 == 0 and self.f.tell() > self.max_bytes:
            self.f.close()
            os.replace(self.path, self.path + ".1")
            self.f = open(self.path, "a", buffering=1 << 16)


class ThermalGuard:
    """Throttles MediaPipe by temperature.  Conservative and NOT validated as safe limits:
    the headset once rebooted under high thermal load (cpu7_middle ~95-97 C, fan at max);
    the cause is unproven (kernel CPU trip points are 110 C), so we stay well below.

    Sensors: max over cpu*-thermal (incl. cpu7-middle, which the fan controller watches),
    cpuss*-thermal and gpuss-*-thermal (ptgrab uses the GPU).
      < SLOW_C          : max_hz
      SLOW_C..SLOWER_C  : max_hz / 2
      SLOWER_C..PAUSE_C : 2 Hz
      >= PAUSE_C        : MediaPipe paused (fast tracker continues) until < RESUME_C
    A separate watchdog (watchdog.py) stops the whole workload at WATCHDOG_C sustained."""
    SLOW_C, SLOWER_C, PAUSE_C, RESUME_C = 88.0, 90.0, 92.0, 88.0   # user-set 2026-10-11 (was 80/84/87/82)

    def __init__(self):
        import glob
        self.zones = [z for z in glob.glob("/sys/class/thermal/thermal_zone*")
                      if open(z + "/type").read().strip().startswith(("cpu", "gpuss"))]
        self.temp = 0.0
        self.paused = False
        self.t_read = 0.0

    def read(self):
        now = time.monotonic()
        if now - self.t_read > 1.0:
            self.t_read = now
            ts = []
            for z in self.zones:
                try:
                    ts.append(int(open(z + "/temp").read()) / 1000.0)
                except (OSError, ValueError):
                    pass
            new = max(ts) if ts else 0.0
            if self.paused and new < self.RESUME_C:
                self.paused = False
                print("thermal: resumed at %.0f C" % new, file=sys.stderr, flush=True)
            elif not self.paused and new >= self.PAUSE_C:
                self.paused = True
                print("thermal: MediaPipe PAUSED at %.0f C" % new, file=sys.stderr, flush=True)
            self.temp = new
        return self.temp

    def interval(self, max_hz):
        """Seconds between MediaPipe runs, or None while paused."""
        t = self.read()
        if self.paused:
            return None
        hz = max_hz
        if t >= self.SLOWER_C:
            hz = min(hz, 2.0)
        elif t >= self.SLOW_C:
            hz = hz / 2
        return 1.0 / hz


class Viewer:
    def __init__(self):
        self.jpeg = None
        self.cv = threading.Condition()
        self.clients = 0   # encode frames only while someone is watching

    def publish(self, bgr):
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cv:
                self.jpeg = enc.tobytes()
                self.cv.notify_all()

    def serve(self, port=8765):
        viewer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                viewer.clients += 1
                try:
                    while True:
                        with viewer.cv:
                            viewer.cv.wait(1.0)
                            j = viewer.jpeg
                        if j:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j))
                            self.wfile.write(j + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    viewer.clients -= 1

        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640, help="detector input size (square)")
    ap.add_argument("--det-resample", choices=("area", "linear", "remap"), default="area",
                    help="how a camera frame is scaled to the detector image (source.Source.resample): "
                         "area (default), linear (~2.5x faster), remap (one pass, ~4x faster, aliased)")
    ap.add_argument("--source", choices=sorted(camsource.SOURCES), default=os.environ.get(camsource.ENV, "passthrough"),
                    help="camera source: passthrough (ptgrab, room view only) or slam (slamgrab, tracking "
                         "cameras, works in games); default passthrough or $HANDTRACK_SOURCE")
    ap.add_argument("--no-http", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="MediaPipe GPU delegate (shares the GPU with the compositor)")
    ap.add_argument("--max-hz", type=float, default=7.0,
                    help="cap on MediaPipe runs per second (position is tracked every frame regardless)")
    ap.add_argument("--mode", choices=("alt", "stereo"), default="alt",
                    help="alt: one camera per cycle (alternating) + periodic stereo; stereo: both cameras every cycle")
    ap.add_argument("--stereo-every", type=int, default=6, help="alt mode: run a stereo cycle every N cycles")
    ap.add_argument("--idle-hz", type=float, default=5.0, help="detection rate while no hand has been seen for 1 s")
    ap.add_argument("--backend", choices=("mediapipe", "mercury"), default="mediapipe",
                    help="mercury: MediaPipe only discovers hands; Mercury keypoints track them continuously")
    ap.add_argument("--merc-inference", choices=("cpu", "framenpu", "service"), default="cpu",
                    help="Mercury inference (default CPU). framenpu: FrameNPU in this process (root); "
                         "service: FrameNPU through the NPU service (headset/setup.sh), the "
                         "tracker runs as the user. FrameNPU failures fall back to CPU")
    ap.add_argument("--merc-model", help="a retrained Mercury keypoint ONNX (same graph) instead of "
                    "grayscale_keypoint_jan18.onnx, for CPU inference; for the NPU service install the "
                    "NPU service (headset/setup.sh)")
    ap.add_argument("--merc-cell-offset", type=float, default=0.5,
                    help="heatmap cell-centre convention of the keypoint model: 0.5 = Monado's Mercury (default); "
                         "0 = the distilled students (ft3/ft7/...), wherever they run (CPU or NPU service)")
    ap.add_argument("--npu-socket", help="NPU service socket (default /run/mrhand-npu/npu.sock)")
    ap.add_argument("--framenpu-jobs", type=os.path.expanduser,
                    help="qualified FrameNPU replay build directory (link.json, image and loader)")
    ap.add_argument("--curl-source", choices=("mercury", "geometric"), default="mercury",
                    help="finger curls from Mercury's curl output (default) or from the 3D joint angles")
    ap.add_argument("--merc-hz", type=float, default=30.0, help="Mercury tracking cycles per second (one crop per hand)")
    ap.add_argument("--max-speed", type=float, default=4.0,
                    help="fast tracker: apparent palm speed (m/s) above which a track is treated as an error")
    ap.add_argument("--lk-levels", type=int, default=3, help="fast tracker: optical-flow pyramid levels")
    ap.add_argument("--merc-predict-roi", action="store_true",
                    help="Mercury crop: extrapolate the palm with its recent velocity when the fast tracker has "
                         "no position, and enlarge the crop while the hand moves fast")
    ap.add_argument("--merc-stereo-ms", type=float, default=40.0,
                    help="max capture-time gap (ms) for triangulating two Mercury views; 0 = only the same capture "
                         "(views from different times describe different hand poses when the hand moves)")
    ap.add_argument("--merc-views", choices=("auto", "both", "alternate", "stereo3d"), default="auto",
                    help="Mercury views per hand and frame: both cameras of the same frame (stereo from one "
                         "instant, at most --merc-crops crops per job) or alternate cameras (stereo against the other "
                         "camera's previous frame); auto = both when inference runs on the NPU")
    ap.add_argument("--merc-crops", type=int, choices=(3, 4), default=3,
                    help="crops per Mercury job: 4 = both hands in both views (needs the 4-crop FrameNPU "
                         "image; the worker splits a job to what its backend takes, so 4 never fails)")
    ap.add_argument("--merc-s3d-model", type=os.path.expanduser,
                    help="3D stereo student ONNX (distill make_stereo3d graph) for --merc-views stereo3d: both cameras' crops of each hand from the same pair, one 3D inference (CPU) plus the keypoint model on both crops")
    ap.add_argument("--merc-s3d-curls", choices=("geometric", "mercury"), default="geometric",
                    help="stereo3d curls: from the joints (the 3D hand shape) or the keypoint model's curl head")
    ap.add_argument("--merc-stereo-comp", choices=("on", "off"), default="on",
                    help="triangulating against an older view: shift it by the palm motion since its capture")
    ap.add_argument("--merc-crop", choices=("auto", "raw", "det"), default="auto",
                    help="Mercury crop source: the full-resolution camera frame (raw) or the detector image "
                         "(det); auto = raw when the published frame is larger than --size (tracking cameras)")
    ap.add_argument("--curl-geo-fingers", default="",
                    help="comma list of fingers (thumb,index,middle,ring,pinky) whose curl comes from each camera "
                         "view's own single-view joints instead of --curl-source; e.g. index")
    ap.add_argument("--splay", action="store_true",
                    help="send finger splay (sideways spread) computed from the 3D joints (default: 0); older PC drivers ignore it")
    ap.add_argument("--curl-latest", action="store_true",
                    help="finger curls from the newest observation only (no blend with older views of other cameras)")
    ap.add_argument("--merc-hint-reset", action="store_true",
                    help="no keypoint hint for Mercury crops placed after a rejected observation or lost flow")
    ap.add_argument("--flow-v2", action="store_true",
                    help="fast tracker: frame-by-frame re-seeding through a short frame history, single-camera "
                         "depth held at most 0.2 s, a lost camera rebuilt from the newest Mercury hand, and "
                         "Mercury's depth anchor taken only from two-camera positions")
    ap.add_argument("--flow-agree", action="store_true",
                    help="fast tracker: a re-seeded or rebuilt camera view is used only if it agrees with the other "
                         "camera and with the current position (prevents short depth jumps)")
    ap.add_argument("--merc-early", action="store_true",
                    help="collect finished Mercury jobs and submit the next one between camera frames")
    ap.add_argument("--merc-preflow", action="store_true",
                    help="collect a finished Mercury job and submit the next one as soon as a camera pair is read, "
                         "before its detector images and optical flow (~8 ms sooner); crops are placed with the "
                         "previous frame's flow positions, and the job gets this frame's images and depth anchor "
                         "when flow is done (raw-frame crops only)")
    ap.add_argument("--merc-fresh", action="store_true",
                    help="submit Mercury right after optical flow (before MediaPipe handling); collect finished jobs "
                         "between camera frames and submit then only if the newest frame became ready less than a "
                         "third of a frame period ago, else wait for the next frame")
    ap.add_argument("--uclamp-max", type=int, default=os.environ.get("HANDTRACK_UCLAMP_MAX") or None,
                    help="cap the CPU frequency this tracker and its workers request (Linux uclamp, 0..1024; "
                         "512 ~ 2 GHz on the big cores); default: no cap (or HANDTRACK_UCLAMP_MAX)")
    ap.add_argument("--frame-ring", action="store_true",
                    help="hand the Mercury worker its camera frames through shared memory (handtrack/framering.py) "
                         "instead of pickled copies: same results, ~1.1 ms sooner and ~1.5 ms less CPU per job")
    ap.add_argument("--merc-cadence", choices=("wall", "capture"), default="wall",
                    help="count the --merc-hz cap in wall time (default) or in camera frames (capture: "
                         "a fixed frame gap, e.g. every 4th frame at 120 Hz, so alternate views stay "
                         "within merctrack.STEREO_MS; mercsched.capture_cap_due)")
    ap.add_argument("--merc-sched", choices=("batch", "perhand"), default="batch",
                    help="batch: all tracked hands in one Mercury job, published together (original); "
                         "perhand: one hand per job on the newest frame, published as soon as it is done")
    ap.add_argument("--discover-hz", type=float, default=2.0,
                    help="with --backend mercury: MediaPipe rate while Mercury is tracking a hand")
    ap.add_argument("--discover", choices=("mediapipe", "mercury"), default="mediapipe",
                    help="hand discovery: MediaPipe (default), or Mercury's hand detector in the Mercury worker "
                         "(needs --backend mercury; both cameras per run, answers kept when the cameras agree, "
                         "seeded by keypoint crops; handtrack/mercdet.py)")
    ap.add_argument("--det-crops", type=int, choices=(1, 2, 4), default=2,
                    help="with --discover mercury: detector crops per camera per run")
    ap.add_argument("--det-model", default=None,
                    help="with --discover mercury: the detector ONNX for the CPU (default: "
                         "~/handtrack/models/mercury/grayscale_detection_160x160.onnx); the NPU service's own "
                         "--det-model is used when it serves one")
    ap.add_argument("--mercury-shadow", action="store_true",
                    help="also run Mercury's keypoint model on each stereo MediaPipe frame and log the comparison")
    ap.add_argument("--fast-orient", action="store_true",
                    help="experimental: 60 Hz orientation from triangulated optical-flow landmarks (measured too noisy: "
                         "p95 14-23 deg frame-to-frame); default = MediaPipe orientation, smoothed")
    ap.add_argument("--rot-smooth-ms", type=float, default=40.0,
                    help="rotation glides toward each new orientation with this time constant (0 = step)")
    ap.add_argument("--blend-ms", type=float, default=120.0,
                    help="ease MediaPipe re-seed corrections in over this time constant (0 = snap)")
    ap.add_argument("--no-bridge", action="store_true", help="disable the UDP bridge (Mercury keeps its local palm pose)")
    ap.add_argument("--telemetry", default=os.path.expanduser("~/handtrack/data/telemetry.csv"),
                    help="per-frame telemetry CSV (rotated at 20 MB); '' disables")
    ap.add_argument("--log", help="append per-frame results as JSON lines")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
    a = ap.parse_args()
    if a.merc_inference == "framenpu" and not a.framenpu_jobs:
        ap.error("--merc-inference framenpu requires --framenpu-jobs")
    bad = [f for f in a.curl_geo_fingers.split(",") if f and f not in FINGERS]
    if bad:
        ap.error("--curl-geo-fingers: unknown finger(s) %s (use %s)" % (",".join(bad), ",".join(FINGERS)))
    if a.merc_views == "stereo3d" and not a.merc_s3d_model:
        ap.error("--merc-views stereo3d needs --merc-s3d-model")
    if a.merc_model and a.merc_inference != "cpu":
        ap.error("--merc-model needs --merc-inference cpu")
    if a.merc_inference == "framenpu" and a.backend != "mercury":
        ap.error("--merc-inference framenpu requires --backend mercury")
    if a.discover == "mercury" and a.backend != "mercury":
        ap.error("--discover mercury requires --backend mercury")
    if a.discover == "mercury" and a.mercury_shadow:
        ap.error("--mercury-shadow compares against MediaPipe discovery")
    import handtrack.fasttrack as _ft
    _ft.MAX_SPEED = a.max_speed
    _ft.LK["maxLevel"] = a.lk_levels

    # stay off the prime core (7, the one that overheated) and below normal priority
    try:
        os.sched_setaffinity(0, cpus_from_env())
        os.nice(10)
    except (AttributeError, OSError):
        pass
    if a.uclamp_max is not None:
        # before any worker pool exists, so MediaPipe / Mercury workers inherit the clamp
        from handtrack import uclamp
        try:
            n = uclamp.set_process_max(int(a.uclamp_max))
            print("uclamp: max %d on %d thread(s)" % (int(a.uclamp_max), n), file=sys.stderr, flush=True)
        except (OSError, ValueError) as e:
            print("uclamp: not applied (%s)" % e, file=sys.stderr, flush=True)
    thermal = ThermalGuard()
    tel = Telemetry(a.telemetry) if a.telemetry else None
    # the source must be set in the environment before any worker process is spawned
    os.environ[camsource.ENV] = a.source
    src = camsource.active()
    src.resample = a.det_resample
    shm = PtShm(src.shm)
    print("camera source: %s (%s, %dx%d published, %.0f Hz)" % (src.name, src.shm, shm.w, shm.h, 1e9 / src.period_ns),
          file=sys.stderr, flush=True)
    # Mercury crops from the full-resolution camera frame when the detector image is a downscale of it
    def crop_is_raw(mode):
        return mode == "raw" or (mode == "auto" and shm.w > a.size)
    merc_raw = crop_is_raw(a.merc_crop)
    if a.backend == "mercury":
        print("mercury crop source: %s" % ("camera frame %dx%d" % (shm.w, shm.h) if merc_raw else
                                           "detector image %d" % a.size), file=sys.stderr, flush=True)
    # MediaPipe's own handedness labels are correct for these cameras (user test 2026-10-10:
    # inverting them swapped the hands), so keep them as-is (mirrored=True = no inversion).
    # MediaPipe runs in its own process (handtrack/mpworker.py) so it can never stall the 60 Hz loop
    import multiprocessing
    from concurrent.futures import Future, ProcessPoolExecutor
    from handtrack import mpworker
    mp_pool = None if a.discover == "mercury" else ProcessPoolExecutor(
        1, mp_context=multiprocessing.get_context("spawn"), initializer=mpworker.init, initargs=(a.gpu, True))
    dets = None
    viewer = None if a.no_http else Viewer()
    if viewer:
        viewer.serve()
        print("debug view: ssh -L 8765:127.0.0.1:8765 frame  ->  http://localhost:8765", file=sys.stderr)
    logf = open(a.log, "a") if a.log else None

    palm = PalmTracker(a.size, source=src) if not a.no_bridge or a.backend == "mercury" else None
    mt = None
    merc_pool = None
    if a.backend == "mercury":
        # continuous per-hand tracking with Mercury's keypoint model, in its own process
        from handtrack.merctrack import MercuryTracker
        merc_pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=mpworker.init_mercury_proc,
                                        initargs=(a.merc_inference, a.framenpu_jobs, a.npu_socket, a.merc_model,
                                                  a.merc_s3d_model, a.merc_cell_offset, a.det_model))
        mt = MercuryTracker(palm, merc_pool, mpworker)
        if a.frame_ring:
            import atexit
            from handtrack.framering import FrameRing
            mt.ring = FrameRing((shm.h, shm.w))
            atexit.register(mt.ring.close)
        mt.curl_source = a.curl_source
        mt.sched = a.merc_sched
        mt.predict_roi = a.merc_predict_roi
        mt.stereo_ms = a.merc_stereo_ms
        mt.views = a.merc_views
        mt.max_crops = a.merc_crops
        mt.s3d_curls = a.merc_s3d_curls
        mt.stereo_comp = a.merc_stereo_comp == "on"
        mt.curl_latest = a.curl_latest
        mt.curl_geo = [FINGERS.index(f) for f in a.curl_geo_fingers.split(",") if f]
        mt.hint_reset = a.merc_hint_reset
        print("mercury: stereo<=%.0f ms, views=%s, motion comp=%s, curls=%s%s, hint reset=%s, roi predict=%s, sched=%s" % (
            mt.stereo_ms, mt.views, mt.stereo_comp, "latest" if mt.curl_latest else "blend",
            (" (single-view joints: %s)" % a.curl_geo_fingers) if mt.curl_geo else "", mt.hint_reset,
            mt.predict_roi, mt.sched),
            file=sys.stderr, flush=True)
    bridge = None if a.no_bridge else Bridge()
    if bridge is not None:
        bridge.splay = a.splay

    def new_stats():
        return {"frames": 0, "hands": 0, "lat": [], "det": [], "pairs": 0, "palms": 0, "monoerr": [],
                "fast_ms": [], "sent": 0, "id_fix": 0, "merc_ms": [], "merc_ishand": [], "merc_e2d": [], "merc_e3d": []}

    lock = threading.Lock()
    stats = new_stats()
    stop = threading.Event()
    # Dashboard stop sends SIGTERM. Let the worker retire its image and release
    # held power before ProcessPoolExecutor exits.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    NAME = {0: "cam0 (right)", 1: "cam1 (left)"}

    pool = ThreadPoolExecutor(2)

    def prep(cam, img, t_ns):
        # ptgrab normally delivers frames already downscaled on the GPU; source.det_image crops,
        # shrinks and rotates as needed
        small = src.det_image(img, cam, a.size)
        td = time.perf_counter()
        hands = dets[cam].detect(small, t_ns / 1e9)
        return small, hands, time.perf_counter() - td

    def reassociate(poses, t, last_sent):
        """Re-label MediaPipe results by where the tracked hands physically are.
        Returns (poses, description-of-swap or '')."""
        tracked = {sd: v[1] for sd, v in last_sent.items() if t - v[0] < 0.5}
        dets = {sd: p for sd, p in poses.items() if p is not None}
        if not dets or (not tracked and len(dets) < 2):
            return poses, ""
        dist = lambda p, sd: float(np.linalg.norm(np.asarray(p[0]) - tracked[sd])) if sd in tracked else 9.0
        other = {"left": "right", "right": "left"}
        new = None
        if len(dets) == 2 and len(tracked) == 2:
            keep = dist(dets["left"], "left") + dist(dets["right"], "right")
            swap = dist(dets["left"], "right") + dist(dets["right"], "left")
            if swap < keep - 0.05:
                new = {"left": dets["right"], "right": dets["left"]}
        elif len(dets) == 2:
            # not both tracked yet: geometry beats labels when they clearly disagree (left hand further left)
            if dets["left"][0][0] > dets["right"][0][0] + 0.10:
                new = {"left": dets["right"], "right": dets["left"]}
        elif len(dets) == 1:
            sd, p = next(iter(dets.items()))
            od = other[sd]
            if od in tracked and dist(p, od) < 0.08 and dist(p, sd) > 0.15:
                new = {od: p, sd: None}
        if new is None:
            return poses, ""
        out = {"left": None, "right": None}
        for sd, p in new.items():
            if p is not None:     # rebuild with the correct hand's geometry (palm axes depend on handedness)
                out[sd] = palm._pose(sd, p[3], t, p[2]) + (p[4],)
        return out, "->".join(sorted(k for k, v in new.items() if v is not None))

    def mono_vs_stereo(poses, h1):
        """On stereo cycles: how far the left camera's single-camera palm estimate is from stereo (m)."""
        errs = []
        R, t = palm.pose[1]
        for side, p in poses.items():
            if p is None:
                continue
            for h in h1:
                if h.side != side:
                    continue
                Pc = palm.mono_points(1, h)
                if Pc is not None:
                    Pm = (Pc * palm.scale.get(side, 1.0)) @ R.T + t
                    errs.append(float(np.linalg.norm(Pm[PALM_IDX].mean(0) - p[3][PALM_IDX].mean(0))))
        return errs

    def coordinator():
        """Fast path every camera frame: optical-flow palm tracking in both cameras ->
        triangulated position -> send (orientation + finger curls from the latest
        MediaPipe result).  Slow path in the background: MediaPipe on alternating
        cameras (stereo every `stereo_every`-th run), which re-seeds the fast tracker."""
        nonlocal stats
        last = {0: 0, 1: 0}
        fast = FastTracker(palm, orient=a.fast_orient, blend_s=a.blend_ms / 1000.0)
        fast.v2 = a.flow_v2
        fast.agree = a.flow_agree
        v2_print = [time.monotonic()]

        def v2_report():
            if a.flow_v2 and time.monotonic() - v2_print[0] > 5.0:
                v2_print[0] = time.monotonic()
                print("   flow v2: %s" % (dict(fast.stats) or "{}"), file=sys.stderr, flush=True)
                if tel is not None and fast.stats:
                    tel.write("V,%d,%s" % (time.monotonic_ns(), ";".join("%s=%d" % kv for kv in sorted(fast.stats.items()))))
                fast.stats.clear()
        rot_tau = a.rot_smooth_ms / 1000.0
        # adaptive: strong smoothing when the hand is nearly still, little when it moves fast
        pos_filt = {sd: OneEuro(1.2, 1.5) for sd in ("left", "right")}
        q_filt = {sd: OneEuro(2.0, 0.6) for sd in ("left", "right")}
        last_q = {}
        slow = {}            # side -> latest MediaPipe pose tuple (pos, quat, conf, P, curls)
        slow_t = {}          # side -> its capture time (s)
        slow_cam_q = {"left": {}, "right": {}}   # side -> slot -> (t, quat, backend) latest orientation per camera
        slow_src = {}        # side -> backend of the latest articulated observation ("merc" / "mp")
        last_sent = {}       # side -> (t, position) last sent to the PC
        last_pkt = [None]    # (poses, ages, t_cap sent) of the last frame packet (curl-only re-sends)
        grid = FrameGrid(src.period_ns)                              # camera frame grid (ns)
        rot_pending = {}     # side -> (t, quat) orientation outlier awaiting confirmation
        shown = {0: None, 1: None}
        job = None           # (future, kind, slot, imgs_at_detection, t_cap)
        shadow = None        # Mercury shadow comparison in flight
        next_merc = 0.0
        last_sub_cap = [None]                # capture time (ns) of the newest submission (--merc-cadence capture)
        next_det = 0.0
        det_pick = 0         # --discover mercury ride-along: the next crop position
        k = 0
        frame_no = 0
        last_hand = 0.0
        adm = Admission(fresh=a.merc_fresh)  # capture admission + held-power release (handtrack/mercsched.py)
        preflow = a.merc_preflow and mt is not None   # --merc-preflow: submit before detector images + flow
        frame_t = {}                         # stage times of the newest processed pair (monotonic ns)
        blocked = ["untracked"]              # why merc_admit last refused to submit (logged with a release)
        def merc_service(between_frames=False, path=None):
            merc_collect()
            merc_admit(between_frames, path)
            merc_power()

        def send_curl_update(sides):
            """A Mercury result collected between camera frames: re-send the last frame's packet with
            the new finger curls now instead of with the next frame (~half a frame period sooner).
            Same capture time and pose, so the PC driver (v4 with curl-only updates) takes only the
            curls; older drivers drop it as a duplicate."""
            if not bridge or last_pkt[0] is None or not sides:
                return
            poses_, ages_, tc_ = last_pkt[0]
            out = dict(poses_)
            changed = []
            for sd in sides:
                p_, sp_ = out.get(sd), slow.get(sd)
                if p_ is None or sp_ is None or len(sp_) < 5:
                    continue
                out[sd] = tuple(p_[:4]) + (sp_[4],)
                changed.append(sd)
            if not changed:
                return
            bridge.send(out, tc_, ages_)
            last_pkt[0] = (out, ages_, tc_)
            if tel is not None:
                t_send = time.monotonic_ns()
                for sd in changed:
                    tel.write("U,%d,%d,%s,%.1f" % (tc_, t_send, sd, (t_send / 1e9 - slow_t[sd]) * 1e3))

        def merc_power():
            """Release held inference power once admission has been blocked for a while."""
            if mt is not None and adm.should_release(time.monotonic(), mt.busy()):
                mt.idle()
                if tel is not None:
                    tel.write("Z,%d,%s" % (time.monotonic_ns(), "stall" if adm.stalled(time.monotonic()) else blocked[0]))

        def reseed_flow(sl_m, side, px_m, img_m, cur_img, t_now, pos3d, P, t_m):
            """Re-seed the fast tracker from a Mercury result (and v2: rebuild a lost camera view)."""
            fast.seed(sl_m, side, types.SimpleNamespace(px=px_m), img_m, cur_img, t_now, pos3d=pos3d)
            try:
                fast.recover_view(side, P, t_m)               # v2: rebuild the other camera if lost
            except Exception as e:                            # never let the experiment stop tracking
                fast.stats["recover_error"] += 1
                if fast.stats["recover_error"] == 1:
                    print("flow v2 recover_view error: %r" % (e,), file=sys.stderr, flush=True)

        def merc_collect(defer=None):
            """Mercury continuous tracking: collect a finished job and publish its observations.
            defer (a list): queue the fast-tracker re-seeds there (arguments bound now) instead of running
            them, so --merc-preflow submits the next job first; the caller runs them before fast.step."""
            nonlocal next_merc, next_det, last_hand
            if mt is not None:
                for side, (pose_m, sl_m, px_m, t_m, img_m) in mt.collect().items():
                    if side not in enabled:                 # result for a hand disabled while in flight
                        continue
                    slow[side], slow_t[side], slow_src[side] = pose_m, t_m, "merc"
                    slow_cam_q[side][sl_m] = (t_m, np.asarray(pose_m[1], float), "merc")
                    args_ = (sl_m, side, px_m, img_m, imgs[sl_m], t_s, pose_m[0], pose_m[3], t_m)
                    if defer is None:
                        reseed_flow(*args_)
                    else:
                        defer.append(args_)
                    last_hand = time.monotonic()
                    if tel is not None:
                        tel.write("K,%d,%s,%d" % (int(t_m * 1e9), side, sl_m))
                if tel is not None:
                    # one row per Mercury observation: capture -> submit -> worker start/end -> collect,
                    # thread CPU vs runqueue wait, hands in the job, outcome, raw (unfiltered) orientation
                    for rc in mt.records:
                        q_ = rc["rq"] if rc["rq"] is not None else (0, 0, 0, 0)
                        # columns 21..24 (zero-based): submit path (frame | early), capture sequence, and the
                        # pair's read / flow-done times (ns): stage ages t_read-t_cap, t_ready-t_read, t_sub-t_ready
                        tel.write("R,%d,%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%d,%s,%.4f,%.4f,%.4f,%.4f,%d,%s,%d,%.3f,%s,%d,%d,%d" % (
                            rc["id"], rc["side"], rc["slot"], rc["t_cap"], rc["t_sub"], rc["t0"], rc["t1"], rc["t_col"],
                            rc["cpu_ns"] / 1e6, rc["wait_ns"] / 1e6, rc["njob"], rc["outcome"], *q_, rc["hint"], rc["inference"], rc["request_crops"], rc["dsp_body_ms"],
                            rc["path"], rc["seq"], rc["t_read"], rc["t_ready"]))
                mt.records.clear()
                if mt.collapsed:
                    # both trackers ended up on one physical hand: forget the dropped one everywhere and
                    # let discovery find the other hand right away
                    fast.hands.pop(mt.collapsed, None)
                    slow.pop(mt.collapsed, None)
                    slow_cam_q[mt.collapsed].clear()
                    next_det = 0.0
                    if tel is not None:
                        tel.write("C,%d,%s" % (t_cap, mt.collapsed))

        def anchors(fp):
            """v2: Mercury anchors single-view depth only to a two-camera position, never to a held
            single-camera depth (that error would otherwise propagate into the joints)"""
            return {sd: v for sd, v in fp.items()
                    if not a.flow_v2 or fast.info.get(sd, {}).get("src") == "stereo"}

        def merc_admit(between_frames, path=None, pending=False):
            """Submit the next job on the newest processed frame (imgs / t_cap / fast_pos) if admitted.
            Blocked admission (no tracked hand, stalled cameras, thermal pause, no placeable crop) is
            not marked active, so merc_power() releases held power after the grace period."""
            nonlocal next_merc, next_det
            if mt is None:
                return
            now = time.monotonic()
            if mt.busy():
                adm.active(now)
                return
            if not mt.tracked() or adm.stalled(now):
                blocked[0] = "untracked" if not mt.tracked() else "stall"
                return
            if a.merc_cadence == "wall" and not rate_cap_due(now, next_merc, grid.period / 1e9):
                adm.active(now)                     # rate cap: still tracking
                return
            # perhand: one crop per job, so the same per-hand rate cap allows n jobs per period
            n_ = max(1, len(mt.hands)) if a.merc_sched == "perhand" else 1
            iv = thermal.interval(a.merc_hz * n_)
            if iv is None:
                blocked[0] = "thermal"
                return
            if a.merc_cadence == "capture" and not capture_cap_due(t_cap, last_sub_cap[0], iv, grid.period):
                adm.active(now)                     # rate cap (in camera frames): still tracking
                return
            if not adm.admissible(now, grid.period / 1e9, between_frames):
                adm.active(now)                     # this capture was used (or is stale): next frame
                return
            # the crop is placed with every fast position (its image location is right); the depth anchor
            # only with two-camera ones (anchors).  pending (--merc-preflow): this capture's detector images
            # and flow do not exist yet -- placed with the previous frame's positions, completed after flow
            fp_ = anchors(fast_pos)
            info = dict(frame_t, path=path or ("early" if between_frames else "frame"), seq=adm.seq)
            if mt.submit(None if pending else imgs, t_cap, fast_pos, anchor_pos=fp_, info=info, raw_imgs=raw_frames):
                adm.submitted(now)
                next_merc = now + iv
                last_sub_cap[0] = t_cap
            else:
                blocked[0] = "unplaceable"          # no crop in front of either camera
                # no job will come back to drop a hand that stays unplaceable: drop it once it is lost
                if mt.expire(t_s):
                    next_det = 0.0


        hands_cfg = EnabledHands()
        enabled = hands_cfg.get()
        have_frame = False
        prof = Profile()
        pv = dict(PROFILE_DEFAULTS)                 # apply the defaults once, then live changes
        send_off_ns = 0
        while not stop.is_set():
            if pv is None:
                pv = prof.poll()
            if pv is not None:
                for sd in ("left", "right"):
                    if palm is not None:
                        fp_, fz_, fx_ = palm.filt[sd]
                        fp_.set(pv["centre_filter"])
                        fz_.set(pv["axis_filter"])
                        fx_.set(pv["axis_filter"])
                        palm.curl_filt[sd].set(pv["curl_filter"])
                    if bridge is not None:
                        bridge.splay_filt[sd].set(pv["curl_filter"])
                    pos_filt[sd].set(pv["pos_filter"])
                rot_tau = (a.rot_smooth_ms if pv["rot_glide_ms"] is None else pv["rot_glide_ms"]) / 1000.0
                send_off_ns = int(pv["send_offset_ms"] * 1e6)
                merc_raw = crop_is_raw(pv["merc_crop"] or a.merc_crop)
                if mt is not None:
                    mt.views = pv["merc_views"] or a.merc_views
                    if mt.views == "stereo3d" and not a.merc_s3d_model:
                        mt.views = "alternate"                 # no 3D student loaded: the profile cannot select it
                    mt.stereo_comp = (a.merc_stereo_comp == "on") if pv["stereo_comp"] is None else pv["stereo_comp"]
                if tel is not None:   # profile switch marker: the report can split results by profile
                    tel.write("P,%d,%s" % (time.monotonic_ns(), ";".join(
                        "%s=%s" % (k, "off" if v is None else (v if not isinstance(v, tuple) else "%g/%g" % v))
                        for k, v in sorted(pv.items()))))
                pv = None
            en = hands_cfg.get()
            if en != enabled:
                for sd in ({"left", "right"} - en):          # just disabled: forget it everywhere
                    fast.hands.pop(sd, None)
                    slow.pop(sd, None)
                    slow_cam_q[sd].clear()
                    if mt is not None:
                        mt.hands.pop(sd, None)
                enabled = en
                next_det = 0.0
            if bridge:
                bridge.enabled_mask = (1 if "left" in enabled else 0) | (2 if "right" in enabled else 0)
            r0, r1 = shm.read(0, last[0]), shm.read(1, last[1])
            if not r0 or not r1:
                if have_frame and mt is not None:
                    # --merc-early / --merc-fresh: between camera frames, collect a finished Mercury job and
                    # (if admitted) start the next one instead of waiting up to a frame for the next loop pass.
                    # On stalled cameras a finished job is always collected so held power can be released.
                    early = a.merc_early or a.merc_fresh
                    if mt.job is not None and mt.job[0].done() and (early or adm.stalled(time.monotonic())):
                        before = dict(slow_t)
                        merc_collect()
                        if early:
                            send_curl_update([sd for sd in slow_t if slow_t[sd] != before.get(sd)])
                            merc_admit(True)
                    merc_power()
                time.sleep(0.002)
                continue
            if abs(r0[1] - r1[1]) > 8e6:          # not the same capture: wait for the lagging camera
                if r0[1] < r1[1]:
                    last[0] = r0[0]
                else:
                    last[1] = r1[0]
                continue
            last[0], last[1] = r0[0], r1[0]
            frame_t = {"t_read": time.monotonic_ns()}
            frame_no += 1
            have_frame = True
            reseeds = []
            if preflow:
                # a finished job lands on the previous processed frame (its images, flow and time), as a
                # between-frames collect does, so the next one can be submitted on this capture right away;
                # its fast-tracker re-seeds wait until after that submit (still before this frame's flow)
                merc_collect(defer=reseeds)
            t_raw = min(r0[1], r1[1])
            adopted = grid.adopted
            t_cap = grid.update(t_raw)
            if grid.adopted != adopted:
                print("camera frame period: %.2f ms measured (source nominal %.2f ms)"
                      % (grid.adopted / 1e6, src.period_ns / 1e6), file=sys.stderr, flush=True)
            t_s = t_cap / 1e9
            raw_frames = {0: r0[2], 1: r1[2]} if merc_raw else None
            pub_frames = {0: r0[2], 1: r1[2]}
            preflow_sub = False
            if preflow and merc_raw and mt.views != "stereo3d":
                adm.frame(frame_no, frame_t["t_read"] / 1e9)
                before_ = adm.submitted_seq
                merc_admit(False, path="preflow", pending=True)
                merc_power()
                preflow_sub = adm.submitted_seq != before_
            for args_ in reseeds:
                reseed_flow(*args_)
            # this capture's detector images (after a --merc-preflow submit, which does not need them)
            imgs = {0: src.det_image(r0[2], 0, a.size), 1: src.det_image(r1[2], 1, a.size)}

            # ---- fast path: position every camera frame (every other frame when hot)
            tf = time.perf_counter()
            skip_frame = thermal.temp >= thermal.SLOWER_C and frame_no % 2 == 1
            fast_pos = {} if skip_frame else fast.step(imgs, t_s)
            v2_report()
            fast_ms = (time.perf_counter() - tf) * 1e3
            frame_t["t_ready"] = time.monotonic_ns()
            adm.frame(frame_no, frame_t["t_ready"] / 1e9)
            if preflow_sub:
                mt.complete(imgs, anchors(fast_pos), frame_t["t_ready"])
            if a.merc_fresh:
                # submit on this frame as soon as its flow positions exist, before MediaPipe handling
                merc_service()

            # ---- slow path: collect a finished MediaPipe run, re-seed the tracker
            if job and job[0].done():
                fut, kind, slot, det_imgs, t_det = job
                job = None
                if t_det is None:                  # a ride-along discovery: its capture is the tracking job's
                    det_imgs, t_det = det_imgs.get("det_imgs"), det_imgs.get("t_cap")
                    if det_imgs is None:           # never attached (it failed before a submit)
                        det_imgs, t_det = dict(imgs), t_cap
                try:
                    res = fut.result()             # {slot: (hands, det_seconds)}
                except Exception as e:             # noqa: BLE001 -- a failed Mercury discovery finds nothing
                    if kind != "mdet":
                        raise
                    print("mercury discovery error: %r" % (e,), file=sys.stderr, flush=True)
                    res = {0: ([], 0.0), 1: ([], 0.0)}
                hands_by_slot = {sl: v[0] for sl, v in res.items()}
                if kind == "mdet":
                    # Mercury-detector discovery: both cameras' records carry the same triangulated joints
                    poses_d = palm.fuse(hands_by_slot[0], hands_by_slot[1], t_det / 1e9)
                    merr = []
                elif kind == "stereo":
                    poses_d = palm.fuse(hands_by_slot[0], hands_by_slot[1], t_det / 1e9)
                    merr = mono_vs_stereo(poses_d, hands_by_slot[1])
                    if a.mercury_shadow and shadow is None:
                        # identical-input comparison: Mercury keypoints on the very frames MediaPipe saw,
                        # ROI from MediaPipe's stereo joints (logged only; does not drive the hands)
                        sj = []
                        for sd, pp in poses_d.items():
                            if pp is None:
                                continue
                            for sl in (0, 1):
                                Rc, tc = palm.pose[sl]
                                sj.append((sl, sd, (pp[3] - tc) @ Rc))
                        if sj:
                            shadow = (mp_pool.submit(mpworker.run_mercury, sj, {sl: det_imgs[sl] for sl in (0, 1)}),
                                      t_det, hands_by_slot, {sd: pp[3] for sd, pp in poses_d.items() if pp is not None})
                else:
                    poses_d = palm.mono(slot, hands_by_slot[slot], t_det / 1e9)
                    merr = []
                # identity by position, not by MediaPipe's (single-view, error-prone) left/right label
                poses_d, swapped = reassociate(poses_d, t_det / 1e9, last_sent)
                # one-hand mode: a lone detection carries MediaPipe's single-view label, which is often
                # wrong and has nothing to be checked against.  If it is labelled as the disabled hand
                # but lies on the enabled hand's side of the head, it is the enabled hand.
                if len(enabled) == 1:
                    want = next(iter(enabled))
                    oth = "right" if want == "left" else "left"
                    po = poses_d.get(oth)
                    if poses_d.get(want) is None and po is not None:
                        x = float(po[0][0])
                        if (x < 0.05) if want == "left" else (x > -0.05):
                            poses_d = {want: palm._pose(want, po[3], t_det / 1e9, po[2]) + (po[4],), oth: None}
                            with lock:
                                stats["id_fix"] += 1
                            if tel is not None:
                                tel.write("I,%d,one-hand->%s" % (t_det, want))
                # identity is settled; only enabled hands go any further
                poses_d = {sd: (pp if sd in enabled else None) for sd, pp in poses_d.items()}
                if swapped:
                    with lock:
                        stats["id_fix"] += 1
                    if tel is not None:
                        tel.write("I,%d,%s" % (t_det, swapped))
                nh = sum(len(h) for h in hands_by_slot.values())
                if nh:
                    last_hand = time.monotonic()
                if mt is not None:
                    for sd_, p_ in poses_d.items():
                        if p_ is not None:
                            mt.seed(sd_, p_[3], t_det / 1e9)
                for side, p in poses_d.items():
                    if p is None:
                        continue
                    if mt is not None and side in mt.tracked() and mt.hands[side].t > t_det / 1e9:
                        continue          # Mercury already has a newer observation of this hand
                    slow[side], slow_t[side], slow_src[side] = p, t_det / 1e9, "mp"
                    qn = np.asarray(p[1], float)
                    lq = last_q.get(side)
                    accept = True
                    if lq is not None and t_s - lq[0] < 0.3:
                        dev = math.degrees(2 * math.acos(min(1.0, abs(float(qn @ lq[1])))))
                        if dev > 50:
                            pend = rot_pending.get(side)
                            if pend is not None and t_s - pend[0] < 0.4 and                                     math.degrees(2 * math.acos(min(1.0, abs(float(qn @ pend[1]))))) < 25:
                                rot_pending.pop(side, None)        # confirmed twice: a real turn
                            else:
                                rot_pending[side] = (t_s, qn)      # likely a bad result: wait for a second one
                                accept = False
                    if accept:
                        for sl in ((0, 1) if kind in ("stereo", "mdet") else (slot,)):
                            slow_cam_q[side][sl] = (t_det / 1e9, qn, "mp")
                    # seed the optical-flow tracker with the detection that is geometrically ON this hand
                    pc3 = p[3][[0, 5, 9, 17]].mean(0)
                    for sl, hs in hands_by_slot.items():
                        if not hs:
                            continue
                        target = palm.det_px(sl, pc3)[0]
                        dists = [np.linalg.norm(h.px[[0, 5, 9, 17]].mean(0) - target) for h in hs]
                        k_ = int(np.argmin(dists))
                        if dists[k_] < 60:
                            fast.seed(sl, side, hs[k_], det_imgs[sl], imgs[sl], t_s, pos3d=p[0])
                for sl, hs in hands_by_slot.items():
                    shown[sl] = hs
                with lock:
                    stats["det"] += [v[1] for v in res.values()]
                    stats["frames"] += len(res)
                    stats["hands"] += nh
                    stats["palms"] += sum(p is not None for p in poses_d.values())
                    stats["monoerr"] += merr
                if tel is not None:
                    td_ns = time.monotonic_ns()
                    for sl, (hs_, dsec) in res.items():
                        tel.write("M,%d,%d,%s,%d,%.1f,%d" % (t_det, td_ns, kind, sl, dsec * 1e3, len(hs_)))
            fast.drop_stale(t_s, max_age=2.0)

            # ---- Mercury shadow comparison results
            if shadow is not None and shadow[0].done():
                sfut, s_t, s_hands, s_P = shadow
                shadow = None
                try:
                    sres = sfut.result()
                except Exception as e:                     # never let the experiment break tracking
                    print("mercury shadow error: %r" % (e,), file=sys.stderr, flush=True)
                    sres = []
                rays_by = {}
                for sl, sd, r, dsec, *_ in sres:
                    if r is None:
                        continue
                    ref = next((h for h in s_hands.get(sl, []) if h.side == sd), None)
                    e2 = float(np.median(np.linalg.norm(r["det_px"] - ref.px, axis=1))) if ref is not None else -1
                    with lock:
                        stats["merc_ms"].append(dsec * 1e3)
                        stats["merc_ishand"].append(r["is_hand"])
                        if e2 >= 0:
                            stats["merc_e2d"].append(e2)
                    if tel is not None:
                        tel.write("S,%d,%s,%d,%.1f,%.3f,%.1f,%s" % (s_t, sd, sl, dsec * 1e3, r["is_hand"], e2,
                                                                     " ".join("%.2f" % c for c in r["curls"])))
                    Rc, tc = palm.pose[sl]
                    rays_by.setdefault(sd, {})[sl] = (tc, r["rays"] @ Rc.T)
                for sd, rv in rays_by.items():
                    if 0 in rv and 1 in rv and sd in s_P:
                        (o1, d1), (o0, d0) = rv[1], rv[0]
                        Pm, gap, _, _ = triangulate(np.broadcast_to(o1, d1.shape), d1, np.broadcast_to(o0, d0.shape), d0)
                        e3 = np.linalg.norm(Pm - s_P[sd], axis=1) * 100
                        with lock:
                            stats["merc_e3d"].append(float(np.median(e3)))
                        if tel is not None:
                            tel.write("S3,%d,%s,%.2f,%.2f,%.2f" % (s_t, sd, float(np.median(e3)),
                                                                  float(np.median(e3[[0, 5, 9, 13, 17]])),
                                                                  float(np.median(gap)) * 100))

            # --merc-fresh: a second chance later in the frame (a job that just finished, or a hand MediaPipe
            # just seeded), admitted like a between-frames submit because this frame has aged meanwhile
            merc_service(between_frames=a.merc_fresh, path="late" if a.merc_fresh else None)

            # ---- schedule the next MediaPipe run (rate cap, thermal guard, idle mode)
            now_m = time.monotonic()
            if job is not None and job[4] is None and mt.discover_req is job[3] and now_m - job[3]["t_req"] > 0.2:
                # no tracking job carried the ride-along discovery (the hands were lost): run it standalone
                mt.discover_req, job, next_det = None, None, 0.0
            lost = set(fast.lost)
            if job is None and (now_m >= next_det or lost):
                interval = thermal.interval(a.max_hz)
                if interval is not None:
                    if now_m - last_hand > 1.0:
                        interval = max(interval, 1.0 / a.idle_hz)
                    if mt is not None and enabled and len(mt.tracked() & enabled) >= len(enabled) and not lost:
                        interval = max(interval, 1.0 / a.discover_hz)   # both hands tracked: discovery only
                    next_det = now_m + interval
                    stereo = a.mode == "stereo" or k % a.stereo_every == 0
                    slot = 1 if k % 2 else 0
                    if lost and not stereo:        # a track just failed: re-detect in that camera now
                        slot = min(lost)
                    fast.lost.clear()
                    k += 1
                    det_imgs = dict(imgs)
                    use = (0, 1) if stereo else (slot,)

                    riding = mt.tracked() and mt.last_submit_t is not None and now_m - mt.last_submit_t < 0.1
                    if a.discover == "mercury" and riding and merc_raw and mt.views != "stereo3d":
                        # hands are tracked and their jobs are going out: discovery rides in the next
                        # tracking job (same capture, one service request), never queued ahead of it; one
                        # crop position per run, in turn. (A tracked hand with no job submitted lately --
                        # one that cannot be placed -- must not hold discovery back: standalone then.)
                        fut = Future()
                        mt.discover_req = dict(future=fut, pick=det_pick, ncrop=a.det_crops, det_size=a.size,
                                               t_req=now_m)
                        det_pick += 1
                        job = (fut, "mdet", slot, mt.discover_req, None)
                    elif a.discover == "mercury":
                        # nothing tracked: a standalone run with every crop position (the one NPU-service
                        # client is the Mercury worker, so it runs there)
                        ref_ = mt.ring.put(pub_frames) if mt.ring is not None else None
                        fut = merc_pool.submit(mpworker.run_discover, ref_ if ref_ is not None else pub_frames,
                                               a.size, a.det_crops,
                                               {sd: mt.palm_c(h.P) for sd, h in mt.hands.items()})
                        if ref_ is not None:
                            mt.ring.release_when_done(ref_, fut)
                        job = (fut, "mdet", slot, det_imgs, t_cap)
                    else:
                        fut = mp_pool.submit(mpworker.run, use, {sl: det_imgs[sl] for sl in use}, t_cap / 1e9)
                        job = (fut, "stereo" if stereo else "mono", slot, det_imgs, t_cap)

            # ---- compose and send: fast position + latest orientation / curls
            # (a frame skipped for heat sends nothing: no old state relabelled as new)
            if skip_frame:
                continue
            poses = {}
            ages = {}            # side -> (position age, orientation age) in ms before t_cap (bridge v4)
            tel_rows = []
            for side in ("left", "right"):
                sp = slow.get(side)
                if side not in enabled or sp is None or t_s - slow_t[side] > 1.0:
                    poses[side] = None
                    continue
                fp = fast_pos.get(side)
                pos, q = (fp[0], fp[1]) if fp is not None else (None, None)
                finfo = fast.info.get(side, {}) if fp is not None else {}
                pos_src, pos_t = finfo.get("src", "none"), finfo.get("t_obs", t_s)
                q_src, q_t = ("flow", t_s) if q is not None else (slow_src.get(side, "mp"), slow_t[side])
                if q is None:
                    # average the two cameras' latest orientations: their single-view biases cancel
                    # instead of alternating (which looked like a few-times-a-second wobble)
                    # both cameras only if both are recent: an old estimate drags a turning hand backwards
                    qs_ = [qq for (tq, qq, _) in slow_cam_q[side].values() if t_s - tq < 0.15]
                    if len(qs_) == 2:
                        a0, a1 = qs_[0], qs_[1] if qs_[0] @ qs_[1] >= 0 else -qs_[1]
                        q = (a0 + a1) / np.linalg.norm(a0 + a1)
                        newest = max(slow_cam_q[side].values(), key=lambda e: e[0])
                        q_src, q_t = newest[2] + "_avg2", newest[0]
                    elif slow_cam_q[side]:
                        # newest accepted orientation (never a result the outlier gate rejected)
                        q_t, q, q_be = max(slow_cam_q[side].values(), key=lambda e: e[0])
                        q_src = q_be + "_newest"
                    else:
                        q = sp[1]
                if pos is None and t_s - slow_t[side] < 0.15 and plausible(sp[0]):
                    # fast tracker lost: use MediaPipe's own estimate only if it agrees with the
                    # last position we sent (single-camera depth can be several cm off)
                    ls = last_sent.get(side)
                    if ls is None or t_s - ls[0] > 0.3 or np.linalg.norm(sp[0] - ls[1]) < 0.10:
                        pos = sp[0]
                        pos_src, pos_t = "mediapipe", slow_t[side]
                if pos is None:
                    poses[side] = None
                    continue
                pos = pos_filt[side](pos, t_s)
                # rotation: glide toward the target (exponential, time constant rot_tau) instead of
                # stepping whenever a new MediaPipe orientation arrives
                q = np.asarray(q, float)
                lq = last_q.get(side)
                if lq is None or t_s - lq[0] > 0.3:
                    qs = q / np.linalg.norm(q)
                else:
                    prev = lq[1]
                    if q @ prev < 0:
                        q = -q
                    # adaptive time constant: ~2.5x rot_tau for tiny differences (still hand),
                    # ~0.6x for big ones (fast turn)
                    ang = math.degrees(2 * math.acos(min(1.0, abs(float(q @ prev)))))
                    k_ = min(1.0, max(0.0, (ang - 2.0) / 18.0))
                    tau = rot_tau * (2.5 - 1.9 * k_)
                    al = 1.0 - math.exp(-max(t_s - lq[0], 0.0) / tau) if rot_tau > 0 else 1.0
                    qs = (1 - al) * prev + al * q
                    qs = qs / np.linalg.norm(qs)
                last_q[side] = (t_s, qs)
                q = qs
                last_sent[side] = (t_s, pos)
                poses[side] = (pos, q, sp[2], sp[3], sp[4])
                tel_rows.append((side, pos_src, pos_t, q_src, q_t, slow_t[side], finfo.get("npts", {}), pos, q))
                ages[side] = ((t_s - pos_t) * 1e3, (t_s - q_t) * 1e3)
            if bridge:
                bridge.send(poses, t_cap + send_off_ns, ages)
                last_pkt[0] = (poses, ages, t_cap + send_off_ns)
            if tel is not None and tel_rows:
                t_send = time.monotonic_ns()
                for (sd, psrc, pt, qsrc, qt, ct, npts, pp, qq) in tel_rows:
                    tel.write("F,%d,%d,%s,%s,%.1f,%s,%.1f,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
                        t_cap, t_send, sd, psrc, (t_send / 1e9 - pt) * 1e3, qsrc, (t_send / 1e9 - qt) * 1e3,
                        (t_send / 1e9 - ct) * 1e3, npts.get(0, 0), npts.get(1, 0), *pp, *qq))
            tel_rows.clear()
            lat_ms = (time.monotonic_ns() - t_cap) / 1e6
            with lock:
                stats["lat"].append(lat_ms)
                stats["pairs"] += 1
                stats["fast_ms"].append(fast_ms)
                stats["sent"] += sum(p is not None for p in poses.values())
            if logf and any(p is not None for p in poses.values()):
                logf.write(json.dumps({"t_ns": t_cap, "lat_ms": round(lat_ms, 2),
                                       "palms": {kk: (None if p is None else {"pos": np.round(p[0], 4).tolist(),
                                                                              "quat": np.round(p[1], 4).tolist(),
                                                                              "curl": np.round(p[4], 3).tolist()})
                                                 for kk, p in poses.items()}}) + "\n")
            if viewer and viewer.clients and shown[0] is not None and shown[1] is not None and frame_no % 3 == 0:
                tiles = []
                for c in (1, 0):   # left camera on the left
                    hs = shown[c]
                    from handtrack.detect import draw   # needs MediaPipe: only for this view
                    t = draw(imgs[c], hs)          # newest image, last MediaPipe skeleton
                    for side, ht in fast.hands.items():
                        ct = ht.cams.get(c)
                        if ct is not None:
                            cv2.circle(t, tuple(int(v) for v in ct.centre), 10, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(t, "%s  %d hand(s)  %.0f ms" % (NAME[c], len(hs), lat_ms),
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    tiles.append(t)
                img = np.hstack(tiles)
                y = img.shape[0] - 14
                for side, p in poses.items():
                    if p is not None:
                        cv2.putText(img, "%s palm (head, cm): x %+.0f y %+.0f z %+.0f  curl %s" % (
                            side, *(100 * p[0]), " ".join("%.1f" % c for c in p[4])),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                        y -= 26
                viewer.publish(img)

    threads = [threading.Thread(target=coordinator, daemon=True)]
    for t in threads:
        t.start()
    t0 = tstat = time.monotonic()
    try:
        while not stop.is_set() and (not a.seconds or time.monotonic() - t0 < a.seconds):
            time.sleep(0.25)
            now = time.monotonic()
            if now - tstat > 5:
                with lock:
                    st = stats
                    stats = new_stats()
                n = max(st["frames"], 1)
                lat = sorted(st["lat"]) or [0]
                me = sorted(st["monoerr"])
                if st["id_fix"]:
                    print("   identity corrections (MediaPipe label disagreed with position): %d" % st["id_fix"],
                          file=sys.stderr, flush=True)
                fm = sorted(st["fast_ms"]) or [0]
                dt = now - tstat
                print("[%.0f C] fast: %.1f frames/s, %.1f ms/frame, %.1f hand poses/s, latency p50 %.0f p95 %.0f ms"
                      "  | MediaPipe: %.1f runs/s, %.1f ms, hands/run %.2f  | mono-vs-stereo %s  hand scale %s"
                      % (thermal.temp, st["pairs"] / dt, fm[len(fm) // 2], st["sent"] / dt,
                         lat[len(lat) // 2], lat[int(len(lat) * 0.95)],
                         st["frames"] / dt, 1e3 * sum(st["det"]) / n, st["hands"] / n,
                         ("p50 %.1f cm (n=%d)" % (100 * me[len(me) // 2], len(me))) if me else "-",
                         {k: round(v, 2) for k, v in (palm.scale.items() if palm else [])}),
                      file=sys.stderr, flush=True)
                if mt is not None:
                    ms = mt.stats
                    print("   mercury tracking: %d crops (%.1f/s), %.1f ms median, accepted %d (stereo %d, single-view %d), "
                          "rejected %d, collapses %d, seeds blocked %d, tracked %s" % (
                              ms["runs"], ms["runs"] / dt, np.median(ms["ms"]) if ms["ms"] else 0, ms["accepted"],
                              ms["stereo"], ms["mono"], ms["rejected"], ms.get("collapses", 0), ms.get("seed_blocked", 0),
                              sorted(mt.tracked())), file=sys.stderr, flush=True)
                    mt.stats = {"runs": 0, "accepted": 0, "rejected": 0, "stereo": 0, "mono": 0, "ms": []}
                if st["merc_ms"]:
                    print("   mercury shadow: %d inferences, %.1f ms median | is_hand>=0.97 %.0f%% | vs MediaPipe: 2D median "
                          "%s px, 3D median %s cm" % (len(st["merc_ms"]), np.median(st["merc_ms"]),
                                                      100 * np.mean(np.array(st["merc_ishand"]) >= 0.97),
                                                      ("%.1f" % np.median(st["merc_e2d"])) if st["merc_e2d"] else "-",
                                                      ("%.1f" % np.median(st["merc_e3d"])) if st["merc_e3d"] else "-"),
                          file=sys.stderr, flush=True)
                tstat = now
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(1)
        if merc_pool is not None:
            try:
                merc_pool.submit(mpworker.close_mercury).result()
            finally:
                merc_pool.shutdown(wait=True)
        if mp_pool is not None:
            mp_pool.shutdown(wait=True)
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
