"""Run the tracker against the synthetic world: world.py film -> handtrack.live -> the real NPU service
(on stand-in models) -> the UDP bridge, as the PC driver reads it. One Stack at a time: the camera
shared memory and the bridge port have fixed names. Never run on a headset (the camera path is live there).
"""
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FI_TRACKER") or HERE.parent)     # the tracker checkout under test
sys.path.insert(0, str(HERE))
import world  # noqa: E402

SHM, PORT = Path("/dev/shm/mrhand-slam"), 47811
HAND = struct.Struct("<I8ff5f5f2f")          # FhHand v4 (pcdriver/src/framehands_proto.h)
HEAD = struct.Struct("<IIQQII")


def blocked():
    """Why a Stack cannot run here, or None."""
    if SHM.exists():
        return "%s exists: a tracker's cameras are running here" % SHM
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", PORT))
    except OSError:
        return "UDP port %d is taken" % PORT
    finally:
        s.close()


class Driver(threading.Thread):
    """The PC driver's end of the bridge: pings every 100 ms, keeps every v4 hand packet."""

    def __init__(self):
        super().__init__(daemon=True)
        self.packets, self.pongs, self.bad, self.run_on = [], 0, 0, True
        self.start()

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.05)
        last = 0.0
        while self.run_on:
            if time.monotonic() - last > 0.1:
                last = time.monotonic()
                try:
                    s.sendto(struct.pack("<IIIIQ", 0x444E4846, 4, 1, 0, time.monotonic_ns()), ("127.0.0.1", PORT))
                except OSError:
                    pass
            try:
                d = s.recv(2048)
            except (socket.timeout, ConnectionRefusedError):
                continue
            now = time.monotonic_ns()
            if len(d) == HEAD.size + 2 * HAND.size:
                magic, ver, seq, t_cap, mode, _ = HEAD.unpack_from(d)
                if magic != 0x444E4846 or ver != 4:
                    self.bad += 1
                    continue
                hands = {}
                for k, side in enumerate(("left", "right")):
                    v = HAND.unpack_from(d, HEAD.size + k * HAND.size)
                    hands[side] = dict(valid=v[0], conf=v[1], pos=np.array(v[2:5]), quat=np.array(v[5:9]), age=v[9],
                                       curl=np.array(v[10:15]), splay=np.array(v[15:20]), pos_age=v[20], rot_age=v[21])
                self.packets.append((now, seq, t_cap, hands))
            elif len(d) == 32:
                self.pongs += 1
            else:
                self.bad += 1
        s.close()


class Stack:
    def __init__(self, models, *, fps=30.0, moving=True, service=True, args=(), env=()):
        """models: (keypoint, detector) ONNX files; args: more live.py options; env: more variables.
        FI_SERVICE_PYTHON runs the service from another environment (version skew)."""
        why = blocked()
        if why:
            raise RuntimeError(why)
        self.dir = Path(tempfile.mkdtemp(prefix="fh"))
        home = self.dir / "h"
        for d in ("handtrack/data/calib", "handtrack/models/mercury", ".config/mrhand"):
            (home / d).mkdir(parents=True)
        world.write_calibration(home / "handtrack/data/calib/sensorRigCalibration.json")
        shutil.copy(ROOT / "tools/profile_slam.conf", home / ".config/mrhand/profile")
        self.kp, self.det = (str(Path(m).resolve()) for m in models)
        # HOME is the tracker's whole world: calibration, profile, and the models its CPU fallback opens
        for m, n in ((self.kp, "student_ft7.onnx"), (self.det, "grayscale_detection_160x160.onnx")):
            (home / "handtrack/models/mercury" / n).symlink_to(m)
        self.sock, self.reqlog = str(self.dir / "npu.sock"), self.dir / "req.log"
        self.env = dict(os.environ, HOME=str(home), XDG_RUNTIME_DIR=str(self.dir), HANDTRACK_CPUS=",".join(map(str, sorted(os.sched_getaffinity(0)))),
                        FRAMEHANDS_CORE=os.environ.get("FRAMEHANDS_CORE", "native"), **dict(env))
        self.service_python = os.environ.get("FI_SERVICE_PYTHON") or sys.executable
        self.fps, self.moving, self.args, self.procs = fps, moving, list(args), {}
        self.driver = None
        self.use_service = service

    def _start(self, name, cmd):
        out = open(self.dir / (name + ".log"), "ab")
        self.procs[name] = subprocess.Popen(cmd, env=self.env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        out.close()

    def start_service(self):
        self._start("serve", [self.service_python, str(HERE / "world.py"), "serve", self.sock, self.kp, self.det, str(self.reqlog)])
        end = time.monotonic() + 20
        while not os.path.exists(self.sock):
            if time.monotonic() > end or self.procs["serve"].poll() is not None:
                raise RuntimeError("the service did not start:\n" + self.log("serve"))
            time.sleep(0.02)

    def start_tracker(self):
        base = (ROOT / "headset/start_args").read_text().split()
        self._start("live", [sys.executable, "-m", "handtrack.live", "--no-http", *base, "--npu-socket", self.sock,
                             "--telemetry", str(self.dir / "t.csv"), *self.args])

    def __enter__(self):
        try:
            if self.use_service:
                self.start_service()
            self._start("film", [sys.executable, str(HERE / "world.py"), "film", str(SHM), "600", str(self.fps), str(int(self.moving))])
            end = time.monotonic() + 20
            while not SHM.exists() or SHM.stat().st_size == 0:
                if time.monotonic() > end:
                    raise RuntimeError("no camera:\n" + self.log("film"))
                time.sleep(0.02)
            time.sleep(0.3)
            self.start_tracker()
            self.driver = Driver()
        except BaseException:
            self.__exit__()
            raise
        return self

    def cpu_s(self, name):
        """CPU seconds used by a process and its descendants so far (live ones; Linux /proc)"""
        ticks, todo = 0, [self.procs[name].pid]
        while todo:
            pid = todo.pop()
            try:
                f = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
                ticks += int(f[11]) + int(f[12])
                for t in Path("/proc/%d/task" % pid).iterdir():
                    todo += [int(c) for c in (t / "children").read_text().split()]
            except (OSError, ValueError):
                pass
        return ticks / os.sysconf("SC_CLK_TCK")

    def stop(self, name, timeout=15.0):
        p = self.procs.pop(name, None)
        if p is None:
            return None
        if p.poll() is None:
            p.terminate()
        try:
            return p.wait(timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            return p.wait()

    def __exit__(self, *_):
        for n in ("live", "serve", "film"):
            self.stop(n)
        if self.driver:
            self.driver.run_on = False
            self.driver.join(2)
        SHM.unlink(missing_ok=True)
        # kept for a failure's report if FI_KEEP is set
        if not os.environ.get("FI_KEEP"):
            shutil.rmtree(self.dir, ignore_errors=True)

    def log(self, name):
        p = self.dir / (name + ".log")
        return p.read_text(errors="replace") if p.exists() else ""

    def requests(self):
        if not self.reqlog.exists():
            return []
        return [json.loads(x) for x in self.reqlog.read_text().splitlines()]

    def wait(self, cond, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cond():
                return True
            for n, p in self.procs.items():
                if p.poll() is not None:
                    raise RuntimeError("%s ended early (%s):\n%s" % (n, p.returncode, self.log(n)[-3000:]))
            time.sleep(0.1)
        return False

    def tracked(self, n=1, sides=("left", "right"), since=0):
        """hand packets with every side valid, from packet index `since`"""
        return sum(all(h[s]["valid"] for s in sides) for _, _, _, h in self.driver.packets[since:]) >= n

    def errors(self, since=0):
        """{side: distances (m) of the sent palm from the truth at each packet's observation time}"""
        out = {"left": [], "right": []}
        for _, _, t_cap, hands in self.driver.packets[since:]:
            for s, h in hands.items():
                if h["valid"]:
                    P = world.truth((t_cap - 1e6 * h["pos_age"]) / 1e9, self.moving)[s]
                    out[s].append(float(np.linalg.norm(h["pos"] - P[[0, 5, 9, 17]].mean(0))))
        return out


def quantile(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


__all__ = ["Stack", "Driver", "blocked", "quantile", "world"]
