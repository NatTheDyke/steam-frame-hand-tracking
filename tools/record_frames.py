#!/usr/bin/env python3
"""Record the camera frames the tracker sees, for offline replay of the whole pipeline.

Reads the same shared memory as handtrack.live (slamgrab -> /dev/shm/mrhand-slam, or ptgrab ->
/dev/shm/mrhand-pt), pairs the two cameras the same way (capture times within 8 ms), and writes
every pair uncompressed, with the capture times the tracker's telemetry uses.  It only reads the
shared memory, so it runs next to the live tracker without changing what the tracker does.  At the
end it copies the tracker's telemetry rows for the recorded interval, the calibration, the
profile and the start arguments, so live results can be lined up with their frames.

  tools/record_frames.py [--seconds 120] [--out DIR] [--source slam|passthrough]

Output DIR (default ~/handtrack/recordings/<date-time>):
  frames.bin      records: PAIR header (<4sIQQQ: b"FPR1", index, seq, t_cap0_ns, t_cap1_ns)
                  then camera 0 (right) and camera 1 (left) luma, h x w uint8 each
  meta.json       source, w, h, frame count, interval, dropped pairs
  telemetry.csv   the tracker's rows (all kinds) with times inside the recording
  calib/, profile, start_args   copies, if present

Reading it back (any machine, numpy only):  for idx, seq, t0, t1, img0, img1 in read_frames(DIR): ...
About 2.2 MB per pair: 65 MB/s at 30 Hz, ~8 GB for two minutes.
"""
import argparse
import json
import mmap
import os
import shutil
import struct
import sys
import time

import numpy as np

SHM = {"slam": "/dev/shm/mrhand-slam", "passthrough": "/dev/shm/mrhand-pt"}
SHM_MAGIC = 0x4D525054                      # MrhandPtHeader (capture/handcam_proto.h), as in live.py
PAIR = struct.Struct("<4sIQQQ")
PAIR_MAGIC = b"FPR1"
SAME_CAPTURE_NS = 8_000_000                 # live.py pairs the cameras within this capture-time gap
HOME = os.path.expanduser("~/handtrack")


class Shm:
    """Read-only view of the frame shared memory (same layout and seqlock rule as live.PtShm)."""

    def __init__(self, path):
        fd = os.open(path, os.O_RDONLY)
        try:
            self.m = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        magic, self.w, self.h, self.cams = struct.unpack_from("<4I", self.m, 0)
        if magic != SHM_MAGIC:
            raise RuntimeError("bad shm magic in %s; is the grabber running?" % path)
        self.frame = self.w * self.h
        self.data0 = struct.calcsize("<4I" + "QQIIQ" * 2 + "2Q")

    def read(self, cam, last_seq):
        """(seq, t_capture_ns, bytes) of the newest frame, or None if unchanged / being written."""
        for _ in range(5):
            seq, t, _buf, _pad, _n = struct.unpack_from("<QQIIQ", self.m, 16 + cam * 32)
            if seq == last_seq or seq == 0:
                return None
            if seq & 1:                         # writer active
                time.sleep(0.0005)
                continue
            off = self.data0 + cam * self.frame
            img = self.m[off:off + self.frame]
            if struct.unpack_from("<Q", self.m, 16 + cam * 32)[0] == seq:
                return seq, t, img
        return None


def telemetry_slice(t0_ns, t1_ns, out_path):
    """Copy telemetry rows whose first time column lies in [t0, t1] (both rotated files)."""
    rows, last_profile = [], None
    for name in ("telemetry.csv.1", "telemetry.csv"):
        p = os.path.join(HOME, "data", name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                cols = line.split(",")
                # R rows: R,id,side,slot,t_cap,...; every other row kind has its time in column 1
                try:
                    t = int(cols[4] if cols[0] == "R" else cols[1])
                except (ValueError, IndexError):
                    continue
                if cols[0] == "P" and t < t0_ns:
                    last_profile = line         # the profile in force when the recording started
                elif t0_ns <= t <= t1_ns:
                    rows.append(line)
    with open(out_path, "w") as out:
        if last_profile is not None:
            out.write(last_profile)
        out.writelines(rows)
    return len(rows)


def record(a):
    shm = Shm(SHM[a.source])
    out = a.out or os.path.join(HOME, "recordings", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=False)
    f = open(os.path.join(out, "frames.bin"), "wb", buffering=8 << 20)
    last = [0, 0]
    pend = [None, None]
    n = dropped = 0
    t_first = t_last = None
    end = time.monotonic() + a.seconds
    next_note = time.monotonic() + 10
    print("recording %s %dx%d to %s for %.0f s" % (a.source, shm.w, shm.h, out, a.seconds), flush=True)
    try:
        while time.monotonic() < end:
            for c in (0, 1):
                r = shm.read(c, last[c])
                if r is not None:
                    if pend[c] is not None:
                        dropped += 1                # a frame of this camera was never paired
                    pend[c] = r
                    last[c] = r[0]
            if pend[0] is None or pend[1] is None:
                time.sleep(0.002)
                continue
            (s0, t0, i0), (s1, t1, i1) = pend
            if abs(t0 - t1) > SAME_CAPTURE_NS:      # not the same capture: drop the older one
                pend[0 if t0 < t1 else 1] = None
                dropped += 1
                continue
            f.write(PAIR.pack(PAIR_MAGIC, n, s0, t0, t1))
            f.write(i0)
            f.write(i1)
            pend = [None, None]
            n += 1
            t_first = t0 if t_first is None else t_first
            t_last = t0
            if time.monotonic() > next_note:
                print("%d pairs, %d dropped" % (n, dropped), flush=True)
                next_note += 10
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    meta = {"source": a.source, "w": shm.w, "h": shm.h, "pairs": n, "dropped": dropped,
            "t_first_ns": t_first, "t_last_ns": t_last, "camera_order": ["right (slot 0)", "left (slot 1)"],
            "pair_header": "<4sIQQQ magic, index, seq, t_cap0_ns, t_cap1_ns"}
    with open(os.path.join(out, "meta.json"), "w") as m:
        json.dump(meta, m, indent=1)
    if t_first is not None:
        rows = telemetry_slice(t_first - 1_000_000_000, t_last + 2_000_000_000, os.path.join(out, "telemetry.csv"))
        print("telemetry rows copied: %d" % rows)
    for src, dst in ((os.path.join(HOME, "data", "calib"), "calib"),
                     (os.path.expanduser("~/.config/mrhand/profile"), "profile"),
                     (os.path.join(HOME, "start_args"), "start_args")):
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, dst))
        elif os.path.exists(src):
            shutil.copy(src, os.path.join(out, dst))
    print("done: %d pairs (%.1f s), %d dropped -> %s" % (
        n, (t_last - t_first) / 1e9 if n > 1 else 0, dropped, out), flush=True)


def read_frames(path):
    """Yield (index, seq, t_cap0_ns, t_cap1_ns, img0, img1) from a recording directory."""
    with open(os.path.join(path, "meta.json")) as m:
        meta = json.load(m)
    w, h = meta["w"], meta["h"]
    with open(os.path.join(path, "frames.bin"), "rb") as f:
        while True:
            hd = f.read(PAIR.size)
            if len(hd) < PAIR.size:
                return
            magic, idx, seq, t0, t1 = PAIR.unpack(hd)
            if magic != PAIR_MAGIC:
                raise ValueError("bad pair header at index %d" % idx)
            buf = f.read(2 * w * h)
            if len(buf) < 2 * w * h:
                return
            img = np.frombuffer(buf, np.uint8).reshape(2, h, w)
            yield idx, seq, t0, t1, img[0], img[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out")
    ap.add_argument("--source", choices=tuple(SHM), default="slam")
    a = ap.parse_args()
    try:
        record(a)
    except (OSError, RuntimeError) as e:
        sys.exit("record_frames: %s" % e)


if __name__ == "__main__":
    main()
