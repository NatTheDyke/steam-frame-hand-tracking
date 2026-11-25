#!/usr/bin/env python3
"""How the running tracker does on the headset, against integration/budget.json. Read-only: it reads the
tracker's log, /proc and the thermal sensors, and starts, stops and changes nothing.

    python3 -I integration/headset.py [--seconds 90] [--history] [--json OUT]        (on the headset, as the user)

Only log windows with enough Mercury crops count (hands in view: wear the headset and hold both hands up).
--history judges every window of the current log instead of watching (no CPU per crop then).
Exit status: 0 within budget, 1 over, 2 not enough hands seen to say."""
import argparse
import glob
import json
import os
import re
import sys
import time

FAST = re.compile(r"\[(\d+) C\] fast: ([\d.]+) frames/s, ([\d.]+) ms/frame, ([\d.]+) hand poses/s, latency p50 ([\d.]+) p95 ([\d.]+) ms"
                  r".*MediaPipe: ([\d.]+) runs/s")
MERC = re.compile(r"mercury tracking: (\d+) crops \(([\d.]+)/s\), ([\d.]+) ms median, accepted (\d+) \(stereo (\d+), .*rejected (\d+),")
BAD = re.compile(r"using CPU|Traceback|RecoveryRequired")
TICK = os.sysconf("SC_CLK_TCK")


def windows(text):
    out, fast = [], None
    for line in text.splitlines():
        if m := FAST.search(line):
            fast = list(map(float, m.groups()))
        elif (m := MERC.search(line)) and fast:
            n, hz, ms, acc, st, rej = map(float, m.groups())
            out.append(dict(temp=fast[0], fps=fast[1], loop_ms=fast[2], poses=fast[3], lat50=fast[4], lat95=fast[5], det_hz=fast[6],
                            crops=n, crop_hz=hz, request_ms=ms, stereo=st / acc if acc else 0.0, rejected=rej / n if n else 0.0))
            fast = None
    return out


def pids(pattern):
    """processes whose command line matches (this one and its parents never do)"""
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if re.search(pattern, cmd) and int(d[6:]) not in (os.getpid(), os.getppid()):
            out.append(int(d[6:]))
    return out


def tree(pid):
    todo, out = [pid], []
    while todo:
        p = todo.pop()
        out.append(p)
        for f in glob.glob("/proc/%d/task/*/children" % p):
            try:
                todo += map(int, open(f).read().split())
            except OSError:
                pass
    return out


def cpu(pid_list):
    t = 0
    for p in pid_list:
        try:
            f = open("/proc/%d/stat" % p).read().rsplit(")", 1)[1].split()
            t += int(f[11]) + int(f[12])
        except OSError:
            pass
    return t / TICK


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--log", default="/tmp/live.log")
    ap.add_argument("--budget", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "budget.json"))
    ap.add_argument("--json")
    a = ap.parse_args()
    budget = json.load(open(a.budget))
    live = [p for p in pids(r"-m handtrack\.live") if "handtrack" in open("/proc/%d/cmdline" % p).read()]
    if not live and not a.history:
        print("the tracker is not running")
        return 2
    service = pids(r"framenpu serve|framenpu\.service|npuservice")
    size = 0 if a.history else os.path.getsize(a.log)
    c0 = (cpu(service), cpu(tree(min(live)))) if live else (0, 0)
    t0 = time.monotonic()
    if not a.history:
        time.sleep(a.seconds)
    c1, dt = ((cpu(service), cpu(tree(min(live)))) if live else (0, 0)), time.monotonic() - t0
    with open(a.log, errors="replace") as f:
        f.seek(size)
        text = f.read()
    every = windows(text)
    use = [w for w in every if w["crops"] >= budget["min_crops_per_window"]]
    got = {"windows": len(every), "windows used": len(use), "bad log lines": len(BAD.findall(text))}
    if use:
        got.update({"frames/s": median([w["fps"] for w in use]), "request ms (median window)": median([w["request_ms"] for w in use]),
                    "latency p50 ms": median([w["lat50"] for w in use]), "latency p95 ms": median([w["lat95"] for w in use]),
                    "loop ms/frame": median([w["loop_ms"] for w in use]),
                    "stereo share": sum(w["stereo"] * w["crops"] for w in use) / sum(w["crops"] for w in use),
                    "rejected share": sum(w["rejected"] * w["crops"] for w in use) / sum(w["crops"] for w in use),
                    "max C": max(w["temp"] for w in use)}
                   )
        n = sum(w["crops"] for w in every)
        if not a.history and n:
            got["service CPU ms/crop"] = 1e3 * (c1[0] - c0[0]) / n
            got["tracker CPU cores"] = (c1[1] - c0[1]) / dt
            got["detector runs/s"] = median([w["det_hz"] for w in use])
    over = []
    enough = len(use) >= budget["min_windows"]
    print("%-28s %10s  %s" % ("", "measured", "budget"))
    for k, v in got.items():
        lim = budget["limits"].get(k)
        bad = lim is not None and (enough or k == "bad log lines") and not (lim.get("min", v) <= v <= lim.get("max", v))
        if bad:
            over.append(k)
        print("%-28s %10.2f  %s%s" % (k, v, "" if lim is None else " ".join("%s %g" % x for x in lim.items()), "   OVER" if bad else ""))
    missing = [k for k in budget["limits"] if k not in got and not (a.history and k in budget["watch_only"])]
    if missing:
        print("not measured: " + ", ".join(missing))
    if a.json:
        json.dump(dict(got, over=over), open(a.json, "w"), indent=1)
    if over:
        print("OVER BUDGET: " + ", ".join(over))
        return 1
    if not enough or missing:
        print("NOT ENOUGH: %d windows with hands, %d needed" % (len(use), budget["min_windows"]))
        return 2
    print("within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
