#!/bin/bash
# First-time setup on the headset (as steamos, from a clone at ~/handtrack/src):
#
#   git clone https://github.com/NatTheDyke/steam-frame-hand-tracking ~/handtrack/src
#   ~/handtrack/src/headset/setup.sh            the models run on the DSP (a root service; one firmware only)
#   ~/handtrack/src/headset/setup.sh --cpu      the models run on the CPU, no service
#
# On other firmware, setup uses the CPU by itself. Checks first and changes nothing if a check fails.
# Needs sudo for slamgrab's file capability and for the service. Safe to run again: finished steps are kept. Nothing here starts the tracker.
# Paths are literal on purpose; nothing here removes anything.
set -euo pipefail
NPU=1
for a in "$@"; do
  case $a in
    --cpu) NPU= ;;
    *) echo "usage: $0 [--cpu]" >&2; exit 2 ;;
  esac
done
say() { printf '\n== %s\n' "$*"; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }
SUDO() { sudo "$@"; }
[ -n "${SUDO_PASS:-}" ] && SUDO() { echo "$SUDO_PASS" | sudo -S -p "" "$@"; }
SRC=$(cd "$(dirname "$0")/.." && pwd)
CALIB=/home/steamos/handtrack/data/calib/sensorRigCalibration.json
DATASETS=/home/steamos/.local/share/Steam/config/cv/xrservice/datasets

export PIP_DISABLE_PIP_VERSION_CHECK=1

say "checks"
[ "$(uname -m)" = aarch64 ] || die "run this on the headset (aarch64), not on $(uname -m)"
[ "$(id -un)" = steamos ] || die "run as steamos, not $(id -un) (no sudo)"
[ "$SRC" = /home/steamos/handtrack/src ] || die "clone this repository to /home/steamos/handtrack/src (it is in $SRC)"
for c in git g++ curl setcap /usr/bin/python3; do command -v "$c" >/dev/null || die "missing $c"; done
PY=$(/usr/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$PY" = 3.14 ] || die "the system Python is $PY; this release needs 3.14"
R=${XDG_RUNTIME_DIR:-/run/user/1000}
if [ -f "$R/mrhand/live.pid" ] && kill -0 "$(cat "$R/mrhand/live.pid")" 2>/dev/null; then
  die "the tracker is running; stop it first (~/handtrack/stop.sh)"
fi
HAVE=
[ -x /var/lib/mrhand-npu/venv/bin/python ] && HAVE=1
if [ -n "$NPU" ] && [ -z "$HAVE" ]; then
  [ ! -e /var/lib/mrhand-npu ] || die "/var/lib/mrhand-npu is left from a failed install; remove it with: sudo $SRC/headset/uninstall_npu_service.sh"
  free=$(df --output=avail -m /var/lib | tail -1)
  [ "$free" -ge 120 ] || die "/var has $free MB free; the NPU service needs 120 MB there (or use --cpu)"
fi
# the tracker cannot place hands without this headset's own camera calibration, which is never shared
if [ ! -f "$CALIB" ]; then
  found=$(ls -t "$DATASETS"/*/sensorRigCalibration.json 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    cat >&2 <<'T'
setup: this headset's camera calibration is not on disk yet. SteamVR writes it into a tracking
recording, so make a short one while WEARING the headset (two seconds is enough):

    /opt/steamvr/bin/linuxarm64/vrcmd --cmdhmd "start_tracking_recording 1 1 1"; sleep 2
    /opt/steamvr/bin/linuxarm64/vrcmd --cmdhmd "stop_tracking_recording"

The recording contains camera images of your room and stays on the headset. Run this script again
afterwards; it then names the recording, which you can delete.
T
    exit 1
  fi
fi
echo ok

say "camera calibration"
if [ ! -f "$CALIB" ]; then
  mkdir -p /home/steamos/handtrack/data/calib
  cp "$found" "$CALIB"
  echo "copied. The recording it came from holds camera images; delete it when you like: $(dirname "$found")"
fi
/usr/bin/python3 -I -c '
import json, sys
c = json.load(open(sys.argv[1])); names = {k["sourceCamera"] for k in c["cameras"]}
sys.exit(0 if {"slam_left", "slam_right"} <= names and "cad" in c else "calibration lacks the tracking cameras")' "$CALIB"

say "Python environment"
[ -x /home/steamos/handtrack/venv/bin/python ] || /usr/bin/python3 -m venv /home/steamos/handtrack/venv
P=/home/steamos/handtrack/venv/bin/python
# not mediapipe: the Mercury pipeline does not use it, and it brings a second, desktop OpenCV
$P -m pip install --quiet numpy opencv-python-headless onnxruntime

say "libraries of headset/PINS"
TAG=$(date +%Y%m%d-%H%M%S)
B=/home/steamos/handtrack/build-$TAG
mkdir "$B" "$B/wheels"
while read -r pin; do
  name=${pin%%==*}; ver=${pin#*==}
  git init --quiet -b pinned "$B/$name"
  git -C "$B/$name" fetch --quiet --depth 1 "${FRAME_GIT:-https://github.com/NatTheDyke}/$name" "refs/tags/v$ver"
  git -C "$B/$name" reset --quiet --hard FETCH_HEAD
  $P -m pip wheel --quiet --no-deps -w "$B/wheels" "$B/$name"
done < <(grep -v '^\s*#' "$SRC/headset/PINS" | grep .)
ls "$B/wheels"

say "models (fetched by hash)"
mkdir -p /home/steamos/handtrack/models/mercury
$P "$B/framenpu/tools/fetch_models.py" /home/steamos/handtrack/models/mercury \
  --only mercury-keypoint-jan18 mercury-ft7 mercury-detector-160

say "camera reader (slamgrab)"
mkdir -p "$SRC/capture/build"
g++ -std=c++17 -O2 -Wall "$SRC/capture/slamgrab.cpp" -o "$SRC/capture/build/slamgrab"
SUDO setcap cap_sys_ptrace+p "$SRC/capture/build/slamgrab"

$P -m pip install --quiet --no-deps --no-index --force-reinstall "$B"/wheels/*.whl

# the DSP backend runs on the one firmware it was verified on
if [ -n "$NPU" ] && [ -z "$HAVE" ]; then
  $P -I -c '
import hashlib, sys
from framenpu_cdsp.profile import PROFILE as p
try:
    ok = all(hashlib.sha256(open(f, "rb").read()).hexdigest() == want
             for f, want in ((p.firmware_path, p.firmware_sha256), (p.qcom_path, p.qcom_sha256)))
except OSError:
    ok = False
sys.exit(0 if ok else 1)' || { NPU=; echo "This headset's DSP firmware is not the one framenpu-cdsp supports: using the CPU."; }
fi

say "start scripts and settings"
install -m 755 "$SRC/headset/start.sh" "$SRC/headset/stop.sh" /home/steamos/handtrack/
mkdir -p /home/steamos/.config/mrhand
# what exists is yours: kept
[ -f /home/steamos/.config/mrhand/profile ] || cp "$SRC/tools/profile_slam.conf" /home/steamos/.config/mrhand/profile
[ -f /home/steamos/.config/mrhand/hands ] || echo "left right" > /home/steamos/.config/mrhand/hands
if [ ! -f /home/steamos/handtrack/start_args ]; then
  if [ -n "$NPU" ]; then
    cp "$SRC/headset/start_args" /home/steamos/handtrack/start_args
  else
    sed 's#--merc-inference service#--merc-inference cpu --merc-model ~/handtrack/models/mercury/student_ft7.onnx#; s# --merc-crops 4##' \
      "$SRC/headset/start_args" > /home/steamos/handtrack/start_args
  fi
fi
(cd "$SRC" && FRAMEHANDS_CORE=native $P -m handtrack.live --help >/dev/null) || die "the tracker does not start"
(cd "$SRC" && FRAMEHANDS_CORE=native $P -c 'from framehands import _native, calib; assert _native.core; calib.Rig()')

if [ -n "$HAVE" ]; then
  echo "now update the NPU service to the same versions: $SRC/headset/install.sh $B/wheels"
elif [ -n "$NPU" ]; then
  say "NPU service (root, on the DSP it shares with Valve's tracking)"
  SUDO_PASS="${SUDO_PASS:-}" bash "$SRC/headset/install.sh" --first "$B/wheels" || die "the NPU service was not \
installed; the tracker will use the CPU. Remove what it left with: sudo $SRC/headset/uninstall_npu_service.sh"
fi
if [ -n "$NPU$HAVE" ] && grep -q -- "--merc-inference cpu" /home/steamos/handtrack/start_args; then
  echo "your start_args run the CPU; to use the NPU service: cp $SRC/headset/start_args ~/handtrack/start_args"
fi

say "done"
cat <<'T'
Next:
  PC:      pcdriver/install.sh HEADSET_IP   (Linux; Windows: see the README)
  Headset: ~/handtrack/start.sh   and   ~/handtrack/stop.sh   (log: /tmp/live.log)
  Panel:   ~/handtrack/src/overlay/install_overlay.sh   (optional dashboard button)
T
