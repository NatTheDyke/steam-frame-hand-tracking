#!/bin/bash
# Install the NPU service (handtrack/npuservice.py): FrameNPU Mercury inference as a small root
# service, so the tracker runs as the user with --merc-inference service.
#
#   sudo headset/install_npu_service.sh /path/to/framenpu-0.1.0-py3-none-any.whl [JOBS_DIR]
#
# Everything the service runs or reads is a root-owned copy in /var/lib/mrhand-npu (its own venv
# with numpy + FrameNPU, the service script, the Mercury model and the qualified job image), so no
# user-writable file is ever executed or loaded as root. The model and image are hash-checked here
# and again by FrameNPU. Remove with headset/uninstall_npu_service.sh.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
U=${SUDO_USER:?run with sudo from the tracking user}
H=$(getent passwd "$U" | cut -d: -f6)
SRC=$(cd "$(dirname "$0")/.." && pwd)
WHEEL=${1:?usage: sudo $0 FRAMENPU_WHEEL [JOBS_DIR]}
JOBS=${2:-$H/handtrack/framenpu-jobs}
MODEL=$H/handtrack/models/mercury/grayscale_keypoint_jan18.onnx
UVENV=$H/handtrack/venv
MODEL_SHA=40c0daa598cedb993b54fff17685231b7465d6db342656c401b01c2029efd1d5
IMAGE_SHA=d7a78a81469c10acaf193b0d99cc0415f92d1de845a7a9aa292a7a1ebe691819
D=/var/lib/mrhand-npu

echo "$MODEL_SHA  $MODEL" | sha256sum -c --quiet
echo "$IMAGE_SHA  $JOBS/teacher_link.bin" | sha256sum -c --quiet
for f in link.json teacher_link_loader.fjob; do [ -f "$JOBS/$f" ] || { echo "missing $JOBS/$f" >&2; exit 1; }; done

systemctl stop mrhand-npu.service 2>/dev/null || true
install -d -m 755 -o root -g root $D $D/model $D/jobs
[ -x $D/venv/bin/python ] || /usr/bin/python3 -m venv $D/venv
PYV=$($D/venv/bin/python -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
SP=$D/venv/lib/$PYV/site-packages
# numpy: copied from the tracker venv (same interpreter), no network needed
cp -a "$UVENV/lib/$PYV/site-packages/"numpy "$UVENV/lib/$PYV/site-packages/"numpy.libs \
      "$UVENV/lib/$PYV/site-packages/"numpy-*.dist-info $SP/
$D/venv/bin/python -m pip install --quiet --no-deps --no-index --force-reinstall "$WHEEL"
install -m 644 -o root -g root "$SRC/handtrack/npuservice.py" $D/npuservice.py
install -m 644 -o root -g root "$MODEL" $D/model/
install -m 644 -o root -g root "$JOBS/link.json" "$JOBS/teacher_link.bin" "$JOBS/teacher_link_loader.fjob" $D/jobs/
chown -R root:root $D
chmod -R go-w $D
$D/venv/bin/python -I -c "import numpy, framenpu.mercury_runtime; print('service venv ok: numpy', numpy.__version__)"

cat > /etc/systemd/system/mrhand-npu.service <<UNIT
[Unit]
Description=Hand tracking NPU service (FrameNPU Mercury inference for steam-frame-hand-tracking)

[Service]
Type=simple
ExecStart=$D/venv/bin/python -I $D/npuservice.py --model $D/model/grayscale_keypoint_jan18.onnx --jobs $D/jobs --user $U
RuntimeDirectory=mrhand-npu
RuntimeDirectoryMode=0755
CPUAffinity=5-6
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now mrhand-npu.service
sleep 1
systemctl --no-pager --lines=3 status mrhand-npu.service | head -8
echo "installed: use --merc-inference service in ~/handtrack/start_args"
