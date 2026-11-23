#!/bin/bash
# Install the pinned library set (headset/PINS) from wheels: framenpu into the NPU service's root venv
# and the tracker venv, framehands into the tracker venv. The service unit runs `framenpu serve` on
# the Mercury image shipped in framenpu_cdsp; the models stay root-owned copies in /var/lib/mrhand-npu/model.
#
#   headset/install.sh WHEEL_DIR          (as steamos; asks sudo for the service side)
#
# Refuses a wheel set that does not match PINS, and a result where the two venvs hold different
# framenpu versions. Stops the tracker first (aborts if it will not stop) and checks the DSP is clean
# before changing anything. What it replaces is kept: the service side in a root-owned
# /var/lib/mrhand-npu/backup-libs-TAG, the tracker side in ~/handtrack/rollback/TAG, which holds
# rollback.sh; ~/handtrack/rollback-libs.sh links the newest.
#
# Paths are literal on purpose (steamos, python3.14): nothing here removes a computed path.
set -euo pipefail
[ "$(id -un)" = steamos ] || { echo "run as steamos" >&2; exit 1; }
SRC=$(cd "$(dirname "$0")/.." && pwd)
WHEELS=$(cd "${1:?usage: $0 WHEEL_DIR}" && pwd)
SUDO() { sudo "$@"; }
[ -n "${SUDO_PASS:-}" ] && SUDO() { echo "$SUDO_PASS" | sudo -S -p "" "$@"; }
/var/lib/mrhand-npu/venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 14)' \
  || { echo "service venv is not python3.14" >&2; exit 1; }
/home/steamos/handtrack/venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 14)' \
  || { echo "tracker venv is not python3.14" >&2; exit 1; }
SSP=/var/lib/mrhand-npu/venv/lib/python3.14/site-packages
USP=/home/steamos/handtrack/venv/lib/python3.14/site-packages

# ---- the wheel set must be exactly PINS
declare -A WHL
while read -r pin; do
  name=${pin%%==*}; ver=${pin#*==}
  found=("$WHEELS/$name-$ver-"*.whl)
  [ ${#found[@]} = 1 ] && [ -f "${found[0]}" ] || { echo "no single $name-$ver wheel in $WHEELS" >&2; exit 1; }
  WHL[$name]=${found[0]}
done < <(grep -v '^\s*#' "$SRC/headset/PINS" | grep .)
for n in framenpu framehands; do [ -n "${WHL[$n]:-}" ] || { echo "PINS lacks $n" >&2; exit 1; }; done
FNPU_VER=$(grep '^framenpu==' "$SRC/headset/PINS" | cut -d= -f3)
FH_VER=$(grep '^framehands==' "$SRC/headset/PINS" | cut -d= -f3)

# ---- stop the tracker; the DSP must be clean with nothing running
bash /home/steamos/handtrack/stop.sh >/dev/null 2>&1 || true
sleep 2
if pgrep -u steamos -f "python -m handtrack[.]live" >/dev/null; then echo "tracker did not stop: abort" >&2; exit 1; fi
SUDO /var/lib/mrhand-npu/venv/bin/python -I -c '
try:
    from framenpu_cdsp.transport import Transport
except ImportError:  # a service venv from before the split
    from framenpu.transport import Transport
t = Transport(); s = t.status(); t.close(); print("dsp", s)
raise SystemExit(0 if s["control"] == [1, 0, 1] and s["callback"] == 0 and not s["poisoned"] else 1)' \
  || { echo "DSP not clean: abort (tracker left stopped)" >&2; exit 1; }

# ---- keep what this replaces (fresh folders)
TAG=$(date +%Y%m%d-%H%M%S)
RB=/home/steamos/handtrack/rollback/$TAG
SB=/var/lib/mrhand-npu/backup-libs-$TAG
mkdir -p /home/steamos/handtrack/rollback
mkdir "$RB" "$RB/user-sp" "$RB/src"
cp -a "$USP"/framenpu* "$USP"/framehands* "$RB/user-sp/" 2>/dev/null || true
for p in framenpu framenpu_cdsp framehands; do
  if [ -e "/home/steamos/handtrack/src/$p" ]; then cp -a "/home/steamos/handtrack/src/$p" "$RB/src/"; fi
done
SUDO mkdir "$SB" "$SB/sp" "$SB/unit"
SUDO sh -c "cp -a $SSP/framenpu* $SB/sp/; cp -a /etc/systemd/system/mrhand-npu.service $SB/unit/; \
  if [ -d /etc/systemd/system/mrhand-npu.service.d ]; then cp -a /etc/systemd/system/mrhand-npu.service.d $SB/unit/; fi"
cat > "$RB/rollback.sh" <<RBS
#!/bin/bash
# Undo the library install of $TAG: venv packages, the src copies and the service unit.
#   bash $RB/rollback.sh          (as steamos; asks sudo)
set -euo pipefail
SUDO() { sudo "\$@"; }
[ -n "\${SUDO_PASS:-}" ] && SUDO() { echo "\$SUDO_PASS" | sudo -S -p "" "\$@"; }
bash /home/steamos/handtrack/stop.sh >/dev/null 2>&1 || true
sleep 2
SUDO systemctl stop mrhand-npu.service
SUDO sh -c 'rm -rf /var/lib/mrhand-npu/venv/lib/python3.14/site-packages/framenpu /var/lib/mrhand-npu/venv/lib/python3.14/site-packages/framenpu_cdsp /var/lib/mrhand-npu/venv/lib/python3.14/site-packages/framenpu-*.dist-info'
SUDO cp -a $SB/sp/. $SSP/
rm -rf /home/steamos/handtrack/venv/lib/python3.14/site-packages/framenpu /home/steamos/handtrack/venv/lib/python3.14/site-packages/framenpu_cdsp /home/steamos/handtrack/venv/lib/python3.14/site-packages/framenpu-*.dist-info /home/steamos/handtrack/venv/lib/python3.14/site-packages/framehands /home/steamos/handtrack/venv/lib/python3.14/site-packages/framehands-*.dist-info
cp -a $RB/user-sp/. $USP/
cp -a $RB/src/. /home/steamos/handtrack/src/
SUDO rm -f /etc/systemd/system/mrhand-npu.service
SUDO cp -a $SB/unit/. /etc/systemd/system/
SUDO systemctl daemon-reload
SUDO systemctl start mrhand-npu.service
sleep 2
bash /home/steamos/handtrack/start.sh
echo "\$(date -Is) rollback: library install $TAG undone ($RB, $SB)" >> /home/steamos/handtrack/CHANGES.log
echo "rolled back $TAG"
RBS
chmod 755 "$RB/rollback.sh"
ln -sfn "$RB/rollback.sh" /home/steamos/handtrack/rollback-libs.sh

# ---- install: service venv (framenpu, root-owned), tracker venv (framenpu + framehands); no network
SUDO systemctl stop mrhand-npu.service
SUDO /var/lib/mrhand-npu/venv/bin/python -m pip install --quiet --no-deps --no-index --force-reinstall "${WHL[framenpu]}"
SUDO chown -R root:root /var/lib/mrhand-npu/venv
SUDO chmod -R go-w /var/lib/mrhand-npu/venv
/home/steamos/handtrack/venv/bin/python -m pip install --quiet --no-deps --no-index --force-reinstall \
  "${WHL[framenpu]}" "${WHL[framehands]}"
# the tracker runs from ~/handtrack/src: copies there would shadow the venv packages
rm -rf /home/steamos/handtrack/src/framenpu /home/steamos/handtrack/src/framenpu_cdsp /home/steamos/handtrack/src/framehands

# ---- one version everywhere, imported from the venvs
SV=$(/var/lib/mrhand-npu/venv/bin/python -I -c 'import framenpu, framenpu_cdsp.mercury_device; print(framenpu.__version__)')
UV=$(cd /home/steamos/handtrack/src && /home/steamos/handtrack/venv/bin/python -c '
import framenpu, framehands, handtrack.mpworker
for m in (framenpu, framehands):
    assert m.__file__.startswith("'"$USP"'"), m.__file__
print(framenpu.__version__, framehands.__version__)')
if [ "$SV" != "$FNPU_VER" ] || [ "$UV" != "$FNPU_VER $FH_VER" ]; then
  echo "mixed set: service $SV, tracker $UV (pins $FNPU_VER $FH_VER); roll back with bash $RB/rollback.sh" >&2
  exit 1
fi

# ---- the unit: framenpu serve on the bundled image, the models as before
ARGS="--model /var/lib/mrhand-npu/model/student_ft7.onnx --user steamos --unpinned"
if [ -f /var/lib/mrhand-npu/model/grayscale_detection_160x160.onnx ]; then
  ARGS="$ARGS --det-model /var/lib/mrhand-npu/model/grayscale_detection_160x160.onnx"
fi
cat > "$RB/mrhand-npu.service" <<UNIT
[Unit]
Description=Hand tracking NPU service (framenpu serve, for steam-frame-hand-tracking)

[Service]
Type=simple
ExecStart=/var/lib/mrhand-npu/venv/bin/python -I -m framenpu serve $ARGS
RuntimeDirectory=mrhand-npu
RuntimeDirectoryMode=0755
CPUAffinity=5-6
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
SUDO install -m 644 -o root -g root "$RB/mrhand-npu.service" /etc/systemd/system/mrhand-npu.service
SUDO rm -rf /etc/systemd/system/mrhand-npu.service.d
SUDO systemctl daemon-reload
SUDO systemctl start mrhand-npu.service
for _ in $(seq 30); do [ -S /run/mrhand-npu/npu.sock ] && break; sleep 0.5; done
/home/steamos/handtrack/venv/bin/python -I -c '
from framenpu.service import Client
c = Client(); print("service protocol", c.info["version"], "crops", c.max_crops)
for k, v in sorted(c.manifests.items()):
    print(" model", k, v["name"], (v["weights_sha256"] or "")[:8], "max_crops", v["max_crops"])
assert c.manifests and all(v["name"] for v in c.manifests.values()), "a served model has no manifest"
c.close()'
echo "$(date -Is) installed libraries framenpu $SV + framehands $FH_VER from wheels ($(sha256sum "${WHL[framenpu]}" "${WHL[framehands]}" | cut -c1-12 | tr '\n' ' ')); unit runs framenpu serve on the bundled image; rollback $RB/rollback.sh" \
  >> /home/steamos/handtrack/CHANGES.log
bash /home/steamos/handtrack/start.sh
echo "installed; rollback: bash $RB/rollback.sh"
