#!/bin/bash
# Remove the NPU service installed by install_npu_service.sh (the tracker falls back to CPU).
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
systemctl disable --now mrhand-npu.service 2>/dev/null || true
rm -f /etc/systemd/system/mrhand-npu.service
systemctl daemon-reload
rm -rf /var/lib/mrhand-npu
echo "NPU service removed"
