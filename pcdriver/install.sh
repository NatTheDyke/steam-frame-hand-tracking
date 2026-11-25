#!/bin/bash
# Build and register the SteamVR driver on a Linux PC (close SteamVR first):
#
#   pcdriver/install.sh HEADSET_IP       e.g. pcdriver/install.sh 192.168.1.50
#
# The driver goes to ~/.local/share/framehands-driver. Run again to update it; your settings file there
# is kept, only headsetHost is set. Remove: vrpathreg.sh removedriver ~/.local/share/framehands-driver/framehands
set -euo pipefail
IP=${1:?usage: $0 HEADSET_IP}
[[ $IP =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || { echo "not an IPv4 address: $IP" >&2; exit 1; }
command -v g++ >/dev/null || { echo "missing g++" >&2; exit 1; }
REG=$HOME/.local/share/Steam/steamapps/common/SteamVR/bin/vrpathreg.sh
[ -x "$REG" ] || { echo "SteamVR not found ($REG)" >&2; exit 1; }
if pgrep -x vrserver >/dev/null; then echo "close SteamVR first" >&2; exit 1; fi
cd "$(dirname "$0")"
./build.sh
D=$HOME/.local/share/framehands-driver/framehands
mkdir -p "$D"
[ ! -f "$D/framehands_live.ini" ] || cp "$D/framehands_live.ini" "$D/framehands_live.ini.keep"
cp -r framehands/. "$D/"
[ ! -f "$D/framehands_live.ini.keep" ] || mv "$D/framehands_live.ini.keep" "$D/framehands_live.ini"
sed -i "s/^headsetHost = .*/headsetHost = $IP/" "$D/framehands_live.ini"
grep -q "^headsetHost = $IP$" "$D/framehands_live.ini"
"$REG" adddriver "$D"
echo "installed: start SteamVR and stream from the headset at $IP"
