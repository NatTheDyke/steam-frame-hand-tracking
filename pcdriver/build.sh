#!/bin/sh
# Build driver_framehands.so into framehands/bin/linux64 for Linux SteamVR (needs g++).
set -e
cd "$(dirname "$0")"
mkdir -p build framehands/bin/linux64
g++ -O2 -std=c++17 -shared -fPIC -fvisibility=hidden -Wall -Wno-unused-function \
  -Iopenvr -Ithird_party/openvr_handsim \
  src/driver_framehands.cpp third_party/openvr_handsim/hand_simulation.cpp \
  -o framehands/bin/linux64/driver_framehands.so -pthread -static-libstdc++ -static-libgcc
echo built framehands/bin/linux64/driver_framehands.so
