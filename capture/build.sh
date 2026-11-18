#!/bin/bash
# Build VK_LAYER_MRHAND (vrcompositor camera-buffer registry), ptgrab (passthrough frame grabber)
# and slamgrab (tracking-camera frames from XRService buffers).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -std=c++17 -O2 -g -fPIC -shared -fvisibility=hidden -Wall -Wno-missing-field-initializers \
    handcam_layer.cpp -o build/libVkLayer_mrhand.so -lpthread
echo "built build/libVkLayer_mrhand.so"
g++ -std=c++17 -O2 -Wall ptgrab.cpp -o build/ptgrab -lvulkan -lpthread
echo "built build/ptgrab"
g++ -std=c++17 -O2 -Wall slamgrab.cpp -o build/slamgrab
echo "built build/slamgrab (needs: sudo setcap cap_sys_ptrace+p build/slamgrab)"
