# Mercury inference backends

Mercury tracking keeps its existing crop geometry, normalization, hint policy,
heatmap decoder, handedness, camera assignment, discovery, optical flow and hand
fit. Inference is selected independently with `--merc-inference cpu|framenpu`.
CPU is the default and still uses the same ONNX Runtime session settings.

The optional FrameNPU backend runs `Conv_0..Conv_139` on the Frame's DSP using
the qualified 55-descriptor replay image. Hint projection, pooling and heads
remain on the CPU. Short live-camera trials alongside Valve tracking confirm
inference-worker CPU savings and a modest two-crop processing-time gain.
Sustained thermal/coexistence qualification and PC-rendering latency remain
unmeasured; CPU stays the default.

## Install and select

Install FrameNPU in the tracker's Python environment (the version with
`framenpu.mercury_runtime.MercurySession`) and build its qualified image in the
FrameNPU source checkout:

```sh
python tools/build_teacher_link.py --stride2 --s2-even --s2-direct --pw-fast \
  --dw-fast --pw-group 8 --replay --output /tmp/mercury-jobs
```

Deploy `link.json`, `teacher_link.bin` and `teacher_link_loader.fjob` together
into, for example, `~/handtrack/framenpu-jobs`. The linked image SHA256 is
`d7a78a81469c10acaf193b0d99cc0415f92d1de845a7a9aa292a7a1ebe691819`;
the loader code SHA256 is
`72280ad93eb4ffcf49e7de26908fa2d8f9056e0d9e1e8618f75eba867a7ba3f0`.
FrameNPU verifies both and the existing Mercury model hash before use. Weights
are still fetched privately via the existing model setup; none are bundled.

CPU, including on machines with no FrameNPU installation:

```sh
python -m handtrack.live --backend mercury --merc-inference cpu
```

FrameNPU needs `/dev/mem` access (currently root) and the firmware supported by
FrameNPU. Stop an existing tracker first. From the tracker checkout, using the
same environment and the user's model/calibration paths:

```sh
sudo env HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  HANDTRACK_CPUS="${HANDTRACK_CPUS:-0-4}" \
  ../venv/bin/python -m handtrack.live --backend mercury \
  --merc-inference framenpu --framenpu-jobs "$HOME/handtrack/framenpu-jobs"
```

The normal unprivileged launcher can also pass these flags, but without device
access the worker logs the reason and uses CPU. There is no automatic elevation.
Mercury shadow comparisons in the MediaPipe worker continue to use CPU; a
separate process must not contend for the same DSP session.

## Submit the observations that exist

`MercuryKeypoint.run()` submits its single observation immediately.
`run_many(observations)` processes the currently available list in chunks of at
most three. One or two crops are sent immediately at their actual count; no
padding, batch timer, accumulation across captures or wait for a third crop is
introduced. Four observations become requests of three and one. Zero-contrast
crops keep a `None` result in their original position and use no inference slot.
Camera, side, flip, hint and result order stay attached to each observation.

The continuous tracker normally has one or two hand observations per cycle.
The existing rate/thermal caps and `--merc-sched batch|perhand` selection remain;
`perhand` still submits one hand at a time. Each request in the worker contains
only the observations supplied by that scheduling decision. Group wall time in
worker telemetry is the latency of every observation in the group; it is not
divided by the crop count. The telemetry also records the active backend,
actual request crop count and DSP body time. `--no-bridge` keeps Mercury's
local palm tracking while disabling UDP output.

## Fallback and lifetime

The CPU session stays available. Missing FrameNPU, unsupported firmware, missing
privileges, a busy DSP, a bad image/model, malformed accelerator output, or an
inference error switches this worker to CPU. It reruns the complete affected
batch on CPU, preserving order. It logs the failure once and never retries or
reopens the accelerator during that worker's lifetime.

HMX power is held between tracking requests and released when the tracker has
no observations or the thermal guard pauses inference. A later observation
resumes the same session. Shutdown waits for the active job, releases power,
retires the image and restores the borrowed arena. The worker also registers a
multiprocessing finalizer. The stop script lets the parent perform that cleanup
before terminating leftover worker processes.

An uncertain DSP completion retains the refusal record and device bytes, as
required by FrameNPU; fallback does not clear gates, restart services, replay a
failed DSP request, or attempt firmware recovery. CPU inference can continue
without touching the DSP. A hard kill cannot run cleanup.

## Plugin contract and validation

A custom `backend=` passed to `MercuryKeypoint` implements:

- `infer(feeds)`: one to three normalized ONNX feed dictionaries, returning one
  dictionary per crop with `heatmap_xy`, `heatmap_depth`, `scalar_extras`, `curls`;
- `idle()`: release active compute resources while keeping reusable state;
- `close()`: release the session.

Run local tests (NumPy + OpenCV; ONNX Runtime and the pinned model enable CPU
parity tests):

```sh
FNPU_MERCURY_MODEL=/path/to/grayscale_keypoint_jan18.onnx \
  python -m unittest discover -s tests -v
```

Tests cover immediate small submissions, 3+1 splitting, blank slots, metadata
identity/hints, CPU default, CPU parity, cleanup and failure latching. The
FrameNPU package separately checks planner byte parity, actual native bounds,
metadata corruption and session ownership/restoration. Its packaged session
passed 16 mixed-count requests on the Frame, with all feature tensors identical
to the prior replay evidence and full arena/power/service restoration. Actual
camera ground-truth accuracy, sustained live coexistence and thermals remain
to be measured before enabling this backend by default.

A subsequent CPU/NPU/NPU/CPU live comparison (four 18-second runs, three seconds
warmup excluded) measured two-crop worker wall time of 34.09 ms CPU versus
31.23 ms FrameNPU, and worker thread CPU of 29.66 versus 17.76 ms. One-crop
wall time was slightly worse (17.87 versus 18.48 ms). Whole-pipeline median
camera-detected-to-collect latency was 74.93 versus 72.07 ms, while accepted
observations/s was 32.45 versus 31.04; the differing live hand/crop mixes and
short trials do not establish a consistent throughput gain. A paired test of
53 identical live crops produced no hand-presence decision flips, with max
0.52 detector-pixel displacement. No ground-truth accuracy claim is made.
FrameNPU's `docs/MERCURY_LIVE.md` and `evidence/mercury-live-validation.json`
contain methodology, limitations and aggregate results.
