# Integration and performance across the four repositories

Each repository's own CI tests it against pinned versions of the others. These run them together:
the tracker (this repository), framehands, framenpu (with its service) and framenpu-cdsp.

```
python3 integration/setup_env.py /tmp/cand --framenpu ../framenpu --models /tmp/models   # a REF or a checkout per library
set -a; . /tmp/cand/env; set +a
/tmp/cand/venv/bin/python -m unittest discover -s integration -v                          # about 45 s
```

| What | Where |
| --- | --- |
| The libraries' versions, entry points, plugin ABI and the PC driver's packet agree | `test_versions.py` |
| `handtrack.live` end to end: synthetic cameras of two hands at known places, the real service, the bridge as the driver reads it. Position, orientation, curls, hint, mirroring, crops per frame and per request, restarts, a second client, a DSP fault, a killed service | `test_closed_loop.py` (`world.py`, `stack.py`) |
| `framenpu serve` and its client on framenpu-cdsp's lane simulator with the real weights, bit for bit (aarch64's native kernels: to 1e-5 where framenpu-cdsp's tests allow it); framehands' batching on it | `test_device.py` |
| CPU cost of one set of libraries against another, paired on one machine | `perf.py` |
| The running tracker on the headset against `budget.json` (read-only) | `headset.py` |

The closed loop's models are stand-ins that find the drawn joints (`world.BlobSession`), so a wrong
answer means a wrong pipeline, not a wrong network. Trained-model accuracy is not tested here.
Never run it on a headset: it publishes its own cameras under the live tracker's name (it refuses when
they exist).

## Version skew

The root service and the tracker install separately. `FI_SERVICE_PYTHON=<python of another environment>`
runs the service from that one; CI runs both directions against the released set.

## CI

`.github/workflows/integration.yml` runs on every change here and every night on the libraries' `main`.
A library calls it with its own commit:

```yaml
  integration:
    uses: NatTheDyke/steam-frame-hand-tracking/.github/workflows/integration.yml@main
    with:
      framenpu: ${{ github.sha }}        # or framenpu-cdsp, framehands
```

`FI_REQUIRED=1` makes a test that cannot run an error instead of a skip.

## Performance

`perf.py compare` alternates the two environments and judges the paired ratio of CPU time: a benchmark
fails when its whole 95% interval is above 1.10, and says `unresolved` when it cannot tell. It sees
host code only. The DSP's own time, heat and contention exist only on the headset:

```
python3 -I integration/headset.py --seconds 90     # on the headset, tracker running, both hands in view
```
