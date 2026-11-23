#!/bin/bash
# Start the hand-tracking pipeline: ptgrab or slamgrab (camera frames), live (tracking + PC bridge),
# and the independent thermal watchdog.  Args are passed to handtrack.live; with no args the
# defaults in ~/handtrack/start_args are used.
# The tracker runs under SCHED_IDLE on HANDTRACK_CPUS (default 5-6) so XRService/compositor
# always win the CPU (passthrough banding was traced to tracker CPU contention).  5-6 keeps it off
# cores 2-4, where XRService head tracking and most SteamVR threads run and the hottest spikes were;
# the compositor's real-time render thread (allowed 4-6) still preempts SCHED_IDLE work there.
# HANDTRACK_CPUS=0-4 restores the previous placement.
cd ~/handtrack/src
export FRAMEHANDS_CORE=${FRAMEHANDS_CORE:-native}   # C++ geometry core (framehands._core); python = the reference
R=$XDG_RUNTIME_DIR/mrhand; mkdir -p $R
ARGS="$*"
[ -z "$ARGS" ] && [ -f ~/handtrack/start_args ] && ARGS=$(cat ~/handtrack/start_args)
export HANDTRACK_CPUS=${HANDTRACK_CPUS:-5-6}
# Camera source: "--source slam" in the args (or HANDTRACK_SOURCE=slam) uses the tracking cameras via
# slamgrab (works in games); otherwise the passthrough cameras via ptgrab (room view only).
SRC=${HANDTRACK_SOURCE:-passthrough}
[[ " $ARGS " =~ \ --source[=\ ]([a-z]+) ]] && SRC=${BASH_REMATCH[1]}
export HANDTRACK_SOURCE=$SRC
if [ "$SRC" = slam ]; then
  GRAB=slamgrab
  # slamgrab needs its file capability (sudo setcap cap_sys_ptrace+p capture/build/slamgrab)
  if ! kill -0 "$(cat $R/slamgrab.pid 2>/dev/null)" 2>/dev/null; then
    nohup nice -n 10 taskset -c 0-6 capture/build/slamgrab > /tmp/slamgrab.log 2>&1 &
    echo $! > $R/slamgrab.pid; sleep 1
  fi
else
  GRAB=ptgrab
  if ! kill -0 "$(cat $R/ptgrab.pid 2>/dev/null)" 2>/dev/null; then
    DISABLE_MRCHROMA=1 DISABLE_MRHAND=1 nohup nice -n 10 taskset -c 0-6 capture/build/ptgrab stream > /tmp/ptgrab.log 2>&1 &
    echo $! > $R/ptgrab.pid; sleep 1
  fi
fi
if ! kill -0 "$(cat $R/live.pid 2>/dev/null)" 2>/dev/null; then
  nohup chrt --idle 0 ../venv/bin/python -m handtrack.live --no-http $ARGS > /tmp/live.log 2>&1 &
  echo $! > $R/live.pid
elif [ -n "$*" ]; then
  # a tracker is already running (e.g. started from the dashboard panel): these args are NOT applied
  echo "WARNING: tracker already running (pid $(cat $R/live.pid)); args NOT applied: $* -- run stop.sh first" >&2
  ARGS="(already running, args not applied)"
fi
if ! kill -0 "$(cat $R/watchdog.pid 2>/dev/null)" 2>/dev/null; then
  nohup ../venv/bin/python -m handtrack.watchdog > /tmp/watchdog.log 2>&1 &
  echo $! > $R/watchdog.pid
fi
echo "running: $GRAB $(cat $R/$GRAB.pid)  live $(cat $R/live.pid)  watchdog $(cat $R/watchdog.pid)  (tracker args: $ARGS, cpus $HANDTRACK_CPUS, SCHED_IDLE)"
