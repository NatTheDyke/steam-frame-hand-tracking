#!/bin/bash
# Stop the hand-tracking pipeline (tracker, its MediaPipe worker, ptgrab, watchdog).
R=$XDG_RUNTIME_DIR/mrhand
# Let live.py finish outstanding inference and close its Mercury session before
# terminating spawned workers. Do not interrupt DSP teardown with a second kill.
if [ -f "$R/live.pid" ]; then
  live_pid=$(cat "$R/live.pid")
  kill "$live_pid" 2>/dev/null
  for attempt in {1..100}; do
    kill -0 "$live_pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$live_pid" 2>/dev/null; then
    echo "tracker is still shutting down; leaving its workers running" >&2
    exit 1
  fi
fi
names="live ptgrab"
[ "$1" = "--from-watchdog" ] || names="$names watchdog"
for n in $names; do f=$R/$n.pid; [ -f $f ] && kill "$(cat $f)" 2>/dev/null; rm -f $f; done
pkill -x ptgrab 2>/dev/null
# MediaPipe worker processes of the tracker (spawned children); match on our venv path only
for p in $(pgrep -f "handtrack/src/../venv/bin/python -c from multiprocessing"); do kill $p 2>/dev/null; done
echo stopped
