"""Mercury admission: which camera capture to submit, and when to release held inference power.

Pure timing policy (no I/O, no clocks of its own) so it can be unit tested; live.py feeds it.

Rules
- A capture is submitted at most once. Re-running Mercury on the same frame (for example while
  the cameras are stalled) produces no new information and would keep a lost hand "tracked".
- Cameras count as stalled when no new camera pair has been processed for STALL_S. Admission
  stops; a job already in flight finishes and is collected normally.
- Held power (FrameNPU HMX) is released once admission has been blocked for IDLE_S: no tracked
  hand, stalled cameras, a thermal pause, or no crop that can be placed. Short gaps (rate cap,
  waiting for the next frame, one unplaceable crop) keep it, so power is never cycled per crop.
- fresh=True (--merc-fresh): when the worker frees up between camera frames, submit the newest
  frame only if it became ready less than FRESH_FRACTION of a frame period ago; otherwise wait
  for the next frame. With a busy worker, results arrive every W s and the mean age of the newest
  result is (submit age + W + period between results / 2). Submitting a frame that has been ready
  for x s now gives L + x + W + W/2; waiting gives L + W + (W + period - x)/2. Submitting now wins
  only while x < period / 3.
"""

STALL_S = 0.1             # no new camera pair for this long (6 frames at 60 Hz) = cameras stalled
IDLE_S = 0.2              # admission blocked this long -> release held power
FRESH_FRACTION = 1.0 / 3.0


class Admission:
    def __init__(self, fresh=False, stall_s=STALL_S, idle_s=IDLE_S):
        self.fresh = fresh
        self.stall_s, self.idle_s = stall_s, idle_s
        self.seq = 0                  # newest processed capture (0 = none yet)
        self.t_ready = None           # monotonic s at which it was ready to submit
        self.submitted_seq = 0
        self.t_active = None          # last time tracking wanted (or used) the worker
        self.released = False

    def frame(self, seq, t_ready):
        """A new camera pair has been processed (flow positions exist) and may be submitted."""
        self.seq, self.t_ready = seq, t_ready

    def stalled(self, now):
        return self.t_ready is None or now - self.t_ready > self.stall_s

    def admissible(self, now, period_s, between_frames):
        """May the newest processed capture be submitted now?"""
        if self.seq == 0 or self.seq <= self.submitted_seq or self.stalled(now):
            return False
        if between_frames and self.fresh:
            return now - self.t_ready < FRESH_FRACTION * period_s
        return True

    def submitted(self, now):
        self.submitted_seq = self.seq
        self.active(now)

    def active(self, now):
        """Tracking is live (job in flight, or waiting only for the rate cap / next frame)."""
        self.t_active = now
        self.released = False

    def should_release(self, now, busy):
        """True once per idle period: admission has been blocked for idle_s and nothing is in flight."""
        if busy:
            self.active(now)
            return False
        if self.released or self.t_active is None or now - self.t_active < self.idle_s:
            return False
        self.released = True
        return True
