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
- The rate cap tolerates a frame arriving up to a quarter period early (rate_cap_due).
- fresh=True (--merc-fresh): when the worker frees up between camera frames, submit the newest
  frame only if it became ready less than FRESH_FRACTION of a frame period ago; otherwise wait
  for the next frame. With a busy worker, results arrive every W s and the mean age of the newest
  result is (submit age + W + period between results / 2). Submitting a frame that has been ready
  for x s now gives L + x + W + W/2; waiting gives L + W + (W + period - x)/2. Submitting now wins
  only while x < period / 3.
"""

RATE_SLACK_FRACTION = 0.25  # rate cap: a capture may arrive this fraction of a frame period early


class FrameGrid:
    """Capture times on the camera's frame grid: the raw pair time (min of the two cameras') snapped
    to a slowly following phase and period, so per-frame jitter does not reach the sent ages.

    The period starts at the source's nominal one, but the cameras' real rate is measured: once
    ADOPT_N intervals are in and the period they show (the camera period itself; a busy loop that
    skips frames only adds multiples of it) is more than ADOPT_TOL away from the nominal period, that
    becomes the nominal period and the grid restarts. XRService's 120 Hz SLAM mode (~114 usable
    frames/s) otherwise ran the 33 ms grid ahead of real time by ~24 ms per frame."""
    ADOPT_N = 24
    ADOPT_TOL = 0.03             # the 30 Hz cameras run within 0.1% of nominal; the grid follows a wrong
                                 # period only slowly (and not past its +-10% clamp)

    def __init__(self, nominal_ns):
        from collections import deque
        self.nominal = self.period = float(nominal_ns)
        self.t = None
        self.prev = None
        self.dts = deque(maxlen=64)
        self.adopted = None          # the measured period (ns) once adopted

    @staticmethod
    def _fits(dts, p):
        """99% of the sorted intervals are whole multiples of p, within a quarter of p (a doubled
        period fits ~98% when the loop mostly gets every 2nd frame: the odd skips land half a period
        off the grid)"""
        res = sorted(abs(d - p * max(1, round(d / p))) for d in dts)
        return res[int(0.99 * (len(res) - 1))] < 0.25 * p

    @classmethod
    def _period(cls, dts):
        """The camera period behind sorted intervals: the largest p (the low end / k) they fit (a
        loop that only ever skips 2 or 3 frames never shows one period directly), as the median of
        each interval / its frame count."""
        low = dts[len(dts) // 20]
        for k in (1, 2, 3, 4):
            p = low / k
            if cls._fits(dts, p):
                break
        per = sorted(d / max(1, round(d / p)) for d in dts)
        return per[len(per) // 2]

    def update(self, t_raw):
        """The newest pair's raw capture time (ns) -> its grid capture time (int ns)."""
        if self.prev is not None and t_raw > self.prev:
            self.dts.append(t_raw - self.prev)
        self.prev = t_raw
        if len(self.dts) >= self.ADOPT_N:
            dts = sorted(self.dts)
            # hysteresis: a measured period that still fits is kept (multiples of the period fit its
            # halves too); the source's nominal period is not, since a loop that gets every 3rd-4th
            # 114 Hz frame produces intervals that fit 33 ms
            keep = self.adopted is not None and self._fits(dts, self.nominal)
            low = self.nominal if keep else self._period(dts)
            if abs(low - self.nominal) > self.ADOPT_TOL * self.nominal:
                self.nominal = self.period = self.adopted = float(low)
                self.t = None
                self.dts.clear()
        if self.t is None or abs(t_raw - self.t) > 1e9:
            self.t = float(t_raw)
        else:
            nfr = max(1, round((t_raw - self.t) / self.period))
            pred = self.t + nfr * self.period
            resid = t_raw - pred
            self.t = pred + 0.05 * resid                       # follow phase slowly
            self.period += 0.002 * resid / nfr                 # and the true frame period
            self.period = min(max(self.period, 0.9 * self.nominal), 1.11 * self.nominal)
        return int(self.t)


def capture_cap_frames(interval_s, period_s):
    """Camera frames between submissions for a cap interval: the nearest whole number (at least 1)."""
    return max(1, int(round(interval_s / period_s)))


def capture_cap_due(t_cap_ns, last_cap_ns, interval_s, period_ns):
    """--merc-cadence capture: the rate cap counted in camera frames instead of wall time.

    A wall-clock cap of 33.3 ms over 8.74 ms (120 Hz) captures submits after 3, 4 or 5 frames as
    loop timing jitters; each 5-frame gap (43.7 ms) leaves the other camera's view older than
    merctrack.STEREO_MS, so that crop cannot be triangulated.  Counting frames gives a fixed
    capture_cap_frames gap (4 frames, 35 ms, at 120 Hz; every frame at 30 Hz).  Half a frame of
    slack absorbs capture-time jitter; a frame the loop skipped is replaced by the next one."""
    if last_cap_ns is None:
        return True
    k = capture_cap_frames(interval_s, period_ns / 1e9)
    return t_cap_ns - last_cap_ns >= (k - 0.5) * period_ns


def rate_cap_due(now, next_t, period_s):
    """The Mercury rate cap (--merc-hz) allows a new submission at next_t (previous submission + the
    cap interval).  Camera frames arrive one period apart with ~1 ms jitter, so with the cap equal to
    the frame rate a frame that became ready slightly early was refused and, as nothing retried it,
    that cycle was lost: in a live 30 Hz trial 65% of frames were skipped (one result per 67 ms).
    Allow a frame up to RATE_SLACK_FRACTION of a period early; each capture is still submitted at
    most once (Admission), so this never runs Mercury faster than the cameras."""
    return now >= next_t - RATE_SLACK_FRACTION * period_s


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
