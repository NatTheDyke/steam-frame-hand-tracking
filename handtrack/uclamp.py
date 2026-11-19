"""Per-task utilization clamp (Linux uclamp): cap the CPU frequency our own work asks schedutil for.

The tracker's bursty single-threaded work (MediaPipe, optical flow, Mercury host work) otherwise
drives its cores to their top frequency, where each cycle costs about a third more energy and the
core hot spots spike. uclamp_max limits only the frequency requested on behalf of these tasks;
other processes on the same cores keep theirs. Children inherit the clamp across fork and exec.
Lowering one's own clamp needs no privileges.
"""
import ctypes
import os
import platform
import struct

_SCHED_SETATTR = {"aarch64": 274, "x86_64": 314}
_FLAGS = 0x08 | 0x10 | 0x40   # SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MAX


def _set(tid, value):
    nr = _SCHED_SETATTR.get(platform.machine())
    if nr is None:
        raise OSError("sched_setattr syscall number unknown for " + platform.machine())
    # struct sched_attr, SCHED_ATTR_SIZE_VER1 (56 bytes): size, policy, flags, nice, priority,
    # runtime, deadline, period, util_min, util_max
    attr = ctypes.create_string_buffer(struct.pack("<IIQiIQQQII", 56, 0, _FLAGS, 0, 0, 0, 0, 0, 0, value))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(nr, tid, attr, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def set_process_max(value):
    """Clamp every current thread of this process (new threads and children inherit it)."""
    if not 0 <= value <= 1024:
        raise ValueError("uclamp max must be 0..1024")
    n = 0
    for tid in [int(t) for t in os.listdir("/proc/self/task")]:
        try:
            _set(tid, value)
            n += 1
        except ProcessLookupError:
            pass                      # thread exited after the listing: nothing left to clamp
    return n


def current_max(pid="self"):
    """Effective uclamp.max of a process (None when the kernel does not report it)."""
    try:
        with open("/proc/%s/sched" % pid) as f:
            for line in f:
                if line.startswith("effective uclamp.max"):
                    return int(line.split(":")[1])
    except OSError:
        pass
    return None
