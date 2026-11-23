"""Camera-frame pairs handed to the Mercury worker through shared memory instead of pickled copies.

A Mercury job used to carry both published camera frames (2 x 1056 x 1024 bytes) through the process
pool's pipe: pickled by a feeder thread in the tracker, read and unpickled in the worker.  On the
headset that is 1.56 ms per job round trip and 0.9 + 1.15 ms of CPU (tracker + worker); through shared
memory it is 0.45 ms and 0.39 + 0.10 ms (tools-dsp ipc_bench, 2026-11-22).

The tracker owns a FrameRing of SLOTS frame pairs.  put() copies one capture's pair into a free slot
and returns a FrameRef, a small picklable handle the job carries instead of the frames; the slot stays
reserved until release() (the job's future is done: the worker no longer reads it).  With no free slot
put() returns None and the caller sends the frames as before.  The worker turns a FrameRef back into
{slot: frame} with frames(), views on the shared memory (it keeps none after a job).
"""
import threading

import numpy as np

SLOTS = 4                 # pairs in the ring: one tracking job and one discovery job in flight, plus spare


class FrameRef:
    """Picklable handle to one frame pair of a FrameRing."""
    __slots__ = ("name", "k", "shape", "cams")

    def __init__(self, name, k, shape, cams):
        self.name, self.k, self.shape, self.cams = name, k, tuple(shape), tuple(cams)

    def __getstate__(self):
        return (self.name, self.k, self.shape, self.cams)

    def __setstate__(self, s):
        self.name, self.k, self.shape, self.cams = s


class FrameRing:
    def __init__(self, shape, slots=SLOTS):
        from multiprocessing import shared_memory
        self.shape = tuple(shape)
        self.slots = slots
        self.shm = shared_memory.SharedMemory(create=True, size=slots * 2 * int(np.prod(self.shape)))
        self.buf = np.ndarray((slots, 2) + self.shape, np.uint8, self.shm.buf)
        self.free = list(range(slots))
        self.lock = threading.Lock()
        self.stats = {"put": 0, "full": 0}

    def put(self, imgs):
        """imgs {0: frame, 1: frame} (uint8, self.shape) -> FrameRef, or None (no free slot / other shape)."""
        if any(np.shape(imgs.get(c)) != self.shape for c in (0, 1)):
            return None
        with self.lock:
            if not self.free:
                self.stats["full"] += 1
                return None
            k = self.free.pop(0)
        self.buf[k, 0] = imgs[0]
        self.buf[k, 1] = imgs[1]
        self.stats["put"] += 1
        return FrameRef(self.shm.name, k, self.shape, (0, 1))

    def release(self, ref):
        if ref is None:
            return
        with self.lock:
            if ref.k not in self.free:
                self.free.append(ref.k)

    def release_when_done(self, ref, future):
        """Free ref's slot once the job reading it has finished (any outcome)."""
        if ref is not None:
            future.add_done_callback(lambda _f: self.release(ref))

    def close(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except (OSError, FileNotFoundError):
            pass


_open = {}                # worker side: name -> SharedMemory, opened once


def frames(imgs):
    """Worker side: a FrameRef -> {cam: frame view}; anything else (a frames dict) is returned as is."""
    if not isinstance(imgs, FrameRef):
        return imgs
    from multiprocessing import shared_memory
    shm = _open.get(imgs.name)
    if shm is None:
        # the tracker owns (and unlinks) the segment; keep Python's resource tracker in this process
        # from unlinking it too when the worker exits
        shm = shared_memory.SharedMemory(name=imgs.name, track=False)
        _open[imgs.name] = shm
    pair = 2 * int(np.prod(imgs.shape))
    ring = np.ndarray((shm.size // pair, 2) + imgs.shape, np.uint8, shm.buf)
    return {c: ring[imgs.k, c] for c in imgs.cams}
