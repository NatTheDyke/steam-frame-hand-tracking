"""Alias of framenpu.backends (moved to FrameNPU in the library split; this name stays an alias during the library split)."""
import sys

from framenpu import backends as _module

sys.modules[__name__] = _module
