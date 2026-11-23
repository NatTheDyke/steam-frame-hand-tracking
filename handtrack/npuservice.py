"""Alias of framenpu.service (moved to FrameNPU in the library split; this name stays an alias during
the library split). Run as a script (the installed unit runs `python -I npuservice.py ...`), it serves."""
import sys

from framenpu import service as _module

if __name__ == "__main__":
    sys.exit(_module.main())
sys.modules[__name__] = _module
