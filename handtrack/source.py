"""Alias of framehands.source (moved to framehands in the library split; this name stays an alias during the library split)."""
import sys

from framehands import source as _module

sys.modules[__name__] = _module
