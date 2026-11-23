"""Alias of framehands.merctrack (moved to framehands in the library split; this name stays an alias during the library split)."""
import sys

from framehands import merctrack as _module

sys.modules[__name__] = _module
