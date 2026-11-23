"""The tracker after the library split (framenpu docs/CONTRACTS.md).

The hand-tracking modules moved to framehands and the inference backends and NPU service to
framenpu; their old handtrack names stay aliases of the same module objects (imports, test
patches and the installed `python -I npuservice.py` unit keep working). Each library tests its
own import direction. What stays here is the app: the live tracker and its worker process.
"""
import importlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PKG = ROOT / "handtrack"
ALIAS = "this name stays an alias during the library split"

MOVED = {
    "framehands": {"calib", "camsrc", "detect", "fasttrack", "framering", "mercdet", "mercsched",
                   "merctrack", "mercury_kp", "palm", "source", "stereo3d"},
    "framenpu": {"inference": "backends", "npuservice": "service"},
}
APP = {"__init__", "live", "mpworker", "uclamp", "watchdog"}


def moved():
    out = {m: "framehands." + m for m in MOVED["framehands"]}
    out.update({old: "framenpu." + new for old, new in MOVED["framenpu"].items()})
    return out


class SplitTests(unittest.TestCase):
    def test_every_module_is_the_app_or_an_alias(self):
        modules = {p.stem for p in PKG.glob("*.py")}
        self.assertEqual(modules, APP | set(moved()))
        for name in moved():
            self.assertIn(ALIAS, (PKG / f"{name}.py").read_text(), name)
        for name in APP:
            self.assertNotIn(ALIAS, (PKG / f"{name}.py").read_text(), name)

    def test_old_names_are_the_moved_modules(self):
        for old, new in moved().items():
            if old == "detect":  # the legacy MediaPipe detector needs mediapipe
                continue
            with self.subTest(old=old):
                self.assertIs(importlib.import_module("handtrack." + old), importlib.import_module(new))


if __name__ == "__main__":
    unittest.main()
