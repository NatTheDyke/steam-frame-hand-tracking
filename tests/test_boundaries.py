"""Import direction for the library split (framenpu docs/CONTRACTS.md).

- hands: modules that become the framehands library; may import only each other. They reach
         models through the session API, never through framenpu or this repo's backends.
- npu:   the inference backends and NPU service, which move into framenpu; may import each
         other and framenpu, never tracking code.
- app:   the live tracker and its worker process; may import anything.

KNOWN is today's coupling across those lines; it may only shrink.
"""
import ast
import pathlib
import unittest

PKG = pathlib.Path(__file__).resolve().parents[1] / "handtrack"

LAYERS = {
    "hands": {"calib", "camsrc", "detect", "fasttrack", "framering", "mercdet", "mercsched",
              "merctrack", "mercury_kp", "palm", "source", "stereo3d"},
    "npu": {"inference", "npuservice"},
    "app": {"__init__", "live", "mpworker", "uclamp", "watchdog"},
}
ALLOWED = {"hands": {"hands"}, "npu": {"npu", "framenpu"}, "app": {"hands", "npu", "app", "framenpu"}}

KNOWN = {
    # mercury_kp builds its own backend and re-exports crops_per_call for the worker
    ("mercury_kp", "inference"),
}


def layer_of(module):
    if module == "framenpu":
        return "framenpu"
    for name, members in LAYERS.items():
        if module in members:
            return name
    raise KeyError(module)


def imports(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        names = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] == "handtrack":
                names = [parts[1]] if len(parts) > 1 else [a.name for a in node.names]
            elif parts[0] == "framenpu":
                names = ["framenpu"]
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            names = [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "handtrack" and len(parts) > 1:
                    names.append(parts[1])
                elif parts[0] == "framenpu":
                    names.append("framenpu")
        found.update(names)
    return found


def crossings():
    out = set()
    for path in sorted(PKG.glob("*.py")):
        src = path.stem
        for dst in imports(path) - {src}:
            if layer_of(dst) not in ALLOWED[layer_of(src)]:
                out.add((src, dst))
    return out


class BoundaryTests(unittest.TestCase):
    def test_every_module_has_a_layer(self):
        modules = {p.stem for p in PKG.glob("*.py")}
        assigned = set().union(*LAYERS.values())
        self.assertEqual(modules - assigned, set(), "assign new modules to a layer")
        self.assertEqual(assigned - modules, set(), "drop removed modules from LAYERS")

    def test_no_new_crossings(self):
        self.assertEqual(crossings() - KNOWN, set())

    def test_known_crossings_still_occur(self):
        self.assertEqual(KNOWN - crossings(), set(), "a crossing was removed: drop it from KNOWN")

    def test_hands_never_import_framenpu(self):
        self.assertFalse({s for s, d in crossings() if d == "framenpu"})


if __name__ == "__main__":
    unittest.main()
