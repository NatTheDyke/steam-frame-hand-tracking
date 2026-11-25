"""The four repositories' declared versions agree, checked on what is installed."""
from importlib.metadata import entry_points, requires, version
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIBS = ("framenpu", "framenpu-cdsp", "framehands")


def pins():
    return dict(x.split("==") for x in (ROOT / "headset/PINS").read_text().split() if "==" in x)


class Versions(unittest.TestCase):
    def test_the_installed_libraries_accept_each_other(self):
        from packaging.requirements import Requirement
        for lib in LIBS:
            for r in map(Requirement, requires(lib) or ()):
                if r.name.lower() in LIBS and not (r.marker and not r.marker.evaluate({"extra": ""})):
                    self.assertTrue(r.specifier.contains(version(r.name), prereleases=True), "%s needs %s, %s is installed" % (
                        lib, r, version(r.name)))

    def test_the_pinned_set_is_complete(self):
        self.assertEqual(set(pins()), set(LIBS))

    def test_the_workflow_installs_the_pins(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        for k, v in pins().items():
            self.assertRegex(ci, r"%s==%s\b" % (re.escape(k), re.escape(v)))

    def test_the_device_backend_is_found_and_its_abi_is_loaded(self):
        from framenpu.contracts import abi_compatible
        self.assertIn("cdsp", {e.name for e in entry_points(group="framenpu.devices")})
        from framenpu_cdsp.plugin import CDSPPlugin
        self.assertTrue(abi_compatible(CDSPPlugin.abi), CDSPPlugin.abi)

    def test_one_service_module_serves_and_calls(self):
        from framenpu import service
        import handtrack.npuservice as alias
        self.assertIs(alias.Server, service.Server)

    def test_the_bridge_packs_what_the_driver_reads(self):
        """pcdriver/src/framehands_proto.h against framehands.palm's packing"""
        import struct
        from framehands import palm
        h = (ROOT / "pcdriver/src/framehands_proto.h").read_text()
        self.assertEqual(int(re.search(r"FH_VERSION (\d+)u", h)[1]), max(palm.FH_VERSIONS))
        self.assertEqual(int(re.search(r"FH_PORT (\d+)", h)[1]), palm.FH_PORT)
        self.assertEqual(int(re.search(r"FH_MAGIC (0x[0-9a-f]+)", h)[1], 16), palm.FH_MAGIC)
        body = re.search(r"struct FhHand \{(.*?)\};", h, re.S)[1]
        n = {"uint32_t": "I", "float": "f", "uint64_t": "Q"}
        fmt = ""
        for line in body.splitlines():
            m = re.match(r"\s*(\w+)\s+([^;]+);", line)
            if m:
                for name in m[2].split(","):
                    k = re.search(r"\[(\d+)\]", name)
                    fmt += n[m[1]] * int(k[1] if k else 1)
        self.assertEqual(struct.calcsize("<" + fmt), palm._HAND4.size)
        self.assertEqual(struct.Struct("<" + fmt).format.replace("<", ""), _flat(palm._HAND4.format))


def _flat(fmt):
    return "".join(c * int(k or 1) for k, c in re.findall(r"(\d*)([a-zA-Z])", fmt if isinstance(fmt, str) else fmt.decode()))


if __name__ == "__main__":
    unittest.main()
