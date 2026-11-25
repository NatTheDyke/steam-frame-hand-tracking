#!/usr/bin/env python3
"""A virtual environment with one set of the three libraries, for the integration tests.

    python3 integration/setup_env.py DIR [--framenpu REF] [--framenpu-cdsp REF] [--framehands REF] [--models MODELS]

REF is a commit, branch or tag of the GitHub repository, or a local checkout's path (installed as it stands,
uncommitted changes included: files git knows or could add). Default: the release headset/PINS names.
DIR must not exist. It gets DIR/venv, DIR/src/<library> and DIR/env (the FI_* variables, to source or
append to $GITHUB_ENV). --models fetches the weights the tests open (by hash, framenpu's tools/fetch_models.py)."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LIBS = ("framenpu", "framenpu-cdsp", "framehands")


def run(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch(lib, ref, to):
    """The sources at ref (a path, a commit, a branch or a tag) in the new directory `to`; what they are."""
    to.mkdir(parents=True)
    if Path(ref).is_dir():
        files = subprocess.run(["git", "-C", ref, "ls-files", "-z", "-co", "--exclude-standard"], check=True, capture_output=True).stdout
        listed = [f for f in files.split(b"\0") if f and (Path(ref) / f.decode()).exists()]
        tar = subprocess.run(["tar", "c", "-C", ref, "--null", "-T", "-"], input=b"\0".join(listed), check=True, capture_output=True).stdout
        run("tar", "x", "-C", to, input=tar)
        return "%s (working tree of %s)" % (subprocess.run(["git", "-C", ref, "rev-parse", "--short", "HEAD"], check=True,
                                                          capture_output=True, text=True).stdout.strip(), ref)
    run("git", "init", "-q", to)
    run("git", "-C", to, "fetch", "-q", "--depth", "1", "https://github.com/NatTheDyke/%s" % lib, ref)
    run("git", "-C", to, "checkout", "-q", "FETCH_HEAD")
    return subprocess.run(["git", "-C", str(to), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path)
    for lib in LIBS:
        ap.add_argument("--" + lib, metavar="REF", default="")
    ap.add_argument("--models", type=Path)
    a = ap.parse_args()
    if a.dir.exists():
        ap.error("%s exists" % a.dir)
    d = a.dir.resolve()
    pins = dict(x.split("==") for x in (ROOT / "headset/PINS").read_text().split() if "==" in x)
    for lib in LIBS:
        ref = getattr(a, lib.replace("-", "_")) or "v" + pins[lib]
        print("%-14s %s -> %s" % (lib, ref, fetch(lib, ref, d / "src" / lib)), flush=True)
    run(sys.executable, "-m", "venv", d / "venv")
    py = d / "venv/bin/python"
    run(py, "-m", "pip", "-q", "install", "%s[cpu]" % (d / "src/framenpu"), d / "src/framenpu-cdsp", d / "src/framehands", "packaging")
    env = ["FI_CDSP_TESTS=%s" % (d / "src/framenpu-cdsp/tests")]
    if a.models:
        run(py, d / "src/framenpu/tools/fetch_models.py", a.models, "--only", "mercury-ft7", "mercury-detector-160")
        env.append("FI_MODELS=%s" % a.models.resolve())
    (d / "env").write_text("".join("%s\n" % e for e in env))


if __name__ == "__main__":
    main()
