"""Verify an actual public GitHub wheel download in a fresh environment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
import venv

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def select_wheel(release, version, tags):
    if release.get("tag_name") != f"v{version}":
        raise ValueError("public release tag does not match the expected version")
    ranks = {tag: index for index, tag in enumerate(tags)}
    matches = []
    for asset in release.get("assets", []):
        if not asset["name"].endswith(".whl"):
            continue
        name, candidate, _, wheel_tags = parse_wheel_filename(asset["name"])
        if canonicalize_name(name) != "neurify" or str(candidate) != version:
            continue
        positions = [ranks[tag] for tag in wheel_tags if tag in ranks]
        if positions:
            url = asset["browser_download_url"]
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.netloc != "github.com" or "/releases/download/" not in parsed.path:
                raise ValueError("wheel asset must be a public HTTPS GitHub release download")
            matches.append((min(positions), url))
    if not matches:
        raise ValueError("the GitHub release has no compatible wheel for this interpreter/platform")
    return min(matches)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-json", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    release = json.loads(args.release_json.read_text())
    url = select_wheel(release, args.expected_version, sys_tags())
    with tempfile.TemporaryDirectory(prefix="neurify-github-install-") as directory:
        root = Path(directory)
        venv.create(root, with_pip=True)
        python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", url], check=True)
        script = Path(__file__).with_name("smoke_install.py")
        subprocess.run([str(python), "-I", str(script), "--expected-version", args.expected_version], check=True)


if __name__ == "__main__":
    main()
