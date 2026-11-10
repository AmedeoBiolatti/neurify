"""Check release versions, source completeness, and portable wheel contents.

Maintainer tool: Python 3.11+ and packaging. Does not import the extension.
"""

from __future__ import annotations

import argparse
import ast
from email import message_from_bytes
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tomllib
import zipfile

from packaging.utils import canonicalize_name, parse_wheel_filename


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(root, tag=None):
    root = Path(root)
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    require(project["name"] == "neurify", "unexpected project name")
    require(project["license"] == "MIT", "project license must be MIT")
    require(project["license-files"] == ["LICENSE"], "distribution must include LICENSE")
    require(project["requires-python"] == ">=3.10", "Python requirement and abi3 baseline must agree")
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]
    require(workspace["version"] == version, "Cargo and Python versions differ")
    require(workspace["license"] == "MIT", "workspace license must be MIT")
    tree = ast.parse((root / "python/neurify/__init__.py").read_text())
    exported = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets))
    require(exported == version, "neurify.__version__ differs from distribution version")
    locked = {p["name"]: p["version"] for p in tomllib.loads((root / "Cargo.lock").read_text())["package"]}
    license_text = (root / "LICENSE").read_bytes()
    require(license_text.startswith(b"MIT License"), "unexpected license text")
    for name in ("neurify-core", "neurify-frontend", "neurify-py"):
        crate = root / "crates" / name
        metadata = tomllib.loads((crate / "Cargo.toml").read_text())
        require(metadata["package"]["version"] == {"workspace": True}, f"{name} must inherit workspace version")
        require(metadata["package"]["license"] == "MIT", f"{name} license mismatch")
        require(locked[name] == version, f"{name} lockfile version mismatch")
        require((crate / "LICENSE").read_bytes() == license_text, f"{name} license text mismatch")
        for dep in ("neurify-core", "neurify-frontend"):
            if dep in metadata.get("dependencies", {}):
                require(metadata["dependencies"][dep]["version"] == version, f"{name}: {dep} version mismatch")
    binding = tomllib.loads((root / "crates/neurify-py/Cargo.toml").read_text())
    require("abi3-py310" in binding["dependencies"]["pyo3"]["features"], "wheel ABI baseline must be CPython 3.10")
    if tag:
        require(tag == f"v{version}", f"release tag {tag!r} must be v{version}")
    return version


SDIST_REQUIRED = (
    "pyproject.toml", "Cargo.toml", "Cargo.lock", "LICENSE", "README.md",
    "BLOCKS.md", "CONTROLS.md", "SCOPE_TRAINING.md", "RELEASING.md", "RELEASE_NOTES.md",
    "python/neurify/blocks.py", "python/neurify/residual_block.py",
    "python/neurify/export_residual_block_torch.py", "python/neurify/export_residual_block_jax.py",
    "tests/test_blocks.py", "tests/test_scope_training.py", "examples/blocks.py",
    "scripts/check_release.py", "scripts/smoke_install.py",
    "crates/neurify-core/Cargo.toml", "crates/neurify-core/src/lib.rs",
    "crates/neurify-frontend/Cargo.toml", "crates/neurify-frontend/src/lib.rs",
    "crates/neurify-py/Cargo.toml", "crates/neurify-py/src/lib.rs",
    "crates/neurify-py/tests/test_api_validation.py",
)


def check_sdist(path, version):
    prefix = f"neurify-{version}"
    with tarfile.open(path) as archive:
        names = set()
        for member in archive:
            parts = PurePosixPath(member.name).parts
            require(parts and parts[0] == prefix and ".." not in parts
                    and "\\" not in member.name and ":" not in member.name,
                    f"unsafe source member: {member.name}")
            require(member.isdir() or member.isfile(), f"source links/devices are not supported: {member.name}")
            rel = "/".join(parts[1:])
            require(not rel.endswith((".so", ".pyd", ".dll", ".pyc")), f"binary/cache included in source: {rel}")
            if member.isfile():
                require(rel not in names, f"duplicate source file: {rel}")
                names.add(rel)
        missing = set(SDIST_REQUIRED) - names
        require(not missing, f"incomplete source distribution: {sorted(missing)}")
        metadata = tomllib.loads(archive.extractfile(f"{prefix}/pyproject.toml").read().decode())
        require(metadata["project"]["version"] == version, "source distribution version mismatch")
    return prefix


def extract_sdist(path, destination, version):
    check_sdist(path, version)
    destination = Path(destination)
    require(not destination.exists(), f"extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isfile():
                parts = PurePosixPath(member.name).parts[1:]
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def check_wheel(path, version):
    name, wheel_version, _, tags = parse_wheel_filename(Path(path).name)
    require(canonicalize_name(name) == "neurify" and str(wheel_version) == version, "wheel filename/version mismatch")
    require(all(t.interpreter == "cp310" and t.abi == "abi3" for t in tags), "wheel must use cp310-abi3")
    platforms = {t.platform for t in tags}
    require(not any(p.startswith("linux_") or p == "any" for p in platforms), "wheel must use a portable platform tag")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), "duplicate wheel entries")
        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata_files) == 1, "wheel must have exactly one metadata record")
        metadata = message_from_bytes(wheel.read(metadata_files[0]))
        require(canonicalize_name(metadata["Name"]) == "neurify" and metadata["Version"] == version, "wheel metadata mismatch")
        require(metadata["License-Expression"] == "MIT", "wheel license expression mismatch")
        require(metadata["Requires-Python"] == ">=3.10", "wheel Python requirement mismatch")
        require(any(n.endswith(".dist-info/licenses/LICENSE") for n in names), "wheel license file missing")
        require("neurify/blocks.py" in names and "neurify/residual_block.py" in names, "new interface missing from wheel")
        require(any(n.startswith("neurify/_core") and n.endswith((".so", ".pyd")) for n in names), "native extension missing")
    return platforms


def check_dist(directory, version, checksums=False):
    directory = Path(directory)
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    require(len(wheels) == 5, "release requires five platform wheels")
    require(len(sources) == 1, "release requires one source distribution")
    seen = []
    for path in wheels:
        platforms = check_wheel(path, version)
        families = {
            "linux-x86_64" if p.startswith("manylinux") and p.endswith("_x86_64") else
            "linux-aarch64" if p.startswith("manylinux") and p.endswith("_aarch64") else
            "macos-x86_64" if p.startswith("macosx_") and p.endswith("_x86_64") else
            "macos-aarch64" if p.startswith("macosx_") and p.endswith("_arm64") else
            "windows-amd64" if p == "win_amd64" else "unsupported"
            for p in platforms
        }
        require(len(families) == 1 and "unsupported" not in families, f"unexpected wheel platforms: {platforms}")
        seen.append(next(iter(families)))
    require(len(set(seen)) == 5, "missing or duplicate platform wheel")
    check_sdist(sources[0], version)
    if checksums:
        paths = sorted(wheels + sources)
        (directory / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag")
    parser.add_argument("--github-output", action="store_true")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    tag = args.tag or (os.environ.get("GITHUB_REF_NAME") if os.environ.get("GITHUB_REF_TYPE") == "tag" else None)
    version = check_source(args.root, tag)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"version={version}\n")
    if args.dist:
        check_dist(args.dist, version, args.checksums)
    if args.sdist or args.extract:
        paths = list((args.sdist or args.extract).glob("*.tar.gz"))
        require(len(paths) == 1, "expected one source distribution")
        if args.extract:
            require(args.destination is not None, "--extract requires --destination")
            extract_sdist(paths[0], args.destination, version)
        else:
            check_sdist(paths[0], version)
    print(f"Neurify {version}: release checks passed")


if __name__ == "__main__":
    main()
