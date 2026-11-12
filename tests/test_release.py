"""Release failures must be caught before uploading native distributions.

Archive fixtures validate metadata and source handling; they are not built
wheels and never exercise the separate installation smoke check.
"""

import io
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest

tomllib = pytest.importorskip("tomllib", reason="maintainer release tools require Python 3.11+")
from scripts.check_release import (
    SDIST_REQUIRED, check_dist, check_sdist, check_source, check_wheel, extract_sdist,
)
from scripts.verify_github_install import select_wheel
from packaging.tags import Tag

ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def sdist(path, *, omit=(), extra=None):
    with tarfile.open(path, "w:gz") as archive:
        for name in SDIST_REQUIRED:
            if name not in omit:
                if name == "pyproject.toml":
                    data = (ROOT / name).read_text().replace(f'version = "{CURRENT_VERSION}"', 'version = "0.1.0"', 1).encode()
                    entry = tarfile.TarInfo(f"neurify-0.1.0/{name}")
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
                else:
                    archive.add(ROOT / name, arcname=f"neurify-0.1.0/{name}", recursive=False)
        if extra is not None:
            entry = tarfile.TarInfo(extra)
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
    return path


def wheel(path, *, license="MIT", version="0.1.0", include_native=True, include_classical=True):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("neurify-0.1.0.dist-info/METADATA",
                         f"Metadata-Version: 2.4\nName: neurify\nVersion: {version}\n"
                         f"License-Expression: {license}\nRequires-Python: >=3.10\n")
        archive.writestr("neurify-0.1.0.dist-info/licenses/LICENSE", "MIT License\n")
        archive.writestr("neurify/blocks.py", "")
        archive.writestr("neurify/residual_block.py", "")
        if include_classical:
            archive.writestr("neurify/transformer.py", "")
        if include_native:
            archive.writestr("neurify/_core.abi3.so", b"metadata fixture, not a native binary")
    return path


def test_source_versions_and_tag_must_agree():
    assert check_source(ROOT, f"v{CURRENT_VERSION}") == CURRENT_VERSION
    with pytest.raises(ValueError, match="release tag"):
        check_source(ROOT, "v99999.0.0")


def test_exported_version_mismatch_is_caught(tmp_path):
    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy2(ROOT / "Cargo.toml", tmp_path / "Cargo.toml")
    (tmp_path / "python/neurify").mkdir(parents=True)
    (tmp_path / "python/neurify/__init__.py").write_text('__version__ = "0.0.0"\n')
    with pytest.raises(ValueError, match="__version__"):
        check_source(tmp_path)


def test_complete_sdist_extracts_to_a_new_directory(tmp_path):
    path = sdist(tmp_path / "source.tar.gz")
    assert check_sdist(path, "0.1.0") == "neurify-0.1.0"
    destination = tmp_path / "extracted"
    extract_sdist(path, destination, "0.1.0")
    assert (destination / "python/neurify/blocks.py").read_bytes() == (ROOT / "python/neurify/blocks.py").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        extract_sdist(path, destination, "0.1.0")


def test_missing_scope_interface_source_is_rejected(tmp_path):
    path = sdist(tmp_path / "source.tar.gz", omit=("python/neurify/blocks.py",))
    with pytest.raises(ValueError, match="incomplete source"):
        check_sdist(path, "0.1.0")


@pytest.mark.parametrize("name", ["../escape", "neurify-0.1.0/../../escape", "neurify-0.1.0/C:/escape"])
def test_archive_traversal_is_rejected_before_extraction(tmp_path, name):
    path = sdist(tmp_path / "source.tar.gz", extra=name)
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="unsafe source"):
        extract_sdist(path, destination, "0.1.0")
    assert not destination.exists()


def test_old_native_binary_is_not_allowed_in_sdist(tmp_path):
    path = sdist(tmp_path / "source.tar.gz", extra="neurify-0.1.0/python/neurify/_core.abi3.so")
    with pytest.raises(ValueError, match="binary/cache"):
        check_sdist(path, "0.1.0")


def test_source_symlinks_are_rejected(tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("neurify-0.1.0/link")
        entry.type, entry.linkname = tarfile.SYMTYPE, "/tmp/outside"
        archive.addfile(entry)
    with pytest.raises(ValueError, match="links/devices"):
        check_sdist(path, "0.1.0")


def test_portable_abi3_wheel_metadata(tmp_path):
    path = wheel(tmp_path / "neurify-0.1.0-cp310-abi3-manylinux_2_17_x86_64.whl")
    assert check_wheel(path, "0.1.0") == {"manylinux_2_17_x86_64"}


@pytest.mark.parametrize("filename", [
    "neurify-0.1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
    "neurify-0.1.0-cp310-abi3-linux_x86_64.whl",
    "neurify-0.1.0-py3-none-any.whl",
    "neurify-0.2.0-cp310-abi3-win_amd64.whl",
])
def test_incompatible_or_nonportable_wheel_tags_are_rejected(tmp_path, filename):
    with pytest.raises(ValueError):
        check_wheel(wheel(tmp_path / filename), "0.1.0")


@pytest.mark.parametrize("kwargs", [{"license": "Apache-2.0"}, {"version": "0.2.0"},
                                   {"include_native": False}, {"include_classical": False}])
def test_invalid_wheel_contents_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        check_wheel(wheel(tmp_path / "neurify-0.1.0-cp310-abi3-win_amd64.whl", **kwargs), "0.1.0")


def test_release_requires_every_platform_and_writes_checksums(tmp_path):
    platforms = ["manylinux_2_17_x86_64", "manylinux_2_17_aarch64", "macosx_10_12_x86_64", "macosx_11_0_arm64", "win_amd64"]
    sdist(tmp_path / "neurify-0.1.0.tar.gz")
    for platform in platforms[:-1]:
        wheel(tmp_path / f"neurify-0.1.0-cp310-abi3-{platform}.whl")
    with pytest.raises(ValueError, match="five platform"):
        check_dist(tmp_path, "0.1.0")
    wheel(tmp_path / f"neurify-0.1.0-cp310-abi3-{platforms[-1]}.whl")
    check_dist(tmp_path, "0.1.0", checksums=True)
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 6


def test_duplicate_platform_cannot_hide_a_missing_platform(tmp_path):
    sdist(tmp_path / "neurify-0.1.0.tar.gz")
    for platform in ("manylinux_2_17_x86_64", "manylinux2014_x86_64", "macosx_10_12_x86_64", "macosx_11_0_arm64", "win_amd64"):
        wheel(tmp_path / f"neurify-0.1.0-cp310-abi3-{platform}.whl")
    with pytest.raises(ValueError, match="missing or duplicate"):
        check_dist(tmp_path, "0.1.0")


def public_assets():
    return {"tag_name": "v0.1.0", "assets": [
        {"name": f"neurify-0.1.0-cp310-abi3-{platform}.whl",
         "browser_download_url": f"https://github.com/AmedeoBiolatti/neurify/releases/download/v0.1.0/neurify-0.1.0-cp310-abi3-{platform}.whl"}
        for platform in ("win_amd64", "manylinux_2_17_x86_64", "macosx_11_0_arm64")
    ]}


def test_public_install_selects_the_compatible_wheel():
    tags = [Tag("cp312", "cp312", "manylinux_2_17_x86_64"), Tag("cp310", "abi3", "manylinux_2_17_x86_64")]
    assert select_wheel(public_assets(), "0.1.0", tags).endswith("manylinux_2_17_x86_64.whl")


def test_public_install_does_not_fall_back_to_source_on_unsupported_platform():
    with pytest.raises(ValueError, match="no compatible wheel"):
        select_wheel(public_assets(), "0.1.0", [Tag("cp310", "abi3", "win_arm64")])


def test_public_install_rejects_a_wrong_release_or_non_github_asset():
    tags = [Tag("cp310", "abi3", "win_amd64")]
    with pytest.raises(ValueError, match="tag"):
        select_wheel(public_assets(), "0.2.0", tags)
    release = public_assets()
    release["assets"][0]["browser_download_url"] = "http://example.org/wheel.whl"
    with pytest.raises(ValueError, match="HTTPS GitHub"):
        select_wheel(release, "0.1.0", tags)
