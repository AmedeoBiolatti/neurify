# GitHub releases

Neurify is released under MIT at `AmedeoBiolatti/neurify`. PyPI is not used.

## Installation

With Rust, Git, and a C linker installed:

```sh
python -m pip install "neurify @ git+https://github.com/AmedeoBiolatti/neurify.git@v0.1.0"
python -m pip install "neurify[torch] @ git+https://github.com/AmedeoBiolatti/neurify.git@v0.1.0"
```

For installation without Rust, download the matching `.whl` file from the
[release page](https://github.com/AmedeoBiolatti/neurify/releases/tag/v0.1.0):

```sh
python -m pip install /path/to/downloaded.whl
```

The five `cp310-abi3` wheels cover regular CPython 3.10–3.14:

| OS | Architectures |
| --- | --- |
| Linux (manylinux2014 / glibc 2.17+) | x86-64, ARM64 |
| macOS | Intel, Apple Silicon |
| Windows | x86-64 |

PyPy, free-threaded CPython, musl Linux, and Windows ARM64 do not have
prebuilt wheels in this matrix. Source installation requires a Rust toolchain.

## Release process

1. Update the version in `pyproject.toml`, the workspace package in
   `Cargo.toml`, `python/neurify/__init__.py`, internal crate dependency
   versions, and the corresponding entries in `Cargo.lock`.
2. Update `RELEASE_NOTES.md` and the versioned installation commands.
3. Run `python scripts/check_release.py --tag vNEW_VERSION` with Python
   3.11+ and `packaging` installed.
4. Push the commit to `main` and require successful CI. For a complete
   build-only trial, run `gh workflow run release.yml --ref main`.
5. Tag the tested commit and push the tag:

```sh
git tag -a v0.1.0 -m 'Neurify 0.1.0'
git push origin v0.1.0
```

The release workflow runs full Rust/Python/backend CI, builds a source
distribution, builds all five wheels **from that source distribution**,
installs and tests every wheel, checks Python 3.10/3.14 compatibility, and
validates versions, licensing, source completeness, and wheel platform tags.
Only after these jobs pass does it attach the wheels, source distribution,
and `SHA256SUMS` to a GitHub release for the existing tag.

Manual runs on branches build and test without publishing. Build jobs have
read-only repository access; only the GitHub release job gets write access.
No registry credentials or publisher account setup is required.

The complete project source export and Git bundle in `crates/release` are
local backup artifacts, not prebuilt wheels or evidence of public publication.
The public GitHub repository, release assets, and Actions run provide the
publication and build status.
