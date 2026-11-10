# Public release and installation

The intended first release is **0.1.0**, under MIT, at
`AmedeoBiolatti/neurify`, with the PyPI distribution name `neurify`.
The public repository and PyPI project have **not been created or published
from this session**. PyPI name availability could not be confirmed. Check
that name in the intended PyPI account before enabling publication.

## What users will install

```sh
python -m pip install neurify
python -m pip install 'neurify[torch]'  # Optional training export.
```

The release builds five `cp310-abi3` wheels, covering regular CPython
3.10–3.14 without requiring a Rust compiler on the user's machine:

| OS | Architectures | Wheel baseline |
| --- | --- | --- |
| Linux | x86-64, ARM64 | manylinux2014 / glibc 2.17+ |
| macOS | Intel, Apple Silicon | platform tags selected by maturin |
| Windows | x86-64 | win_amd64 |

Other platforms can build the source distribution with Rust and a linker.
PyPy, free-threaded CPython, musl Linux, and Windows ARM64 do not have
prebuilt wheels in this first release matrix.

Before PyPI setup is complete, the public GitHub release can still provide
installable wheels. Users download the wheel matching their platform and
run `python -m pip install /path/to/downloaded.whl`. The full source bundle
supplied in `crates/release` is a project export; it is not a built wheel or
the maturin-generated PyPI source distribution.

## First repository setup

The supplied `neurify-0.1.0.git.bundle` contains a local `main` branch with
the complete release candidate already committed. On a connected machine:

```sh
git clone -o source /path/to/neurify-0.1.0.git.bundle neurify
cd neurify
gh repo create AmedeoBiolatti/neurify --public --source=. --remote=origin --push
gh workflow run release.yml --ref main
```

This creates the public repository and starts a build-only trial run.
The bundle itself has not been pushed or published. Its commit uses the
intended owner's public GitHub no-reply author identity.

Alternatively, initialize the complete source archive manually:

Use the complete updated source archive, rather than the older parent
checkout. Extract it into a new directory on a machine with working GitHub
and package-registry access. From the extracted `neurify-0.1.0` directory:

```sh
python3 scripts/check_release.py --tag v0.1.0
git init -b main
git add .
git commit -m 'Prepare Neurify 0.1.0 public release'
gh repo create AmedeoBiolatti/neurify --public --source=. --remote=origin --push
gh workflow run release.yml --ref main
gh run list --workflow release.yml
```

The check script needs Python 3.11+ and `packaging`; install it into a local
maintainer virtual environment if needed. Configure your Git author identity
before committing. If a Neurify repository already exists by the time these
commands are run, inspect it and use its normal contribution/release process
instead of creating or replacing it.

The manual workflow on `main` builds and tests everything but publishes
nothing. Inspect the run, fix failures, and require a successful run before
tagging. A connected GitHub CLI can watch a selected run with
`gh run watch RUN_ID --exit-status`.

## PyPI account setup

Create or sign in to the PyPI account that will own Neurify. In
[account publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub publisher with these exact fields:

| Field | Value |
| --- | --- |
| PyPI project name | `neurify` |
| Owner | `AmedeoBiolatti` |
| Repository | `neurify` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

If the account already owns that PyPI project, add the trusted publisher in
the project's publishing settings instead. A pending publisher does not
reserve the name. If another account owns `neurify`, resolve ownership or
choose a different distribution name before proceeding. Do not upload a
package to an unrelated name without updating and checking the metadata.

PyPI's [pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
describes the account setup. Authentication uses GitHub-issued short-lived
tokens; no PyPI password or API token belongs in the repository.

Once the trusted publisher is configured, enable the publishing job:

```sh
gh variable set PYPI_PUBLISH_ENABLED --body true --repo AmedeoBiolatti/neurify
```

The workflow uses the GitHub environment `pypi`; create/configure it in
repository settings if the repository's environment policy requires that.
Leaving the variable unset produces a GitHub release with downloadable
wheels while skipping PyPI. Publication can later be run manually on the
same tag after enabling PyPI; an existing GitHub release is retained.

## Publish the tested commit

After the manual build passes and account setup is complete:

```sh
git tag -a v0.1.0 -m 'Neurify 0.1.0'
git push origin v0.1.0
```

`.github/workflows/release.yml` performs the following steps:

1. Run the reusable Rust, Python, backend, and distribution CI.
2. Check Python/Cargo/exported versions, licenses, and the release tag.
3. Produce a source distribution, checking its required sources and docs.
4. Build each wheel **from that source distribution**, then install the
   actual wheel and run isolated smoke checks plus Python/native tests.
5. Install the Linux wheel on Python 3.10 and 3.14 and run the same checks.
6. Require five distinct platform wheels and one source distribution, run
   strict metadata checks, and write `SHA256SUMS`.
7. Attach the tested files to the public GitHub release for the existing tag.
8. If enabled, publish to PyPI through the configured trusted publisher,
   then verify a fresh wheel-only install from the public PyPI index.

The publishing jobs run only for version tags. A version mismatch fails
before building. Build/test jobs have read-only repository access; the
GitHub upload job alone gets release-write permission, and the PyPI job
alone gets OIDC token permission. PyPI publishing stays in a top-level
workflow, separate from the reusable CI.

## Subsequent versions

Update the version in `pyproject.toml`, the workspace package in
`Cargo.toml`, `python/neurify/__init__.py`, internal crate dependency versions,
and the corresponding local-package entries in `Cargo.lock`. Update release
notes, run `python scripts/check_release.py --tag vNEW_VERSION`, and complete
the build/test process before pushing that version tag. Published package
versions cannot be replaced; use a new version for changed distributions.

## Current validation limits

The current local Python/Torch run passed **472 tests with 18 skipped**,
including 19 release-check cases. The checks use synthetic archive metadata;
they do not substitute for real native builds or installed-wheel tests.
The release workflow uses maturin 1.16.0 for source and wheel builds. This session cannot
build fresh native wheels: `rustpython-parser` is absent from the local
Cargo cache and terminal DNS cannot reach the registries. Maturin and Twine
are also unavailable locally. JAX and cross-platform build validation must
run in GitHub Actions before the release can be called installable.

The GitHub connection confirms access to the intended owner's repositories
but has no repository/release creation tool. The browser's navigation API
failed before opening GitHub. No public repository, release, PyPI upload,
or installable wheel is claimed by these preparation files.
