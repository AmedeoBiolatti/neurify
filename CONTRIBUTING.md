# Contributing to Neurify

Neurify compiles bounded Python programs to neural networks. Changes should
preserve the declared-domain semantics and report approximations explicitly.
Contributions are provided under the project's MIT license.

## Local setup

Use Python 3.10 or newer, a recent stable Rust toolchain, and a C linker.
From the project root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
For optional backends, install `'.[test,torch,jax]'` instead.

After changing Rust code, rebuild the extension before running Python tests:

```sh
python -m pip install -e '.[test]'
cargo test -p neurify-core -p neurify-frontend --locked
python -m pytest
```

Use a focused regression test for compiler or runtime fixes. Exercise edge
cases and compare the network with the IR or Python reference; exactness on
one sample does not establish exactness on the declared domain. Record the
architecture, domain, precision, and verification settings with bug reports.

For named regions and observation changes, run `python -m pytest
tests/test_regions.py tests/test_scopes.py tests/test_interface.py
tests/test_activations.py`. Install the optional Torch and JAX dependencies to
exercise parameter freezing, stopped gradients, live observations, and
checkpoint behavior. See CONTROLS.md for the API and its boundary contracts.
Activation changes must preserve the parameterization when exposure is
toggled, and main-loss gradients must reach live probes themselves. A probe
must not create a separate prediction head for a source variable.

Core code has no external dependencies. If registry access is unavailable,
its unit and integration tests can be run directly from the project root:

```sh
rustc --edition=2021 --test crates/neurify-core/src/lib.rs -o /tmp/neurify-core-tests
/tmp/neurify-core-tests
rustc --edition=2021 --crate-name neurify_core --crate-type rlib crates/neurify-core/src/lib.rs -o /tmp/libneurify_core.rlib
rustc --edition=2021 --test crates/neurify-core/tests/validation.rs --extern neurify_core=/tmp/libneurify_core.rlib -o /tmp/neurify-validation-tests
/tmp/neurify-validation-tests
```

## Preparing a release

1. Keep the Python version, workspace version, `__version__`, and local Rust
   dependency versions aligned. Update CHANGELOG.md.
2. Run Rust and Python tests against a freshly built extension. Run optional
   backend tests when shipping changes to those exports. CI exercises base
   Python 3.10 and 3.14, plus Torch and JAX on Python 3.12.
3. Build and inspect the Python distributions:

   ```sh
   python -m pip install 'maturin>=1.9.3,<2' twine
   maturin build --release --locked --out dist
   maturin sdist --out dist
   python -m twine check dist/*
   ```

4. Confirm the wheel and source distribution include LICENSE and declare
   `License-Expression: MIT`. Test installation from the source distribution
   in a clean environment. Local wheels target the build platform; publishing
   portable wheels requires platform-specific builds.
5. Publish only to a repository or registry you control. Rust crates must be
   released in dependency order: neurify-core, neurify-frontend, neurify-py.
   Packaging dependent crates against a registry requires their dependencies
   to be published there first.

The Python build backend starts at Maturin 1.9.3 to include the fixes for
modern license metadata and source-distribution license files; see the
[Maturin changelog](https://www.maturin.rs/changelog).
