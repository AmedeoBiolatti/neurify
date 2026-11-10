# Neurify 0.1.0

First public release candidate of Neurify, an MIT-licensed compiler from
bounded Python functions to neural networks, with exact initialization and
optional Torch/JAX exports for subsequent training.

- Compile to MLP, residual MLP, GLU, or transformer architectures.
- Call the network with the original function's arguments and return structure.
- Mark code scopes and inspect their actual shared activations.
- Choose each scope's architecture, extra width/depth, or a zero-initialized
  residual correction using `nf.Block` configurations.
- Select trainable Torch blocks with `model.train_only(...)`, keeping weight
  freezing separate from gradient-flow controls.
- Inspect structure, total/trainable parameter counts, and current policies.
- Validate malformed native API inputs with Python exceptions.

Release wheels are built for CPython 3.10+ on Linux x86-64/ARM64,
macOS Intel/Apple Silicon, and Windows x86-64. Install a downloaded wheel with
`python -m pip install /path/to/neurify-0.1.0-...whl`.
Once the PyPI publishing job succeeds, install with `python -m pip install
neurify` and optional `neurify[torch]` or `neurify[jax]` extras.

This is experimental software. Exactness depends on declared domains,
architecture, and numeric precision; inspect `net.report` and use
`nf.verify(net)`. Integer gadgets may have flat gradients. Code is parsed
from source, so define compiled functions in a Python file.

The release workflow gates publication on Rust/Python/backend CI,
installation tests for every platform wheel, Python 3.10/3.14 compatibility,
source completeness, version consistency, and distribution metadata checks.
