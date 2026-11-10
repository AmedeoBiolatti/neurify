# neurify-py

PyO3 bindings exposing Neurify's compiler and symbolic Python frontend as
`neurify._core`. Python applications should use the public `neurify` package,
which handles input domains, structured values, verification, and exports.

Build the complete Python package from the project root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
python -m pytest tests crates/neurify-py/tests
```

The extension uses Python's stable ABI with a Python 3.10 minimum. Building
from source requires Rust and a C linker. Optional Torch and JAX exports need
the corresponding Python dependencies.

Flat batches at the native API must contain complete rows. Invalid batch
shapes, node IDs, and compiler settings raise `CompileError`.

The Python package also provides `with nf.scope(...)` inside a function and
named `region` / `pipeline` composition for training controls and observation
of intermediate values. See CONTROLS.md in the project root for Torch and
JAX examples.

High-level networks preserve the original function's arguments and output
structure. `net.inspect(...)` returns `Evaluation(output, exposed)` in one
run; trained exports provide function adapters alongside their raw tensor API.

Scoped exposure selects shared, materialized activations consumed downstream;
it adds no trainable readout. Dead locals without a retained boundary are
rejected. Live Torch probes support gradients from the main output loss.

Licensed under the MIT License. See [LICENSE](LICENSE).
