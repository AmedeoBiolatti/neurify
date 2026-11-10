# neurify-core

The dependency-free Rust core of Neurify: compile a typed, bounded program
into a neural network, with an explicit report of any approximations.

```rust
use neurify_core::{compile, Graph, Kind, Options, UnOp};

let mut graph = Graph::new();
let x = graph.input("x", Kind::Real, -2.0, 2.0)?;
let y = graph.un(UnOp::Abs, x)?;
graph.output(y)?;
let (network, report) = compile(&graph, &Options::default())?;
assert!(report.exact());
assert_eq!(network.forward(&[-1.5]), vec![1.5]);
# Ok::<(), neurify_core::CompileError>(())
```

Supported architectures include ReLU MLPs, residual MLPs, gated networks,
and transformers. `Options` controls gadget selection, exactness requirements,
and numerical tolerances. `compile` returns a `CompileError` for unsupported
operations or invalid settings. Networks are exact only on their declared
domains and subject to floating-point precision; inspect the report.

From the project root, run `cargo test -p neurify-core` and
`cargo doc -p neurify-core --open`. See the root README and DESIGN.md for
architecture details and the Python API.

Licensed under the MIT License. See [LICENSE](LICENSE).
