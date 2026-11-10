# neurify-frontend

Neurify's Python-source frontend, written in Rust. It parses functions with
`rustpython-parser`, symbolically interprets them over declared input domains,
and produces the bounded dataflow graph consumed by `neurify-core`.

The `trace` API accepts the source, parameter specifications, a `Host` for
resolving external names and callables, and `TraceOptions` for bounded loop
unrolling and tabulation. It returns the graph, output structure, and warnings.
Unsupported or unbounded constructs return a `TraceError`.

The frontend interprets supplied programs and may call the host during
compilation. Use it with trusted source and host implementations.

From the project root, run `cargo test -p neurify-frontend` and
`cargo doc -p neurify-frontend --open`. See the root README for supported
Python syntax and DESIGN.md for control-flow semantics.

Licensed under the MIT License. See [LICENSE](LICENSE).
