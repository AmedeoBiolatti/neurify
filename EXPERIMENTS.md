# What the experiments contribute

Neurify compiles bounded Python programs into neural networks, preserves their
initial function, and exposes controls for later training. Training an agent,
collecting game trajectories, and managing CUDA graph caches belong to the
application that uses the compiler.

## Decisions

| Area | Decision |
| --- | --- |
| Original-function calls, exposed activations, scopes and selective training | Keep the existing public interfaces and their tests. |
| Exact extra capacity and zero-output residual corrections | Keep `Block`; use examples to explain initialization rather than add another overlapping configuration API. |
| Four classical blocks and uniform FFNs | Keep the explicit, experimental `Transformer()` target and registered-lowering interface. |
| Balanced numeric representations and compile-time normalization | Retain as lowering research; the necessary ranges and invariants depend on the program. |
| Fixed final RMS limiting | Keep optional in the application experiment; do not make it a library default. |
| Shared heuristic computations and learned card/deck/opponent routing | Keep in the application case study. No game-specific columns or embeddings in the compiler. |
| PPO, rollout collection, CUDA capture and cache budgets | Keep in the application backend with its own regression checks. |
| Superseded architectures, pilots, probes and training reports | Archive with provenance; do not turn each experiment into a supported API. |

## What the results establish

Exact outputs and useful gradients are separate properties. Compare forward
values against the source program and test the backward behavior that the
application actually depends on. Passing inference replay checks alone did not
detect a tied-maximum backward error in the experimental PPO adapter.

That adapter's reduction error produced a gradient for adding a common offset
to all action scores even though the policy was invariant to it. Replacing the
reduction with an ordinary `amax` implementation preserved the forward values
and corrected the derivatives. This was a backend correctness fix, not a reason
to change Neurify's generic compiler.

The corrected training run remained finite beyond 4,000 updates. On a checked
late checkpoint, representative final-state RMS values stayed near one and the
experimental RMS limiter was inactive. This does **not** isolate the limiter's
contribution or establish that it is necessary. It is not sufficient evidence
for adding normalization to every compiled model.

CUDA graph caches also require retained-workspace accounting, not just live
tensor accounting. That finding belongs to GPU execution infrastructure and
does not introduce a CUDA dependency into Neurify.

## Supported paths

For named scopes, exposed activations and selective training, use the existing
compiler and `Block` controls:

- [Function interface and probes](CONTROLS.md)
- [Scope training](SCOPE_TRAINING.md)
- [Block configuration](BLOCKS.md)
- [Runnable scope example](examples/train_scopes.py)

For a custom exact classical lowering, use
[the experimental transformer contract](CLASSICAL_TRANSFORMER.md).
`Transformer()` defaults to four blocks, four heads and uniform FFNs.
It requires an explicitly registered builder and does not provide a generic
four-block lowering for arbitrary Python. The existing
`arch="transformer"` compiler keeps its original behavior.

[The classical example](examples/classical_transformer.py) contains a complete
Boolean lowering with exact original-function calls, verification, Torch export
and zero-output FFN initialization. It imports no application project or data.
The remaining runtime limitations, including named-scope probes and JAX export
for this target, are documented in the transformer contract.

## Promotion criteria

A new library feature needs an application-independent contract, domain and
shape validation, exactness checks, useful backward checks, checkpoint/export
behavior, and a self-contained example. Defaults also need evidence across
more than one program. An author assertion of exactness for a registered
lowering is recorded as such; it is not an automatically generated proof.

The private case study and its checkpoint ledger remain outside this repository.
Its training outcomes are research evidence, not claims made by the generic API.
