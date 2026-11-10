# Changelog

## Unreleased

- Prepare version-tagged public releases with five portable abi3 wheels,
  source distributions, clean-install/native smoke checks, full CI gates,
  GitHub release assets and direct GitHub installation. Validate
  version/license consistency, archive completeness, and wheel metadata.

- Add immutable `Block` configurations to `compile(..., blocks={...})` for
  per-scope architecture, extra width/depth, gradient/training policies,
  and zero-initialized residual corrections around frozen compiled bases.
- Add Torch `train_only`, named `blocks`, and current structure/parameter
  summaries. Preserve base freezing across runtime switches and checkpoints;
  corrected scope output probes remain shared downstream activations.

- Add a reproducible selective-training experiment for energy-only,
  scoring-only, and live energy-activation losses. Check optimizer updates,
  frozen state, analytical activation gradients, runtime scope switching,
  and freezing after AdamW has accumulated momentum.

- Materialize shared scalar statement activations inside scopes. Downstream
  computation now consumes exposed values, with no independent probe heads.
- Make exposure a metadata-only choice for retained activations; reject dead
  or unavailable locals instead of adding computation to observe them.
- Preserve the actual live activation tensors for main-loss gradient
  inspection, including scalar and broadcast function calls and region leaves.
- Use fixed return wiring for existing variables in mixed computed/structured
  returns. Add topology-invariance, producer-intervention, and autograd tests.

- Add function-shaped prediction and `inspect` returning a named
  `Evaluation(output, exposed)` from one actual network run.
- Preserve source signatures, argument defaults, positional/keyword rules,
  return pytrees, and broadcast batch shapes. Validate fixed Static arguments
  and avoid collisions with source arguments named cast/detach.
- Add current-weight Torch adapters and JAX parameterized call/inspection
  adapters, including ordinary/residual exports and live auxiliary losses.
- Respect positional-only arguments in Python reference verification and add
  interface, batching, single-pass, and serialization regression tests.

- Add `with nf.scope(...)` inside a single function, automatically lowered
  to compiled blocks with local-variable wiring and inferred scalar domains.
- Expose named scope locals without helper refactoring; preserve bypass
  gradient paths and use fixed wiring for ordinary variable/structured returns.
- Validate the initial scope subset explicitly and add NumPy, Torch/JAX,
  liveness, static-value, and structured-output regression coverage.

- Add `region` and `pipeline` for named, independently compiled helper
  boundaries with separate parameter-freezing and output-gradient controls.
- Add flat and structured observations of actual region outputs in NumPy,
  Torch, and JAX; Torch observations are detached snapshots by default and
  can remain live for auxiliary losses.
- Keep frozen JAX parameters outside the optimizer tree. Support runtime
  Torch control changes and checkpoint their flags with model weights.
- Check intermediate-domain contracts and expose IR output specifications.
- Add a controls guide, a training example, and backend gradient regression
  tests in CI. Reject inlining controlled helpers into ordinary compilation.

- Standardize the workspace, Python distribution, and Rust crates on MIT.
  Include license files in the root project and each crate.
- Add package readmes, Python package metadata, installation instructions,
  contribution guidance, and CI for Rust, base Python, and optional backends.
- Reject non-finite margins and invalid transformer attention/LayerNorm
  tolerances before compilation. Validate only tolerances used by the chosen
  architecture.
- Reject malformed table shapes without integer overflow, and require finite
  integer table origins.
- Validate flat Python batch shapes and node inspection requests with
  CompileError instead of truncating input or panicking.
- Support zero-input networks in the Rust batch API and reject incomplete
  rows and incorrect single-input lengths explicitly.
- Keep Torch optional during Python test collection; run independent
  compiler tests when Torch is absent.
- Add regression coverage for options, tables, batches, and Python boundaries.
