# neurify design notes

## Pipeline

```
Python source ──(rustpython-parser)──> AST
   │  symbolic interpreter (crates/neurify-frontend)
   │    partial evaluation of everything concrete; Host callback resolves
   │    globals / helper source / opaque callables in the Python process
   ▼
IR graph (crates/neurify-core/ir.rs)
   typed (Bool/Int/Real), interval bounds, hash-consed, constant-folded,
   total semantics (x/0 = 0)
   │  analysis.rs: exhaustive evaluation over the joint finite input
   │  domain -> exact bounds and value sets per node
   ▼
lower.rs: exact gadgets -> neuron DAG (neuron.rs)
   program values = affine forms over signals; only units cost anything
   │  assemble.rs: ASAP/ALAP scheduling; MLP carries or residual-stream slots
   ▼
Network (network.rs) ──train.rs──> function-preserving transforms
   └─> Python: numpy forward, verify, torch / JAX export
```

## Control flow in the interpreter

* A symbolic `if` runs both branches and merges every binding with
  `select(c, then, else)`. It also merges the contents of every list/dict
  reachable from any active frame: they are snapshotted, restored and merged
  in place, so mutation through aliases (`swap(xs, i, j)`) is handled.
* `return` / `break` / `continue` / natural loop end are path flags. Once a
  flag is symbolic, the rest of the block runs only on the paths where it is
  not set.
* `raise` and failing `assert` give `Undef`, which a merge drops. Those paths
  are excluded from the domain.
* Loops are unrolled. `while` must provably stop within `max_loop_iters`: the
  "still running" flag must be constant-false under interval or exhaustive
  analysis. Otherwise it is a compile error ("exact for bounded loops").
* Recursion unrolls as long as bounds fold the base case. Otherwise the
  outermost call is tabulated through the host (finite domains only).

## Numerical exactness — lessons that shaped the core

Exact in real arithmetic is not enough. Three mechanisms keep compiled
networks exact in floating point:

1. **Robust discrete gadgets** (`robust=True`). Integer steps and lookups are
   staircases with ramps placed mid-gap, so they are flat within ≥ ¼ gap of
   every admissible value. Rounding errors are *removed* at every discrete
   operation instead of amplified by the big-M constants of later conditionals.
   (The cheaper interpolating gadgets with `robust=False` have slope 1 at
   integers.)
2. **Dyadic weights.** Unit normalisation, ramp widths, input normalisation
   and balancing all use powers of two. Integer programs then have *no*
   rounding at all, up to 2^53 (f64) / 2^24 (f32).
3. **Range-based unit normalisation + materialisation.** Units are scaled so
   their activations are O(1). Affine forms whose coefficients become large
   relative to the value range (e.g. `n = 3n + 1` folded through 100 selects
   gives 3^100 coefficients) are re-emitted as a fresh unit.

## Gadget selection (`method`)

Each operation has alternatives, each with a (depth, units) cost estimate.
For example, int×int can be done as
- a pair lookup (depth 1, about |pairs| units);
- the squares identity `((a+b)² − (a−b)²)/4` (depth 1, about |a+b| + |a−b| units);
- a binary decomposition (depth ≈ log R, about 4 log R units);
- a thermometer decomposition (depth 2, about 3R units).

`depth` minimises (depth, units), `size` minimises (units, depth), and
`trainable` uses depth ordering with symmetric, well-conditioned
conditional-multiplication gadgets (`relu(v−K(1−c)) − relu(−v−K(1−c))`).

## Transformer layout

* Tokens: token 0 = "main" token holding the program state (like `resmlp`);
  tokens 1..n hold the `Seq` inputs (several `Seq`s of equal length share
  tokens as features) plus fixed features `pos`, `pos²`, `is_data`
  (`is_main` on token 0). The per-token embedding bias is the positional
  embedding.
* Every read of a sequence element is an attention signal in the neuron
  DAG. It is scheduled on half-steps (attention = odd, MLP = even), so
  layers alternate attention / MLP as in a standard block.
* Heads have d_k = 4 (keys `is_data, pos, pos², is_main`) and d_v = 1.
  - softmax / hard: score `2β·i·j − β·j²` (= `β i² − β(i−j)²`), main token at
    `−M`. β and M come from `attn_tol`, n and the value bound.
  - relu: score `1 − (i−j)²`, which needs `i²` as a signal (a square lookup).
    Queries issued by data tokens get a `−G·is_data` term, so their scores
    are ≤ 0.
* MLP units get `−K·is_data` added (K from interval bounds of the
  pre-activation on data tokens), so data-token streams stay equal to
  their embedding plus bounded attention outputs.
* Opt-in LayerNorm: every content dimension has a mirrored copy written
  with the opposite sign, so the per-token mean is exactly 0. Two anchor
  dims `±A` make the variance ≈ 2A²/d, so `LN(x) = x·(1 − O(‖c‖²/A²))` with
  gain `√(2A²/d + ε)`. A is chosen from the content bound so that each read
  is within `ln_tol` (tightened so attention scores and integer gadgets stay
  safe).

## Trainability lessons (ConnectX experiment)

* Exact is not the same as well-conditioned. A program whose outputs span a
  huge range while its decisions depend on small differences (lexicographic
  weights such as 1e6 / 1e2 / 1, or sentinels such as -1e9) compiles exactly,
  but gradient steps on the compiled weights flip its decisions immediately.
* Robust discrete gadgets are flat on the data manifold, so gradients do not
  flow through them. The compiled part acts as a fixed feature or policy
  prior unless spare capacity is added.
* `normalize_activations` makes activations O(1), but it moves the dynamic
  range into the weights (up to 1e9 here). `balance` (now the default
  scaling) spreads it evenly over the layers.
* What worked: residual fine-tuning on top of the frozen exact network, with
  a KL penalty towards it and best-checkpoint selection. It reached +10 points
  against the original heuristic and +44 points against Kaggle's `negamax`.
  The same network trained from scratch reached 2%.
* Compiler work that follows from this: an output encoding with a bounded
  range for decision programs, and per-weight learning rates derived from the
  compiled scales.
* Follow-up (rank + lex): bounded outputs fix multiplicative weight noise
  (robust to 1e-3 relative noise) and float32. They do *not* make plain Adam
  safe: Adam moves every weight with a non-zero gradient by about `lr`.
  Gradients reach only the layers after the last flat gadget, whose rows
  have large L1 norms (up to 768 here). So `lr = 1e-5` shifts the comparison
  values past their ¼-gap margins within 2 steps, and the policy collapses.
  Trainability with standard optimisers needs margins that are large
  relative to `lr · fan-in · activation`, and small row norms. This is the
  target of "natural" mode.

## Trainability: precision demand

`trainability_report` gives each layer two numbers:
- `z_tol`: the largest random ±δ perturbation of all its pre-activations that
  keeps the program exact;
- `lr_pred = z_tol / (steps · (|x|₁ + 1))`.

The prediction follows from Adam's first steps, which move every weight that
has a gradient by about lr, so a pre-activation moves by about `lr · |x|₁`. The
smallest `lr_pred` predicts the measured Adam tolerance well (bubble:
predicted 1e-5, measured 3e-6).

For a unit whose input takes R distinct values, `z_tol` scales like 1/R of
its activation scale. ReLU rescaling can only move tolerance between layers,
since the product over layers is fixed by the input and output scales. So the
real levers are structural: gadgets whose units see small ranges (one-hot
indicators, robust ANDs, lexicographic tiers) and fewer long linear
read-outs that sum many units.

The tension is that robust gadgets absorb errors because they are flat, and
flat gadgets pass no gradient. `slope` interpolates between the two. Making a
network exact under arbitrary standard-lr Adam steps is impossible for
programs with large internal ranges. So the supported recipe for fine-tuning
with standard methods is the frozen exact network plus a zero-initialised
trainable branch (`to_torch_residual`).

## One-hot values (`repr="onehot"`)

`Lowerer::lower_onehot` handles any node whose domain, and whose arguments'
domains, have at most `onehot_max` values. Such a node becomes
`[(value, indicator Lin)]`:
- unary: indicator of r = Σ indicators of the arguments mapped to r (no
  units);
- n-ary: one robust AND `ramp(Σ e, k − ¾, k − ¼)` per reachable argument tuple
  (joint samples when the domain is enumerated, otherwise the product,
  capped by `max_table`);
- `select` and `gather`: ANDs with the condition / index indicators only.

The scalar form `Σ v·e` is linear, so scalar gadgets can consume one-hot
values for free. The reverse conversion (robust equality hats) is used only
for nodes that fell back to scalar.

Inputs: `Network.features` is a fixed feature map (`Raw(i)` or
`Eq(i, v) = [round(x_i) = v]`) applied before the first layer, in Rust,
numpy, torch and JAX. With one-hot values, small integer inputs are encoded
there, so even the first layer never reads a raw range.

What is left: in `mlp`, values that skip layers are carried by
`relu(s − lo) + lo`. An indicator component whose value is 0 sits on the
kink and passes errors on to the output read-out. These carries are now the
bottleneck (loops: last hidden layers). The next step would be robust
carries for two-valued signals.

## Output encodings and lexicographic comparisons (`lex.rs`)

`Graph::encode_outputs` rewrites the outputs at IR level (`rank` / `onehot`).
Score comparisons go through `lex::greater`:

1. Strip a common sentinel. Each `select(c, A, K)` with the same constant
   `K` below (or above) every value becomes `φ(s) = select(c, A + S, 0)`,
   which is strictly monotone. `S` dominates every value and is a multiple of
   every tier scale, so the flag forms its own tier.
2. Group every score into tiers (`T_k`, new IR nodes). The flag masks each
   tier: `select(c, T_k, 0)`.
3. For a pair, split the difference `Δ = s_i − s_j` into tiers with a proof:
   at every cut, `gcd(coefficients above) > max |terms below|`. Constants go
   into the highest tier whose gcd divides them. Then
   `sign Δ = lexsign(D_1, …, D_m)`, and `[Δ > 0] = [Σ 3^(m−1−k) sign D_k > 0]`
   (the last tier only needs its `>` bit).

If no proof is found, the comparison falls back to the direct one.

## Named training regions

`region` wraps an independently compiled `Network` with a name, a parameter
training policy, and an output-gradient policy. `pipeline` connects flat
outputs to flat inputs in helper/parameter order. Separate compilation keeps
these boundaries visible after lowering, assembly, and training transforms;
ordinary inlining could fold affine expressions across them and erase the
location where a gradient or observation should apply.

The NumPy pipeline composes initialized networks. Torch exports one module
per region: frozen parameters use `requires_grad=False`, while input
derivatives remain available; stopped outputs use `detach()`. JAX exports
keep frozen weights outside the trainable parameter pytree and stop output
gradients explicitly. Observations read the same region outputs consumed by
the next module, with optional detached snapshots and structured names.

Boundary checks enumerate small discrete domains or use IR output specs
(`Graph.output_specs` in the bindings). Python also has conservative network
interval bounds for older extensions. Unproven domain compatibility prevents
a whole-pipeline exactness claim. Region controls are deliberately rejected
inside ordinary `compile` rather than silently inlined or tabulated.

Lexical `scope` blocks are lowered in Python to independently compiled code
blocks. A backward liveness pass selects cross-block values. Within each
scope, live scalar statement outputs are materialized with fixed wiring to
their consumers, independent of exposure. A probe reads the actual retained
activation and never adds an output/readout head. Dead locals cannot be
exposed. Runtime wiring updates only locals written in a
block, so untouched values can bypass a stopped scope. Ordinary local or
structured returns use fixed wiring, including local leaves alongside
computed return expressions. The complete scope-stripped program is
traced separately for verification against both IR and original Python.

Torch live probes preserve the exact tensor objects used by subsequent
operations. Function adapters retain their batch dimensions instead of
creating sibling reshape views for probes. Detached clones are only created
for requested snapshots. JAX follows the same shared dataflow, with frozen
weights outside the parameter tree and optional observation stop-gradients.

This implementation supports flat scope blocks, scalar boundary state, and
one final return. Arbitrary nested scopes, mutable container aliases across
boundaries, and early returns require a more general control-flow graph and
source-to-IR metadata that survives optimization.

## Roadmap

1. Attention aggregation shortcuts: `sum` / `mean` over a `Seq` with one
   uniform-attention head instead of n reads, and a parallel argmax over a
   `Seq` instead of the sequential scan the program spells out.
2. Residual-stream slot reuse (register allocation) to cut `d_model`; long
   unrolled loops currently give large networks.
3. Cost-model driven global gadget selection (currently local / greedy).
4. Python-side niceties: `@nf.compile` caching keyed on source, better
   source locations in error messages for inlined helpers.
