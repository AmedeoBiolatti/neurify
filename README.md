# neurify

Compile plain Python functions into neural networks that compute **exactly**
the same function, then use them as an initialisation and fine-tune them.

Neurify is experimental software (0.1.0), licensed under [MIT](LICENSE).
Exactness depends on the declared input domain, architecture, and numeric
precision. Inspect `net.report` and run `nf.verify(net)` for your program.

## Install

Published releases install with:

```sh
python -m pip install neurify
```

For training exports, choose `python -m pip install 'neurify[torch]'` or
`python -m pip install 'neurify[jax]'`. The base package needs only NumPy.
Prebuilt release wheels target regular CPython 3.10–3.14 on Linux (glibc,
x86-64 and ARM64), macOS (Intel and Apple Silicon), and Windows x86-64.
Installing a matching wheel does not need Rust or a C compiler.

The first public release still needs its repository, native wheel builds,
and PyPI setup completed. See [RELEASING.md](RELEASING.md) for the exact
release steps and current validation limits.

### Install from source

Requirements: Python 3.10 or newer, a recent stable Rust toolchain, and a C
linker. From the project root:

```sh
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

For optional exports, install `'.[torch]'` or `'.[jax]'` instead. Compilation
reads Python source, so put functions in a `.py` file; interactive functions
without retrievable source are unsupported. Compile trusted functions:
compilation may execute helpers and host callbacks.

## Quick start

```python
import neurify as nf

def f(x: nf.Int[0, 15], y: nf.Int[0, 15], cfg: {"w": nf.Int[1, 3]}):
    if x > y:
        return (x * y) % 7
    return max(x - y, -3) * cfg["w"]

net = nf.compile(f, arch="resmlp", method="depth")
net(9, 4, {"w": 2})          # 1.0, computed by the network
nf.verify(net)               # OK: 768 all points, max |err| = 0
print(net.summary())

init = net.prepare_for_training(width=64, depth=net.depth + 2)  # same function, spare capacity
model = init.to_torch()      # or: apply, params = init.to_jax()
```

## What "exact" means

The network equals the program on the **whole declared input domain**, not
just on samples. Every operation is lowered to an *exact gadget*: a small
fixed group of ReLU (or gated) units that computes it exactly. Some
operations cannot be exact in a given architecture, and neurify tells you:

| situation | behaviour |
|---|---|
| affine, `max/min/abs`, `relu`, bool logic, if/else on bounded values | exact |
| any operation on bounded `Int`/`Bool` values (`*`, `//`, `%`, comparisons, tables) | exact |
| real comparisons / `floor` when the inputs make the values a **finite set** | exact (exhaustive value analysis) |
| real comparisons / `floor` over a continuum | exact except in a `margin`-wide band at each decision boundary; reported in `net.report`; `strict=True` refuses |
| `real * real` | exact with `arch="glu"`; ReLU architectures refuse (`CompileError`) |
| `while` loops | unrolled; exact if termination within `max_loop_iters` can be **proven**, otherwise rejected |
| transformer, `attention="relu"` / `"hard"` | exact |
| transformer, `attention="softmax"` (default) | each attention read within `attn_tol` (1e-6); reported |
| transformer, `layernorm=True` | each LayerNorm read within relative error `ln_tol`; reported |

`net.exact` / `net.report["approx"]` state which case applies, and
`nf.verify` checks it.

The default `robust=True` makes discrete gadgets flat around every admissible
value, and all weights dyadic (powers of two). Integer programs then evaluate
bit-exactly in float64 *and float32* even when they are hundreds of layers
deep. For example, a 111-iteration Collatz loop compiles to 500 layers with
zero error.

## Supported Python

Functions (helpers are inlined; default and keyword args, nested `def`,
lambdas, closures), `if/elif/else`, ternaries, `and/or/not`, `for` over known
iterables and `range(n)` with a bounded symbolic `n`, `while`, `break`,
`continue`, early `return`, `assert`/`raise` (they exclude paths from the
domain), lists/tuples of fixed length (symbolic indexing, mutation, aliasing
across helper calls, `sorted` via sorting networks), dicts with keys known at
compile time (pytree-like, as in JAX), comprehensions, `min/max/sum/abs/
divmod/pow/int/float/any/all/enumerate/zip`, `math.floor/ceil/trunc`.
Anything computable from constants is evaluated at compile time. Opaque
callables (e.g. `math.comb`) and recursion that does not unroll are
**tabulated** over their finite argument domain.

With `Seq` inputs, a transformer is compiled like this:

```python
def f(xs: nf.Seq(nf.Int[0, 9], 6), k: nf.Int[0, 5]):
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return xs[best], xs[k] * 2 - xs[0]

net = nf.compile(f, arch="transformer", attention="relu")   # exact
print(net.n_tokens, net.heads)                              # 7 tokens, heads per layer
```

Not supported: strings beyond constants, data-dependent container
lengths or dict keys, classes, `try`, generators.

## Domains

`Real(lo, hi)`, `Int(lo, hi)` (inclusive), `Bool()`, `Vec(dom, n)`, nested
`[...]`, `(...)`, `{"key": dom}`, `Static(value)`. They can be given as
annotations (`x: Int[0, 9]`) or via `compile(f, inputs={...})`.
Tighter bounds give smaller networks.

## Architectures and methods

* `arch="mlp"`: plain ReLU MLP. A value used several layers later is passed
  through each intermediate layer by a ReLU unit (`relu(v - lo) + lo`).
* `arch="resmlp"`: residual ReLU MLP (`x += W2 relu(W1 x)`); values live in
  the residual stream, so nothing needs passing through.
* `arch="glu"`: residual ReGLU (`x += W2 (relu(Wa x) * Wb x)`), the MLP block of
  modern transformers. Products of real values are exact.
* `arch="transformer"`: token 0 runs the program; inputs declared as
  `Seq(dom, n)` live on tokens 1..n (value, position and position² features),
  and every read `xs[i]` (constant or data-dependent) is an attention head
  from token 0. MLP blocks are `mlp="relu"` or `"glu"` and are gated off on
  data tokens. Attention: `attention="softmax"` (default, standard; each read
  is one-hot within `attn_tol`, reported as approximate), `"relu"`
  (unnormalised `relu(1-(i-j)²)` scores, exact) or `"hard"` (argmax, exact,
  no gradient through the selection). `layernorm=True` opts into a stock
  pre-LN layout (approximate, see below).
* `method="depth" | "size" | "trainable" | "sparse"` chooses among
  alternative gadgets (e.g. int×int via squares, a binary or thermometer
  decomposition, or a lookup table) and the schedule. `trainable` prefers
  symmetric, well-conditioned gadgets. `sparse` minimises non-zero weights
  (then units): affine forms read by several units are factored into one
  unit, and gadgets use one unit where possible. It trades depth for
  sparsity (ConnectX: 61k → 29k non-zeros, depth 29 → 59); combine it with
  `arch="resmlp"` to also avoid pass-through units.

## Output encodings

`output="raw"` (default) returns the program's values. For decision
programs, `output="rank"` or `"onehot"` reads the flat output vector as the
scores of one decision and keeps only their order:
`rank_i = #{j: s_i > s_j}` (ties share a rank) and
`onehot_i = [s_i is a maximum]`. The argmax set is unchanged, but the outputs
are O(1).

The score comparisons are also compiled *lexicographically*. Scores like
`1e6*wins - 1e2*threats + builds`, or `select(legal, score, -1e9)`, are split
into tiers whose scale dominates everything below them; this is checked
from interval bounds. Each comparison then works on small integers, one tier
at a time. For the ConnectX heuristic:
- max |weight| drops from 477 to 16;
- the network becomes exact in float32;
- its argmax survives 1e-3 relative weight noise; with raw scores, 1e-6
  noise already changes 23% of the moves.

## Training

`prepare_for_training(width, depth, extra_dims)` applies function-preserving
transforms:
- normalise inputs to `[-1, 1]`;
- balance per-unit weight norms (ReLU positive homogeneity);
- add spare units with random incoming and zero outgoing weights;
- add identity layers / zero-output residual blocks;
- for residual nets, add spare stream dimensions.

`to_torch()` returns an `nn.Module`; `to_jax()` returns `(apply, params)`.

### What makes a compiled network trainable

`nf.trainability_report(net)` measures:
- which layers and parameters receive gradient;
- how many plain-Adam noise steps the network survives while still being
  exact, at each learning rate;
- per layer: the pre-activation noise it tolerates, the learning rate that
  implies, and the bottleneck layer;
- He ratios and activation scales.

The main lesson is that one Adam step moves every pre-activation by about
`lr · |x|₁`. A unit whose input ranges over R integer values needs relative
precision of about 1/(4R). So the tolerated learning rate is set by the value
ranges inside the program:
- small-range programs (booleans, small counters) tolerate lr = 1e-3;
- programs with large internal ranges tolerate 1e-9 to 1e-6, whatever the
  parametrisation.

Knobs:
- `output="rank"` / `"onehot"` with lexicographic tiers removes large score
  ranges.
- `method="trainable"` chooses gadgets that need little precision: one-hot
  pair lookups instead of packed indices, and AND / OR gates that are flat
  around every 0/1 input. On `gcd` this raises the predicted tolerance 700×.
- `slope=s` (0..1) gives discrete gadgets a slope of `s` at valid values.
  They stay exact, but gradients flow through them. Absorbing errors and
  passing gradients pull in opposite directions; `slope` sets the balance.
- `repr="onehot"` represents every value with at most `onehot_max` (64)
  distinct values as a vector of 0/1 indicators. Small integer inputs are
  one-hot encoded by a fixed, parameter-free input feature map.
  - Unary operations are free regroupings of indicators.
  - Every n-ary operation is one robust AND per reachable argument tuple.
  - No unit sees an input range above the number of arguments, and few
    units are active at once.
  - This is the first representation that survives Adam at standard
    learning rates while staying exact in float32:

    | Adam tolerance (10 noise steps, 99% exact) | gcd | int_branches | loops | bubble |
    |---|---|---|---|---|
    | scalar (default) | none (< 1e-6) | none | none | 3e-6 |
    | `repr="onehot"`, `arch="mlp"` | **1e-4** | **3e-4** | **3e-5** | **1e-5** |

    It costs 2–7× more units. Use it with `arch="mlp"`: in residual
    architectures every block reads a wide stream, and that cancels the
    gain. It does not scale to programs whose intermediate values have huge
    joint domains (the ConnectX window counts would need hundreds of
    thousands of AND units).
- `net.to_torch_residual()` / `net.to_jax_residual()` is the robust way to
  fine-tune with standard optimisers and learning rates:
  `y = exact(x) + delta(features(x))`.
  - The compiled network is frozen.
  - `delta` is a plain MLP on one-hot or normalised inputs, with its last
    layer set to zero.
  - The model starts as the program and training can only add to it.

## Training regions and inspecting variables

Compiled networks support both function-shaped prediction and inspection:

```python
net = nf.compile(program)
output = net(x=0.5)
result = net.inspect(x=0.5)
# result.output
# result.exposed["physics"]["energy"]
```

Both calls preserve the original arguments and return structure. Inspection
gets all exposed values and the output in one network run. For a trained
Torch model, use `model.as_function()` or `model.call(...)` / `model.inspect(...)`;
raw tensor calls remain available for training. See
[CONTROLS.md](CONTROLS.md) for complete examples and JAX calls.

Use scopes directly inside a function:

```python
def program(x: nf.Real[-1, 1]):
    with nf.scope("physics", trainable=False, expose=("energy",)):
        energy = x + 2
    with nf.scope("decision"):
        score = 3 * energy
    return score + x

net = nf.compile(program)
model = net.to_torch()
# y, values = model.forward_with_intermediates(X)
# values["physics"]["energy"]
```

Keep architecture and training choices separate from the source function:

```python
net = nf.compile(program, blocks={
    "physics": nf.Block(arch="glu"),
    "decision": nf.Block(residual=(32, 32)),
})
model = net.to_torch().train_only("decision")
print(model.summary())
```

Each block can override architecture, add function-preserving capacity, or
learn a zero-initialized correction around a frozen compiled base.
See [BLOCKS.md](BLOCKS.md) and `examples/blocks.py` for the full interface.

Split a function into named helpers to preserve boundaries during training:

```python
net = nf.pipeline(
    nf.region(features, trainable=False),  # fixed weights; input gradients still flow
    nf.region(physics, differentiable=False),  # stop gradients at this output
    nf.region(decision),
)
model = net.to_torch()
y, values = model.forward_with_intermediates(X, structured=True)
```

Define `features`, `physics`, and `decision` with declared input domains.
Return named variables in a dict to access them through `values[region][key]`.
The observations come from the exported model's current weights, so they can
show changes during training. Scoped probes read shared activations consumed
by later statements; exposing one adds no trainable prediction head. Dead
locals without a retained activation cannot be exposed. See [CONTROLS.md](CONTROLS.md) for complete
examples, auxiliary losses, JAX exports, and how to distinguish local weight
changes from changed upstream inputs. [SCOPE_TRAINING.md](SCOPE_TRAINING.md)
records real Torch experiments training energy only or scoring only; run
`python examples/train_scopes.py` to reproduce them.

## Layout

```
crates/neurify-core      IR, analysis, gadgets, assembly, transforms (pure Rust, no deps)
crates/neurify-frontend  Python parser (rustpython-parser) + symbolic interpreter
crates/neurify-py        PyO3 bindings (module neurify._core)
python/neurify           API: compile, verify, Network, torch/JAX export
tests/                  Python regression tests
```

## Development

```sh
python -m pip install -e '.[test]'
cargo test -p neurify-core -p neurify-frontend --locked
python -m pytest
```

Optional Torch and JAX tests skip when their dependencies are absent.
See [CONTRIBUTING.md](CONTRIBUTING.md) for rebuilding the extension and release
checks, and [DESIGN.md](DESIGN.md) for internals and the roadmap.

## License

Copyright (c) 2026 Neurify contributors. Released under the
[MIT License](LICENSE). The root project, Python package, and all three Rust
crates declare MIT; each Rust crate includes its own license file.
