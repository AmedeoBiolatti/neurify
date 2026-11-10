# Training controls and intermediate values

Use [`Block` configurations](BLOCKS.md) to choose each scope's architecture,
extra capacity, or frozen-base residual correction independently of the
function's source. Torch `model.train_only("scoring")` selects the blocks
that learn; `model.summary()` reports current structure and training flags.

## Calling the NN and getting exposed variables

The high-level interface has two calls with the original function's
arguments:

```python
import neurify as nf

def program(x: nf.Real[-1, 1], *, shift: nf.Real[0, 2] = 1):
    with nf.scope("physics", expose=("energy",)):
        energy = x + shift
    return {"score": 2 * energy, "energy": energy}

net = nf.compile(program)

output = net(0.5)  # {"score": 3.0, "energy": 1.5}
result = net.inspect(0.5)
assert result.output == output
assert result.exposed == {"physics": {"energy": 1.5}}

# It can also be unpacked as a pair.
output, exposed = net.inspect(x=0.5, shift=0.25)
```

`inspect` returns `nf.Evaluation(output, exposed)` from **one network
execution**. It collects all exposed values in all scopes, keyed by scope
and variable name. It does not re-run the original Python function to obtain
those values. Prediction calls compute only the output; ordinary networks
without exposed values return an empty `exposed` dictionary.

Arguments honor Python defaults, positional-only and keyword-only rules,
and duplicate/unknown argument errors. Scalar, tuple, list, dict, and nested
return structures are reconstructed. `inspect.signature(net)` and
`inspect.signature(net.as_function())` show the source signature. Parameters
compiled as Static remain fixed: they can be omitted, and explicit values
must match the compiled value.

NumPy calls return Python numeric leaves for a single example, or NumPy
arrays for batches. Numeric outputs remain floating point by default. Use
`net.as_function(cast=True)` for integer/bool inference results. This keeps
adapter options separate from original arguments, including arguments named
`cast`, `detach`, or `structured`. The older `net(..., cast=True)` convenience
also works when the source function has no `cast` parameter.

Scalar argument leaves can contain broadcastable batch arrays. Declared Vec
or Seq inputs can use arrays with the declared shape in trailing dimensions,
such as `(batch, vector_length)`. Output and exposed leaves preserve the
broadcast batch dimensions. The original input/return tree remains intact.

### Calling a trained Torch model like the function

```python
model = net.to_torch()
program_nn = model.as_function()

output = program_nn(x=0.5)
result = program_nn.inspect(x=0.5)
# Equivalent convenience calls:
output = model.call(x=0.5)
result = model.inspect(x=0.5)
```

These calls return the original output structure with Torch tensor leaves.
They read the model's **current** parameters, including later optimizer
updates or loaded checkpoints. Input tensor gradients and output gradients
are preserved. Continue to pass `model.parameters()` to your optimizer.

Exposed tensors are detached snapshots by default. For live observations
and auxiliary losses, use `model.as_function(detach=False).inspect(...)`.
`cast=True` requests integer/bool inference tensors; those casts do not
preserve gradients for the cast values.

The raw model remains a normal module: `model(X)` and `model.forward(X)` take
flat tensors of shape `(batch, n_in)` or `(n_in,)` and return flat outputs.
`forward_with_intermediates(X)` is still available for raw tensor inspection.
The adapters are also attached to ordinary and residual Torch exports.

### JAX function calls

```python
apply, params = net.to_jax()
output = apply.call(params, x=0.5)
result = apply.inspect(params, x=0.5)

# Bind one parameter tree for a callable with just the original arguments.
program_nn = apply.as_function(params)
output = program_nn(x=0.5)
```

Use `apply.call` / `apply.inspect` with the current parameter tree during
training. If you replace `params`, rebind an adapter created with
`apply.as_function(params)`. A JIT-compatible inspection call is
`jax.jit(lambda p, x: apply.inspect(p, x))`. The result is a named tuple pytree
with output/exposed array leaves. To differentiate an auxiliary loss, create
the bound adapter with `detach=False` inside the loss using its current
parameters. The raw `apply(params, X)` API remains available.

## Scopes inside one function

Use `nf.scope` to keep ordinary local variables in a single function:

```python
import neurify as nf

def program(x: nf.Real[-1, 1]):
    with nf.scope("physics", trainable=False, expose=("energy",)):
        energy = x + 2
    with nf.scope("decision"):
        score = 3 * energy
    return score + x

net = nf.compile(program)
assert nf.verify(net).ok
y, values = net.forward_with_intermediates([[0.5]])
assert values["physics"]["energy"][0] == 2.5

# model = net.to_torch()
# y, values = model.forward_with_intermediates(X)
# print(values["physics"]["energy"])
```

`nf.scope` is a no-op context manager in normal Python. `nf.compile` detects
it and compiles the code blocks separately, with explicit variable wiring.
Locals still have ordinary Python function scope: a name assigned inside a
`with` block remains available afterward.

`trainable=False` freezes the scope's weights and preserves input
derivatives. `differentiable=False` stops gradients on values defined in that
scope. Values that bypass a scope keep their own paths. For example, in
`y = 3*x` inside a stopped scope followed by `return x+y`, the direct `x`
path still has derivative 1. Returning an existing local uses fixed wiring;
it does not add another trainable identity network.

By default, observations contain the scope's live output locals. `expose`
selects retained activations, including internal values used by later
statements: `expose=("energy", "force")`. These locals must be assigned in the
block and defined on every path reaching their boundary. Scoped observations
default to named dicts; `structured=False` returns flat snapshots in `expose`
order. Live Torch/JAX scope probes require `structured=True`. Torch
observations are detached snapshots unless `detach=False` is requested.

The existing runtime methods work with scope names:
`model.set_trainable("physics", False)` and
`model.set_differentiable("physics", False)`. JAX uses
`apply, params = net.to_jax()` and `apply.with_intermediates(params, X)`.
Frozen weights stay outside the JAX parameter tree. The general guidance
below on gradients, snapshots, and local versus upstream changes applies.

The initial scope syntax supports top-level, non-nested `with nf.scope(...)`
blocks and one final return. Loops and branches can be placed **inside** a
scope. Names, options, and exposure lists must be literal constants. Helper
functions can be called from a scope; define them outside the scoped
function. Each scalar variable crossing a boundary gets an inferred domain.
Real/Int/Bool boundary values and immutable Static inputs are supported;
container inputs/locals crossing boundaries and early returns across scopes
are rejected. The final return may have a nested tuple/list/dict structure.
Scoped compilation currently requires `output="raw"`.
Boundary checks are conservative: initialize locals before loops or assign
them in both branches before returning or exposing them across a boundary.

Code outside explicit scopes is compiled with the default training policy
and has generated names beginning with `__neurify_` in exported parameters.
`net.scope_names` lists the explicit scopes. Scopes introduce compilation
boundaries, so they can increase size or require a different architecture
when cross-scope algebra can no longer be folded into one gadget.

## Train energy only or scoring only

The runnable [examples/train_scopes.py](examples/train_scopes.py) compiles
`energy = x + 2` followed by `score = 3 * energy + 1`, then fits the new target
`score = 2 * x + 1`. Both scopes expose their actual activations.

```python
import torch
from examples.train_scopes import program

net = nf.compile(program).prepare_for_training(normalize=True, scaling=None)
model = net.to_torch(dtype=torch.float64)
model.set_trainable("energy", True).set_trainable("scoring", False)

# Frozen scoring weights must still pass derivatives back to energy.
model.set_differentiable("scoring", True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
function = model.as_function(detach=False)
x = torch.linspace(-1, 1, 129, dtype=torch.float64)
target = 2 * x + 1
for _ in range(800):
    optimizer.zero_grad(set_to_none=True)
    loss = (function(x=x) - target).square().mean()
    loss.backward()
    optimizer.step()

# To train scoring instead, start from a fresh model and reverse the flags:
# model.set_trainable("energy", False).set_trainable("scoring", True)
```

With energy trainable, the energy map becomes `(2/3)*x` while scoring keeps
its original `3*energy + 1` map. With scoring trainable, energy stays `x + 2`
and scoring becomes `2*energy - 3`. A frozen scope's output can change when
its input changes; freezing preserves its weights, not its activations.
`set_differentiable("scoring", False)` would cut the final loss's gradient
back to energy in this example, so use `set_trainable` for weight freezing.

For a loss on energy itself, replace the loss computation with:

```python
result = function.inspect(x=x)
energy = result.exposed["energy"]["energy"]
loss = (energy - (2 / 3) * x).square().mean()
```

This trains the actual energy producer used by scoring. Default
`model.inspect(...)` gives detached snapshots for observation; choose
`detach=False` for a probe used in a differentiable loss.

Run `python examples/train_scopes.py` for all three experiments and a JSON
record of losses, learned maps, activation gradients, and frozen-state
checks. See [SCOPE_TRAINING.md](SCOPE_TRAINING.md) for measured results and
the distinction between this affine control test and general trainability.

## Exposed variables are shared activations

Inside a scope, live scalar values are materialized between top-level
statements. Later statements consume these exact values. Exposure only
selects observation metadata; it adds no trainable readout or prediction head
and does not change the model's parameters or topology.

```python
def f(x: nf.Real[-1, 1]):
    with nf.scope("calculation", expose=("a",)):
        a = x + 2
        b = 3 * a
    return b

net = nf.compile(f)
assert net.activation_names["calculation"] == ("a", "b")
result = net.inspect(0.5)
assert result.exposed["calculation"]["a"] == 2.5
assert result.output == 7.5
```

Here `a` is the actual input to the computation of `b`. Changing the producer
of `a` changes both the observation and the downstream result. Training can
change the downstream mapping, so the initial relationship `b = 3*a` need
not remain fixed when its weights are trainable.

Live Torch inspection returns the shared activation tensor itself, including
through the function adapter. It supports gradients from the **main output
loss**, not only auxiliary losses on the probe:

```python
import torch

model = net.to_torch(dtype=torch.float64)
view = model.as_function(detach=False)
result = view.inspect(torch.tensor(0.5, dtype=torch.float64, requires_grad=True))
a = result.exposed["calculation"]["a"]
a.retain_grad()
result.output.backward()
assert a.grad.item() == 3.0
```

The default detached snapshots read the same activation values without
retaining a backward graph. Collecting snapshots does not change prediction
or its gradients. Integer/boolean gadgets can still have zero gradients, and
a scope's `differentiable=False` policy detaches its probes as well.

Only existing activations can be exposed. Dead locals and values entirely
internal to a compound operation have no retained boundary and are rejected
rather than recomputed using a separate head. An `if` or loop is one compiled
operation; live values leaving it can be observed. Container values cannot
cross the materialized statement boundaries. Use helpers/compound operations
for container work and pass scalar results between statements.

For reassigned names, exposure refers to the final definition at scope exit.
If that definition is dead, exposure fails even if an earlier version was
used. `net.activation_names` lists the available names per scope. Return
packing also uses fixed wiring for existing locals, including mixed returns
such as `{"a": a, "b": 3*a}`; only the computed expression needs a new network.

Materializing statement boundaries can increase model size compared with the
earlier whole-scope optimizer and lose cross-statement value correlations.
This compilation policy applies with and without exposure. It preserves the
initial program on its declared domains while keeping source activations
meaningful during training.

## Explicit helper regions

Use named helper regions when parts of a program need different training
rules or when you want to inspect program variables after training. Each
region compiles independently. Its output is the actual input to the next
region, so observing it reads the trained computation at that boundary.

```python
import neurify as nf

def features(x: nf.Real[-1, 1]):
    shifted = x + 2
    return {"shifted": shifted}

def physics(shifted: nf.Real[1, 3]):
    response = 3 * shifted
    return {"response": response}

def decision(response: nf.Real[3, 9]):
    return 2 * response

net = nf.pipeline(
    nf.region(features),
    nf.region(physics, trainable=False),
    nf.region(decision),
)
assert net(0.5) == 15.0
assert nf.verify(net).ok
print(net.summary())
```

Region outputs are flattened in their return-structure order and connected
to the next region's inputs in parameter order. Here `features` returns one
value, so `physics` consumes one value. Lists, tuples, and dicts can carry
multiple variables. Static parameters consume no tensor columns.

Name a region explicitly with `name="features"`, pass an existing compiled
network to `nf.region(net, ...)`, or use `@nf.region(...)` as a decorator.
Use `region.network` to access the underlying compiled network.

## Two independent controls

| Setting | Region weights | Gradients to earlier regions |
|---|---|---|
| `trainable=True, differentiable=True` | trainable | permitted |
| `trainable=False, differentiable=True` | fixed | permitted |
| `trainable=True, differentiable=False` | receive no downstream gradient | stopped |
| `trainable=False, differentiable=False` | fixed | stopped |

Freezing a region fixes its mapping; its output can still change when an
earlier trainable region changes its inputs. Stopping gradients at a region
does not stop later regions from learning. In a sequential pipeline it also
stops the final loss from reaching all earlier regions. Live observations
before that boundary can still provide auxiliary losses to earlier regions.

`differentiable=True` permits the derivatives already present in the compiled
network. Discrete robust gadgets can have zero derivatives at valid inputs.
To adjust that behavior, pass compiler options per region, for example
`nf.region(helper, slope=0.5)` or `arch="glu"` for real products. These
controls do not replace the existing numerical conditioning constraints.

Region flags apply when exporting a pipeline. NumPy evaluation has no
autograd and computes the same numerical result for every flag combination.
Ordinary `nf.compile` rejects Region/Pipeline objects inside a traced
function, so it cannot silently inline away their boundaries and controls.

## Inspecting values during Torch training

```python
import torch

model = net.to_torch(dtype=torch.float64)
X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)

y0, before = model.forward_with_intermediates(X, structured=True)
print(before["features"]["shifted"])
print(before["physics"]["response"])

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
target = y0.detach() + 0.1
for _ in range(10):
    optimizer.zero_grad()
    loss = (model(X) - target).square().mean()
    loss.backward()
    optimizer.step()

y1, after = model.forward_with_intermediates(X, structured=True)
print(after["physics"]["response"] - before["physics"]["response"])
```

By default observations are detached, cloned snapshots. Keeping `before`
does not retain a backward graph, and later training cannot overwrite it.
The final output `y0`/`y1` retains its normal gradient behavior.

With `structured=True`, each observation has the helper's original return
structure and tensor leaves. The default is one flat output tensor per
region: `(batch, region_outputs)` for a batch or `(region_outputs,)` for one
input. `net.forward_with_intermediates(X)` provides the corresponding NumPy
values before export.

For an auxiliary loss, request live observations:

```python
y, values = model.forward_with_intermediates(X, detach=False, structured=True)
loss = (y - target).square().mean() + 0.01 * values["features"]["shifted"].square().mean()
```

Live observations preserve the region's gradient policy. Observations at a
`differentiable=False` output remain detached. Use the same fixed input batch
when comparing before and after training.

An observed change can come from local weights, upstream inputs, or both.
To isolate local change, compare the trained region with its original region
on the **same current input**:

```python
_, flat = model.forward_with_intermediates(X)
original_physics = net.regions["physics"].network.to_torch(dtype=X.dtype).to(X.device)
with torch.no_grad():
    original_on_current_input = original_physics(flat["features"])
    local_change = flat["physics"] - original_on_current_input
```

For the frozen physics mapping above, `local_change` remains zero even if its
observed response changed because the features region learned.

Torch controls can be changed at runtime:

```python
model.set_trainable("physics", False)
model.set_differentiable("physics", False)
model.set_differentiable("physics", True)
```

Freezing clears accumulated gradients for that region. Pass all model
parameters to the optimizer if you plan to unfreeze regions later; parameters
omitted from an optimizer do not get added by `set_trainable`. Region control
flags are included in `model.state_dict()` alongside the weights. Recreate
the same pipeline architecture before loading that state.

## JAX

```python
import jax
import jax.numpy as jnp

apply, params = net.to_jax()
assert "physics" not in params
Xj = jnp.array([[-1.0], [0.0], [1.0]])
y = jax.jit(apply)(params, Xj)
inspect = jax.jit(lambda p, x: apply.with_intermediates(p, x, structured=True))
y, values = inspect(params, Xj)
```

Frozen-region parameters are captured as constants outside the returned
parameter tree, so optimizer weight decay cannot alter them. Input
derivatives through frozen regions are preserved. A stopped region uses
`jax.lax.stop_gradient` at its output.

Observations stop gradients by default; set `detach=False` for auxiliary
losses. Pass `detach` and `structured` as static arguments when using JIT,
as in the lambda above. JAX arrays are immutable snapshots. Change region
policies and export again when you need a different JAX parameter tree.

## Domains and scope

Each helper must declare input domains, including intermediate inputs. The
pipeline checks widths and attempts to establish boundary-domain
compatibility. Small finite domains are checked exhaustively; otherwise it
uses IR output bounds, with conservative network bounds as a fallback for
older extensions. A proven discrete-domain mismatch is rejected. Unproven
compatibility is reported and prevents `pipeline.exact` from being true.
Wider intermediate domains can help satisfy conservative bounds.

`nf.verify(pipeline)` compares the composed network with the composed IR.
For additional checks against Python execution, run
`nf.verify(region.network)` on each region. Exactness describes initialization
on declared domains; training can change the function and move intermediate
values outside those domains.

Explicit regions support sequential named helpers, including multi-value
state passed between them. Return a variable from a helper to expose it at
a preserved boundary; with `structured=True`, dict keys give variables names.
For scalar locals inside one function, use the scope syntax above.
There is no observation-to-source reverse mapping for the flat architecture
produced by ordinary `nf.compile`. Separate compilation also prevents
cross-region optimization and can increase network size.
