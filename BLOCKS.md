# Configuring neural blocks

Scopes name code boundaries and select observations. A `Block` describes how
to compile and train a named scope, without changing the Python function:

```python
import neurify as nf

def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score

net = nf.compile(program, blocks={
    "energy": nf.Block(arch="glu"),
    "scoring": nf.Block(residual=(32, 32)),
})
model = net.to_torch().train_only("scoring")
print(model.summary())
result = model.inspect(x=0.5)
# result.output and result.exposed["energy"]["energy"]
```

The original function remains directly callable. `net(...)`, `model.call(...)`,
and `model.as_function()` keep its arguments and return structure.
The same `blocks` mapping works with `@nf.compile(blocks=...)`.

## Structure and initial behavior

```python
net = nf.compile(program, arch="mlp", blocks={
    "energy": nf.Block(arch="glu"),
    "scoring": nf.Block(extra_width=32, extra_depth=2),
})
assert net(0.5) == 8.5
```

`arch` overrides the global architecture for that scope. Other compiler
options keep their global values. `extra_width` adds spare units to compiled
layers across the scope. `extra_depth` inserts identity layers or blocks in
the scope's last live statement. Earlier activation boundaries stay in place.
These additions preserve the initial function; the compiler still determines
the architecture-specific representation and required base size.

Width additions affect existing hidden layers. An affine MLP has no hidden
layers to widen; pair `extra_width` with `extra_depth` when adding nonlinear
capacity to such a scope. Check `summary()` for the resulting structure.

## Residual corrections

```python
net = nf.compile(program, blocks={
    "scoring": nf.Block(residual=(32, 32), seed=7),
})
model = net.to_torch().train_only("scoring")
```

The scoring block computes
`compiled_scoring(energy) + correction(normalized_energy)`. Its compiled base
is frozen. Only the correction is trainable, and its final layer starts at
zero, so the initial prediction equals the compiled function. Hidden layers
use ReLU. `residual=()` selects a linear correction without hidden layers;
`residual=None`, the default, trains the compiled block directly. `seed`
controls spare capacity and correction initialization without changing the
initial output.

Corrections are added to the scope's live output locals. An exposed output
local reads its corrected activation, which downstream code also consumes.
Exposed internal locals still read the actual internal compiled activations;
the correction does not rewrite the scope's internal formula. Internal and
output observations therefore describe different stages of the computation.
Changing `expose` adds no correction heads or parameters.

`model.blocks["scoring"].base` and `.delta` expose the frozen compiled module
and correction module for inspection. `model.parameters()` remains the
optimizer input. `train_only` and `set_trainable` preserve the frozen base
even when the residual block is re-enabled. Input derivatives pass through
both branches; freezing the base does not detach its inputs.

## Choosing what learns

```python
model.train_only("energy")             # Freeze every other block.
model.train_only("scoring")            # Switch without rebuilding the model.
model.train_only("energy", "scoring")  # Enable both.
model.train_only()                     # Freeze all blocks.
```

All names are validated before flags change. Generated code and return blocks
are frozen unless explicitly selected. Freezing clears accumulated gradients;
it also works when the optimizer already contains all parameters and momentum.
Gradient-flow policies are unchanged. A frozen downstream scope can pass
gradients to an upstream trainable scope.

For defaults that also apply to JAX export, configure the block up front:

```python
net = nf.compile(program, blocks={
    "energy": nf.Block(trainable=False),
    "scoring": nf.Block(residual=(32, 32)),
})
apply, params = net.to_jax()
# params contains scoring's correction only; energy and the scoring base
# remain outside the parameter tree.
```

`trainable=None` and `differentiable=None` inherit the options on `nf.scope`.
Explicit block values override those defaults. Unspecified architecture
inherits `nf.compile(arch=...)`. Unknown names and malformed configurations
raise errors. `Block` configurations are immutable and retained in
`net.block_configs`, including after `prepare_for_training(...)`.

Use `model.as_function(detach=False).inspect(...)` for live probe losses or
activation gradients. Default inspection returns detached snapshots.

## Inspection and verification

`net.summary()` reports architecture, compiled/residual mode, depth, widths,
correction hidden widths, total and trainable parameter counts, gradient
policy, and exposed variables. `model.summary()` reports the current training
flags and actual Torch parameter counts. `net.report["blocks"]` provides
structured initial block information. `net.blocks` and `model.blocks` select
explicit source scopes by name; `regions` also includes generated regions.

`nf.verify(net)` checks the initialized network against the original function.
After training, the exported model's predictions and probes reflect its
learned function; the source `net` retains its compiled initialization.
Zero initialization and spare-capacity transforms preserve the initial
function but do not guarantee convergence on an arbitrary new target.

Run `python examples/blocks.py` for a complete scoring-only residual training
experiment that checks the fit and frozen parameters. Run
`python -m pytest tests/test_blocks.py -q` for architecture, capacity, live
activation, gradient, switching, checkpoint, shape, and parameter-count checks.
JAX coverage is included in backend CI; it is unavailable locally.

Local validation passed 453 Python tests with 18 skipped, including 40 new
block-interface cases on NumPy and real Torch. A complete scoring-only
linear-correction example reached score MSE 1.02e-27 while all frozen
parameters stayed unchanged. The test log is distributed beside this guide.
