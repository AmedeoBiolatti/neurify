# Selective scope training: measured behavior

Real CPU training verifies that either energy or scoring can learn independently.
Both use the same compiled function and final target, so their internal changes
can be compared directly. No control implementation changes were needed during
this validation pass.

```python
def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score
```

The initial prediction is `3*x + 7`. The training target is `2*x + 1`,
evaluated on 129 evenly spaced inputs in `[-1, 1]`. The experiment uses
Python 3.12, Torch 2.11.0+cu128 on CPU, float64, Adam at learning rate 0.05,
and 800 optimizer steps. Fixed input normalization is enabled with
`prepare_for_training(normalize=True, scaling=None)`; no spare capacity is
added. The optimizer deliberately contains **all** model parameters,
including frozen parameters.

| Training loss / enabled scope | Initial score MSE | Final score MSE | Frozen scope state |
| --- | ---: | ---: | --- |
| Score loss / energy only | 36.3385 | 3.06e-32 | scoring: bitwise unchanged |
| Score loss / scoring only | 36.3385 | 1.14e-27 | energy: bitwise unchanged |
| Energy activation loss / energy only | 36.3385 | 9.17e-32 | scoring: bitwise unchanged |

For the third run, the loss is on the live exposed energy tensor with target
`(2/3)*x`; the table still measures the resulting final **score** error.

## Where the learning went

| Run | Energy map after training | Scoring map after training |
| --- | --- | --- |
| Energy only | energy ≈ (2/3)*x | score = 3*energy + 1, unchanged |
| Scoring only | energy = x + 2, unchanged | score ≈ 2*energy - 3 |
| Energy activation loss | energy ≈ (2/3)*x | score = 3*energy + 1, unchanged |

Both paths produce the target score, while the exposed energy clearly
distinguishes the learned implementations. At `x=0`, for example, energy
becomes approximately 0 in the energy-only run and stays exactly 2 in the
scoring-only run. Both scores become approximately 1.

A frozen scoring scope's **output** changes if energy changes. Its weights
and its mapping stay fixed. This is expected propagation from its input.
The experiment checks all frozen parameters and buffers for bitwise
equality, verifies the active scope changed, and checks frozen parameter
gradients are `None` throughout training.

## Shared activation and optimizer checks

Before training, analytical MSE derivatives are checked on the actual
exposed tensors: `dL/dscore = 2*(score-target)/129` and
`dL/denergy = 3*dL/dscore`. The largest energy-gradient discrepancy is
5.55e-17. The score probe equals the returned prediction, and an energy
probe loss updates the same energy producer consumed by scoring.

The 12 new regression cases cover final-output training with energy-only
and scoring-only flags across `mlp`, `resmlp`, `glu`, and `transformer`,
plus the live activation loss, runtime scope switching, freezing after
AdamW momentum and weight decay have been used, and an explicit gradient
stop. Freezing scoring still passes derivatives back to energy.
`set_differentiable("scoring", False)` stops that path as intended;
an upstream live energy loss remains usable.

## Reproduce

After installing the updated Neurify source and Torch:

```sh
python examples/train_scopes.py --output artifacts/scope_training.json
python examples/train_scopes.py --arch transformer --output artifacts/scope_training_transformer.json
python -m pytest tests/test_scope_training.py -q
```

Use `model.set_trainable("energy", True).set_trainable("scoring", False)`
for energy-only training, or reverse the flags for scoring-only training.
Leave scoring differentiable to train energy from the final score loss.
See [CONTROLS.md](CONTROLS.md) for complete training code.

Without input normalization, 1600 steps left score MSE at 3.92e-7 in the
scoring-only run, above the experiment's 1e-7 acceptance threshold. With
3200 steps both unnormalized runs passed: energy-only reached 1.14e-31
and scoring-only reached 1.57e-26, with frozen state unchanged. Reproduce
with `--no-normalize --steps 3200`. This is an optimization-speed effect;
the frozen-state and gradient checks still hold.

## Validation scope

The full Python test directory passed **413 tests**, with **17 skipped**,
using the existing native extension and the real Torch installation.
The new selective-training cases are included in the backend CI job.
The JSON record distributed beside this guide contains the learning curves,
sample activations, effective maps, and per-scope parameter changes for
the three normalized MLP runs and both unnormalized MLP runs.

This affine example isolates scope ownership and gradient flow. It does
not establish that arbitrary nonlinear functions, discrete gadgets, targets,
or training setups can converge. JAX is unavailable locally, and a full
Rust rebuild and wheel/sdist build remain unverified because required
dependencies are uncached and network DNS is unavailable.
