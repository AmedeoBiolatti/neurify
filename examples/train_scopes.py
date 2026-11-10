"""Verify selective scope training with real Torch autograd on CPU.

Run after installing Neurify and Torch:
    python examples/train_scopes.py --output artifacts/scope_training.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _unchanged(module, before):
    import torch

    return all(torch.equal(value, before[name]) for name, value in module.state_dict().items())


def _affine_map(module):
    import torch

    with torch.no_grad():
        y = module(torch.tensor([[0.0], [1.0]], dtype=torch.float64)).flatten()
    return {"slope": (y[1] - y[0]).item(), "intercept": y[0].item()}


def run_experiment(mode, *, arch="mlp", steps=800, lr=0.05, normalize=True):
    """Fit score=2*x+1 while allowing only one named scope to learn.

    energy-only: energy must become (2/3)*x through frozen score=3*energy+1.
    scoring-only: energy stays x+2; score must become 2*energy-3.
    energy-loss: train only energy directly from its live activation loss.
    """
    import torch

    if mode not in ("energy-only", "scoring-only", "energy-loss"):
        raise ValueError("unknown training mode")
    torch.manual_seed(0)
    net = nf.compile(program, arch=arch)
    if normalize:
        net = net.prepare_for_training(normalize=True, scaling=None)
    model = net.to_torch(dtype=torch.float64)
    assert model.region_names == ("energy", "scoring"), "unexpected trainable return/probe head"
    active = "scoring" if mode == "scoring-only" else "energy"
    frozen = "energy" if active == "scoring" else "scoring"
    model.set_trainable(active, True).set_trainable(frozen, False)
    modules = model.regions
    active_parameters = tuple(modules[active].parameters())
    frozen_parameters = tuple(modules[frozen].parameters())
    assert all(p.requires_grad for p in active_parameters)
    assert all(not p.requires_grad for p in frozen_parameters)
    initial_state = {name: _state(modules[name]) for name in model.scope_names}
    initial_parameters = {
        name: {key: value.detach().clone() for key, value in module.named_parameters()}
        for name, module in modules.items()
    }
    initial_maps = {name: _affine_map(modules[name]) for name in model.scope_names}

    x = torch.linspace(-1, 1, 129, dtype=torch.float64)
    target = 2 * x + 1
    target_energy = (2 / 3) * x
    with torch.no_grad():
        before = model.inspect(x=x)
        initial_mse = (before.output - target).square().mean().item()

    # Check activation gradients analytically, including across the frozen
    # scoring map. Input gradients keep activations differentiable even when
    # the energy producer's own parameters are frozen.
    probe_x = x.detach().clone().requires_grad_(True)
    live = model.as_function(detach=False).inspect(x=probe_x)
    energy = live.exposed["energy"]["energy"]
    score = live.exposed["scoring"]["score"]
    energy.retain_grad()
    score.retain_grad()
    loss = (live.output - target).square().mean()
    loss.backward()
    expected_score_grad = 2 * (live.output.detach() - target) / len(x)
    expected_energy_grad = 3 * expected_score_grad
    assert torch.allclose(score.grad, expected_score_grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(energy.grad, expected_energy_grad, atol=1e-12, rtol=1e-12)
    assert all(p.grad is None for p in frozen_parameters)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in active_parameters)
    gradient_error = (energy.grad - expected_energy_grad).abs().max().item()
    model.zero_grad(set_to_none=True)

    # Keep frozen parameters in the optimizer as a deliberate test of the
    # public flag, rather than filtering them away in the training loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    function = model.as_function(detach=False)
    checkpoints = {0, 1, 5, 10, 25, 50, 100, 200, 400, steps}
    checkpoints.update(range(0, steps + 1, max(1, steps // 40)))
    curve = []

    def measure(step):
        with torch.no_grad():
            result = model.inspect(x=x)
            value = (result.output - target).square().mean().item()
        curve.append({"step": step, "score_mse": value})

    measure(0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if mode == "energy-loss":
            result = function.inspect(x=x)
            loss = (result.exposed["energy"]["energy"] - target_energy).square().mean()
        else:
            loss = (function(x=x) - target).square().mean()
        loss.backward()
        assert all(p.grad is None for p in frozen_parameters)
        optimizer.step()
        if step in checkpoints:
            measure(step)

    with torch.no_grad():
        after = model.inspect(x=x)
        final_mse = (after.output - target).square().mean().item()
        final_energy = after.exposed["energy"]["energy"]
        final_score = after.exposed["scoring"]["score"]
    frozen_equal = _unchanged(modules[frozen], initial_state[frozen])
    active_changed = not _unchanged(modules[active], initial_state[active])
    assert frozen_equal, "frozen scope state changed"
    assert active_changed, "selected scope did not learn"
    assert torch.equal(after.output, final_score), "probe differs from actual output"
    if active == "energy":
        assert torch.allclose(final_score, 3 * final_energy + 1, atol=1e-11, rtol=1e-11)
        assert not torch.equal(final_energy, before.exposed["energy"]["energy"])
    else:
        assert torch.equal(final_energy, before.exposed["energy"]["energy"])
    assert final_mse < 1e-7 and final_mse < initial_mse * 1e-6, f"training failed to fit: {final_mse}"

    parameter_changes = {
        name: max((value.detach() - initial_parameters[name][key]).abs().max().item()
                  for key, value in module.named_parameters())
        for name, module in modules.items()
    }
    sample_x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    with torch.no_grad():
        samples = model.inspect(x=sample_x)
    return {
        "mode": mode, "arch": arch, "steps": steps, "normalized": normalize,
        "trainable": active, "frozen": frozen, "initial_mse": initial_mse,
        "final_mse": final_mse, "frozen_state_bitwise_equal": frozen_equal,
        "active_state_changed": active_changed, "frozen_gradients_none": True,
        "activation_gradient_max_error": gradient_error,
        "max_parameter_changes": parameter_changes,
        "initial_maps": initial_maps,
        "final_maps": {name: _affine_map(modules[name]) for name in model.scope_names},
        "curve": curve,
        "samples": {"x": sample_x.tolist(), "target": (2 * sample_x + 1).tolist(),
                    "energy": samples.exposed["energy"]["energy"].tolist(),
                    "score": samples.output.tolist()},
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("energy-only", "scoring-only", "energy-loss", "all"), default="all")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--no-normalize", action="store_true", help="omit fixed input normalization (may need more steps)")
    parser.add_argument("--arch", choices=nf.ARCHS, default="mlp")
    parser.add_argument("--output", type=Path, default=Path("artifacts/scope_training.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    modes = ("energy-only", "scoring-only", "energy-loss") if args.mode == "all" else (args.mode,)
    results = [run_experiment(mode, arch=args.arch, steps=args.steps, normalize=not args.no_normalize) for mode in modes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torch": torch.__version__, "device": "cpu", "results": results}, indent=2) + "\n")
    for result in results:
        print(f"{result['mode']}: MSE {result['initial_mse']:.6g} -> {result['final_mse']:.6g}; "
              f"{result['frozen']} frozen={result['frozen_state_bitwise_equal']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
