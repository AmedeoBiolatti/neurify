"""Real optimizer checks for energy-only and scoring-only learning."""

import pytest

import neurify as nf
from examples.train_scopes import program, run_experiment


@pytest.mark.parametrize("arch", nf.ARCHS)
@pytest.mark.parametrize("mode", ["energy-only", "scoring-only"])
def test_only_selected_scope_learns_from_final_output(mode, arch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    result = run_experiment(mode, arch=arch)
    assert result["frozen_state_bitwise_equal"]
    assert result["max_parameter_changes"][result["frozen"]] == 0
    assert result["max_parameter_changes"][result["trainable"]] > 0
    assert result["activation_gradient_max_error"] < 1e-12


def test_live_energy_loss_trains_the_same_activation_used_for_scoring():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    result = run_experiment("energy-loss")
    assert result["final_mse"] < 1e-7
    assert result["final_maps"]["scoring"] == {"slope": 3.0, "intercept": 1.0}


def test_frozen_scoring_must_remain_differentiable_for_energy_output_loss():
    torch = pytest.importorskip("torch")
    model = nf.compile(program).to_torch(dtype=torch.float64)
    model.set_trainable("energy", True).set_trainable("scoring", False)
    x = torch.tensor([-0.5, 0.5], dtype=torch.float64)
    target = 2 * x + 1
    assert (model.call(x=x) - target).square().mean().requires_grad
    model.set_differentiable("scoring", False)
    assert not (model.call(x=x) - target).square().mean().requires_grad
    # An upstream live activation loss can still train energy through its
    # actual producer while the final scoring output is stopped.
    live = model.as_function(detach=False).inspect(x=x)
    energy_loss = live.exposed["energy"]["energy"].square().mean()
    assert energy_loss.requires_grad
    energy_loss.backward()
    assert all(p.grad is None for p in model.regions["scoring"].parameters())
    assert any(p.grad is not None for p in model.regions["energy"].parameters())


def test_freezing_with_existing_adamw_momentum_keeps_state_unchanged():
    torch = pytest.importorskip("torch")
    model = nf.compile(program).to_torch(dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    target = 2 * x + 1
    (model.call(x=x) - target).square().mean().backward()
    optimizer.step()
    model.set_trainable("scoring", False)
    frozen = {name: value.detach().clone() for name, value in model.regions["scoring"].state_dict().items()}
    energy_before = [p.detach().clone() for p in model.regions["energy"].parameters()]
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        (model.call(x=x) - target).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(value, frozen[name]) for name, value in model.regions["scoring"].state_dict().items())
    assert any(not torch.equal(a, b) for a, b in zip(energy_before, model.regions["energy"].parameters()))


def test_runtime_scope_switch_preserves_the_newly_frozen_parameters():
    torch = pytest.importorskip("torch")
    model = nf.compile(program).to_torch(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    target = 2 * x + 1
    model.set_trainable("scoring", False)
    (model.call(x=x) - target).square().mean().backward()
    optimizer.step()
    model.set_trainable("energy", False).set_trainable("scoring", True)
    frozen = [p.detach().clone() for p in model.regions["energy"].parameters()]
    before = [p.detach().clone() for p in model.regions["scoring"].parameters()]
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        (model.call(x=x) - target).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(frozen, model.regions["energy"].parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.regions["scoring"].parameters()))
