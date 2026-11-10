"""User-facing block configuration, structural controls, and scoped corrections."""

from dataclasses import FrozenInstanceError
import io

import numpy as np
import pytest

import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score


def internal(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("a", "energy")):
        a = x + 2
        energy = 2 * a
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_architecture_override_does_not_change_other_blocks(arch):
    net = nf.compile(program, arch="mlp", blocks={"energy": nf.Block(arch=arch)})
    assert net.blocks["energy"].network.steps[0].region.network.arch == arch
    assert net.blocks["scoring"].network.steps[0].region.network.arch == "mlp"
    assert nf.verify(net).ok
    assert net.inspect(.5).exposed["energy"]["energy"] == 2.5
    assert net(.5) == 8.5


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_additional_capacity_preserves_values_and_selected_ownership(arch):
    plain = nf.compile(internal, arch=arch)
    net = nf.compile(internal, arch=arch, blocks={"energy": nf.Block(extra_width=8, extra_depth=2, seed=5)})
    assert nf.verify(net).ok
    assert net.blocks["energy"].network.depth == plain.blocks["energy"].network.depth + 2
    assert net.blocks["scoring"].network.num_params == plain.blocks["scoring"].network.num_params
    assert net.num_params > plain.num_params
    assert net.inspect(.5) == plain.inspect(.5)


def test_mixed_architecture_handles_a_real_product_in_the_selected_glu_block():
    def quadratic(x: nf.Real[-1, 1]):
        with nf.scope("energy", expose=("energy",)):
            energy = x * x
        with nf.scope("scoring"):
            score = 3 * energy + 1
        return score

    net = nf.compile(quadratic, blocks={"energy": nf.Block(arch="glu")})
    x = np.linspace(-1, 1, 65)
    assert np.allclose(net(x), 3 * x*x + 1, atol=1e-12, rtol=1e-12)
    assert nf.verify(net).ok


def test_explicit_overrides_inherit_scope_defaults_and_survive_training_transforms():
    def f(x: nf.Real[-1, 1]):
        with nf.scope("energy", trainable=False, differentiable=False):
            y = x + 1
        return y

    inherited = nf.compile(f, blocks={"energy": nf.Block(extra_width=4)})
    assert not inherited.blocks["energy"].trainable
    assert not inherited.blocks["energy"].differentiable
    config = nf.Block(trainable=True, differentiable=True)
    changed = nf.compile(f, blocks={"energy": config}).prepare_for_training(width=8)
    assert changed.blocks["energy"].trainable and changed.blocks["energy"].differentiable
    assert changed.block_configs["energy"] == config
    with pytest.raises(TypeError):
        changed.block_configs["other"] = config


@pytest.mark.parametrize("kwargs", [
    {"arch": "wrong"}, {"extra_width": -1}, {"extra_width": True},
    {"extra_depth": 1.5}, {"seed": -1}, {"residual": (0,)},
    {"residual": (True,)}, {"residual": "32"}, {"trainable": "false"},
    {"differentiable": 0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        nf.Block(**kwargs)


def test_configuration_is_immutable_and_copies_hidden_widths():
    hidden = [8, 8]
    config = nf.Block(residual=hidden)
    hidden[0] = 32
    assert config.residual == (8, 8)
    with pytest.raises(FrozenInstanceError):
        config.arch = "glu"


@pytest.mark.parametrize("blocks", [{"typo": nf.Block()}, {"energy": {}}, [], {"": nf.Block()}])
def test_block_mapping_errors_do_not_silently_ignore_settings(blocks):
    with pytest.raises((TypeError, ValueError)):
        nf.compile(program, blocks=blocks)


def test_unscoped_function_rejects_named_block_settings_and_decorator_works():
    def f(x: nf.Real[-1, 1]):
        return x + 1

    with pytest.raises(ValueError, match="named neurify scopes"):
        nf.compile(f, blocks={"energy": nf.Block()})
    decorated = nf.compile(blocks={"energy": nf.Block(trainable=False)})(program)
    assert not decorated.blocks["energy"].trainable


@pytest.mark.parametrize("hidden", [(), (8,), (8, 8)])
def test_residual_initial_function_metadata_and_global_preparation(hidden):
    net = nf.compile(internal, blocks={"energy": nf.Block(residual=hidden)})
    assert nf.verify(net).ok
    assert net.inspect(.5) == nf.compile(internal).inspect(.5)
    info = net.report["blocks"][0]
    block = net.blocks["energy"].network
    assert block.depth == max(block.base.depth, len(hidden))
    assert info["mode"] == "residual"
    assert info["trainable_params"] == block.correction_params
    assert net.num_params == sum(b.network.num_params for b in net.regions.values())
    assert "arch=mlp" in net.summary() and "mode=residual" in net.summary()
    prepared = net.prepare_for_training(width=8)
    assert prepared.block_configs == net.block_configs
    assert nf.verify(prepared).ok


def _torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return torch


def test_train_only_is_atomic_preserves_gradient_policies_and_freezes_generated_head():
    torch = _torch()

    def f(x: nf.Real[-1, 1]):
        with nf.scope("energy"):
            energy = x + 2
        with nf.scope("scoring"):
            score = 3 * energy
        return score + x

    model = nf.compile(f).to_torch(dtype=torch.float64)
    assert model.train_only("energy") is model
    assert model._trainable == {"energy": True, "scoring": False, "__neurify_return": False}
    assert all(model._differentiable.values())
    before = dict(model._trainable)
    with pytest.raises(KeyError):
        model.train_only("scoring", "typo")
    assert model._trainable == before
    with pytest.raises(TypeError):
        model.train_only(["energy"])
    assert model._trainable == before
    model.train_only()
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("selected", ["energy", "scoring"])
def test_train_only_residual_fits_target_and_never_unfreezes_compiled_base(selected):
    torch = _torch()
    model = nf.compile(program, blocks={selected: nf.Block(residual=())}).to_torch(dtype=torch.float64)
    model.train_only(selected)
    residual = model.blocks[selected]
    assert all(not p.requires_grad for p in residual.base.parameters())
    assert all(p.requires_grad for p in residual.delta.parameters())
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    x = torch.linspace(-1, 1, 65, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=0)
    function = model.as_function(detach=False)
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        loss = (function(x=x) - (2*x+1)).square().mean()
        loss.backward()
        optimizer.step()
    assert (function(x=x) - (2*x+1)).square().mean().item() < 1e-10
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name]) and p.grad is None
    model.train_only().train_only(selected)
    assert all(not p.requires_grad for p in residual.base.parameters())
    assert all(p.requires_grad for p in residual.delta.parameters())


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_residual_probes_are_live_after_correction_and_internal_probes_still_feed_loss(shape):
    torch = _torch()
    model = nf.compile(internal, blocks={"energy": nf.Block(residual=())}).to_torch(dtype=torch.float64)
    model.train_only("energy")
    # Deliberately change the correction to distinguish it from base probes.
    with torch.no_grad():
        model.blocks["energy"].delta[-1].bias.fill_(1)
    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    result = model.as_function(detach=False).inspect(x=x)
    a, energy = (result.exposed["energy"][n] for n in ("a", "energy"))
    assert torch.allclose(energy, 2 * a + 1)
    assert torch.allclose(result.output, 3 * energy + 1)
    a.retain_grad()
    energy.retain_grad()
    result.output.sum().backward()
    assert torch.equal(energy.grad, torch.full_like(energy, 3))
    assert torch.equal(a.grad, torch.full_like(a, 6))
    assert torch.equal(x.grad, torch.full_like(x, 6))


def test_frozen_residual_scoring_still_passes_gradients_to_energy():
    torch = _torch()
    model = nf.compile(program, blocks={"scoring": nf.Block(residual=(8,))}).to_torch(dtype=torch.float64)
    model.train_only("energy")
    model.call(x=torch.tensor(.5, dtype=torch.float64)).backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.blocks["energy"].parameters())
    assert all(p.grad is None for p in model.blocks["scoring"].parameters())


def test_residual_checkpoint_flags_and_parameter_counts():
    torch = _torch()
    net = nf.compile(program, blocks={"scoring": nf.Block(residual=(8,))})
    model = net.to_torch(dtype=torch.float64).train_only("scoring")
    assert sum(p.numel() for p in model.parameters()) == net.num_params
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == net.report["blocks"][1]["trainable_params"]
    with torch.no_grad():
        model.blocks["scoring"].delta[-1].bias.fill_(2)
    model.set_differentiable("energy", False)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = net.to_torch(dtype=torch.float64)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(restored.call(x=.5), model.call(x=.5))
    assert not restored._differentiable["energy"]
    assert all(not p.requires_grad for p in restored.blocks["scoring"].base.parameters())
    assert "correction_hidden=(8,)" in restored.summary()
    assert "trainable_params=0" in restored.summary()
    full = io.BytesIO()
    torch.save(restored, full)
    full.seek(0)
    loaded = torch.load(full, weights_only=False)
    assert torch.equal(loaded.call(x=.5), restored.call(x=.5))


def test_multiple_corrected_live_outputs_keep_their_own_loss_paths():
    torch = _torch()

    def f(x: nf.Real[-1, 1]):
        with nf.scope("pair", expose=("a", "b")):
            a = x + 2
            b = 2 * a
        with nf.scope("scoring", expose=("score",)):
            score = a + 3 * b
        return {"score": score, "pair": (a, b)}

    model = nf.compile(f, blocks={"pair": nf.Block(residual=())}).to_torch(dtype=torch.float64)
    model.train_only("pair")
    with torch.no_grad():
        model.blocks["pair"].delta[-1].bias.copy_(torch.tensor([1., -1.], dtype=torch.float64))
    result = model.as_function(detach=False).inspect(x=.5)
    a, b = result.exposed["pair"].values()
    assert a.item() == 3.5 and b.item() == 4
    assert torch.equal(result.output["pair"][0], a)
    assert torch.equal(result.output["pair"][1], b)
    a.retain_grad()
    b.retain_grad()
    result.output["score"].backward()
    assert a.grad.item() == 1 and b.grad.item() == 3


def test_residual_hooks_snapshots_and_empty_batches():
    torch = _torch()
    model = nf.compile(program, blocks={"energy": nf.Block(residual=(4,))}).to_torch(dtype=torch.float64)
    seen = []
    handle = model.blocks["energy"].register_forward_hook(lambda *args: seen.append(True))
    result = model.inspect(x=torch.zeros(2, dtype=torch.float64, requires_grad=True))
    assert seen == [True] and not result.exposed["energy"]["energy"].requires_grad
    handle.remove()
    assert model.call(x=torch.empty((0, 2), dtype=torch.float64)).shape == (0, 2)


def test_zero_input_residual_block():
    torch = _torch()

    def constant():
        with nf.scope("value", expose=("y",)):
            y = 3
        return y

    model = nf.compile(constant, blocks={"value": nf.Block(residual=())}).to_torch(dtype=torch.float64)
    assert model.call().item() == 3
    model.call().backward()
    assert model.blocks["value"].delta[-1].bias.grad.item() == 1


def test_jax_residual_params_exclude_base_and_probes_follow_current_correction():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    net = nf.compile(program, blocks={"energy": nf.Block(trainable=False), "scoring": nf.Block(residual=())})
    apply, params = net.to_jax()
    assert tuple(params) == ("scoring",)
    assert sum(v.size for v in jax.tree_util.tree_leaves(params)) == 2
    params["scoring"][0]["b"] = jnp.ones_like(params["scoring"][0]["b"])
    result = jax.jit(lambda p: apply.inspect(p, x=.5))(params)
    assert np.allclose(result.output, 9.5)
    assert np.allclose(result.exposed["scoring"]["score"], result.output)
    grads = jax.grad(lambda p: apply.call(p, x=.5))(params)
    assert np.allclose(grads["scoring"][0]["b"], 1)
