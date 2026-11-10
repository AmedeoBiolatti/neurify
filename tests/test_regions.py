"""Preserved code boundaries, numerical contracts, gradients, and probes."""

import numpy as np
import pytest

import neurify as nf


def features(x: nf.Real[-1, 1]):
    return x + 2


def physics(z: nf.Real[1, 3]):
    return 3 * z


def head(z: nf.Real[3, 9]):
    return 2 * z


def make_pipeline(arch="mlp", *, frozen=False, stopped=False):
    return nf.pipeline(
        nf.region(features, arch=arch),
        nf.region(physics, trainable=not frozen, differentiable=not stopped, arch=arch),
        nf.region(head, arch=arch),
    )


@pytest.mark.parametrize("arch", nf.ARCHS)
@pytest.mark.parametrize("frozen,stopped", [(False, False), (True, False), (False, True), (True, True)])
def test_controls_preserve_initial_function(arch, frozen, stopped):
    net = make_pipeline(arch, frozen=frozen, stopped=stopped)
    X = np.array([[-1], [-0.25], [0], [1]], dtype=float)
    want = 6 * X + 12
    assert net.exact
    assert np.array_equal(net.forward(X), want)
    assert np.array_equal(net.reference(X)[0], want)
    assert net(0.5) == 15
    assert nf.verify(net).ok


def test_probes_are_values_consumed_by_the_next_region():
    net = make_pipeline()
    X = np.array([[-1], [0], [1]], dtype=float)
    y, values = net.forward_with_intermediates(X)
    assert tuple(values) == ("features", "physics", "head")
    assert np.array_equal(values["features"], X + 2)
    assert np.array_equal(values["physics"], 3 * values["features"])
    assert np.array_equal(values["head"], y)
    values["features"][:] = 100
    assert np.array_equal(net.forward_with_intermediates(X)[1]["features"], X + 2)
    y, values = net.forward_with_intermediates([0])
    assert y.shape == (1,) and values["physics"].shape == (1,)


def test_nested_outputs_connect_in_flattened_order():
    def pair(x: nf.Int[0, 3], flip: nf.Bool):
        return {"amount": 2 * x, "flags": [flip]}

    def combine(amount: nf.Int[0, 6], flag: nf.Bool):
        return amount + flag, amount - flag

    net = nf.pipeline([nf.region(pair), nf.region(combine)])
    assert net.exact and nf.verify(net).ok
    assert net(3, True, cast=True) == (7, 5)
    assert np.array_equal(net.forward_with_intermediates([3, 1])[1]["pair"], [6, 1])
    _, probes = net.forward_with_intermediates([[3, 1]], structured=True)
    assert np.array_equal(probes["pair"]["amount"], [6])
    assert np.array_equal(probes["pair"]["flags"][0], [1])
    assert isinstance(probes["combine"], tuple)
    assert np.array_equal(probes["combine"][0], [7])


def test_decorator_and_existing_network():
    decorated = nf.region(trainable=False)(features)
    assert decorated(0) == 2
    assert decorated.name == "features"
    assert not decorated.trainable and decorated.differentiable
    compiled = nf.compile(head)
    wrapped = nf.region(compiled, name="output")
    assert wrapped.network is compiled
    changed = wrapped.with_controls(differentiable=False)
    assert wrapped.differentiable and not changed.differentiable
    assert changed.network is wrapped.network
    assert nf.pipeline(wrapped).region_names == ("output",)
    with pytest.raises(TypeError, match="already compiled"):
        nf.region(compiled, arch="glu")


def test_pipeline_validation():
    a = nf.region(features)
    for args in [(), (features,)]:
        with pytest.raises(TypeError, match="Region"):
            nf.pipeline(*args)
    with pytest.raises(ValueError, match="unique"):
        nf.pipeline(a, a)
    with pytest.raises(ValueError, match="non-empty"):
        nf.region(a.network, name="")
    with pytest.raises(TypeError, match="bool"):
        nf.region(a.network, trainable="false")

    def pair(x: nf.Real[-1, 1]):
        return x, x

    with pytest.raises(ValueError, match="produces 2"):
        nf.pipeline(nf.region(pair), a)


def test_regular_compilation_does_not_silently_discard_region_controls():
    controlled = nf.region(features, trainable=False)

    def f(x: nf.Real[-1, 1]):
        return controlled(x)

    with pytest.raises(nf.TraceError, match="pipeline"):
        nf.compile(f)


def test_unproven_domain_composition_is_reported():
    def too_narrow(x: nf.Real[0, 1]):
        return abs(x)

    net = nf.pipeline(nf.region(features), nf.region(too_narrow))
    assert not net.exact
    assert any("domain" in w for w in net.warnings)


def test_proven_discrete_domain_mismatch_is_rejected():
    def shifted(x: nf.Int[0, 2]):
        return x + 2

    def narrow(x: nf.Int[0, 1]):
        return x

    with pytest.raises(ValueError, match="outside"):
        nf.pipeline(nf.region(shifted), nf.region(narrow))


def test_sequence_attention_and_output_encodings_compose():
    def maximum(xs: nf.Seq(nf.Int[0, 2], 3)):
        return max(xs)

    def ranks(x: nf.Int[0, 2]):
        return [x, 1]

    net = nf.pipeline(
        nf.region(maximum, arch="transformer", attention="relu"),
        nf.region(ranks, output="rank"),
    )
    assert net.exact and nf.verify(net).ok
    y, probes = net.forward_with_intermediates([0, 2, 1])
    assert np.array_equal(probes["maximum"], [2])
    assert np.array_equal(y, [1, 0])


@pytest.mark.parametrize("shape", [(2,), (2, 2), (1, 1, 1)])
def test_input_shape_validation(shape):
    with pytest.raises(ValueError, match="shape"):
        make_pipeline().forward(np.zeros(shape))


def test_empty_batches_and_constant_regions():
    net = make_pipeline()
    output, values = net.forward_with_intermediates(np.empty((0, 1)))
    assert output.shape == (0, 1)
    assert all(v.shape == (0, 1) for v in values.values())

    def constant():
        return 4

    def scale(x: nf.Int[4, 4]):
        return x * 2

    net = nf.pipeline(nf.region(constant), nf.region(scale))
    assert net.exact and net() == 8
    assert nf.verify(net).ok
    assert np.array_equal(net.forward(np.empty((3, 0))), np.full((3, 1), 8))


def test_training_preparation_preserves_controls_and_observations():
    net = make_pipeline(frozen=True, stopped=True)
    prepared = net.prepare_for_training(width=8, depth=2)
    assert prepared.num_params > net.num_params
    assert not prepared.regions["physics"].trainable
    assert not prepared.regions["physics"].differentiable
    for name, values in net.forward_with_intermediates([[0], [1]])[1].items():
        assert np.allclose(prepared.forward_with_intermediates([[0], [1]])[1][name], values)
    assert "trainable=False, differentiable=False" in prepared.summary()


def test_bounds_fallback_is_conservative_for_real_relu_networks():
    def absolute(x: nf.Real[-2, 2]):
        return abs(x)

    def scale(x: nf.Real[0, 8]):
        return x * 2

    for arch in ("mlp", "resmlp", "glu"):
        net = nf.pipeline(nf.region(absolute, arch=arch), nf.region(scale, arch=arch))
        assert net.exact and nf.verify(net).ok


def test_torch_freezing_preserves_gradient_flow_to_earlier_regions():
    torch = pytest.importorskip("torch")
    model = make_pipeline(frozen=True).to_torch(dtype=torch.float64)
    x = torch.tensor([[0.0], [0.5]], dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 6))
    assert all(p.grad is None and not p.requires_grad for p in model.regions["physics"].parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.regions["features"].parameters())


def test_torch_stopping_gradients_leaves_downstream_learning_enabled():
    torch = pytest.importorskip("torch")
    model = make_pipeline(stopped=True).to_torch(dtype=torch.float64)
    x = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is None
    assert all(p.grad is None for name in ("features", "physics") for p in model.regions[name].parameters())
    assert any(p.grad is not None for p in model.regions["head"].parameters())


def test_torch_inspection_tracks_training_and_keeps_snapshots():
    torch = pytest.importorskip("torch")
    net = nf.pipeline(nf.region(features, trainable=False), nf.region(physics, trainable=False), nf.region(head))
    model = net.to_torch(dtype=torch.float64)
    x = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
    _, before = model.forward_with_intermediates(x)
    saved = {name: value.clone() for name, value in before.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    y, after = model.forward_with_intermediates(x)
    assert torch.equal(y, after["head"])
    assert torch.equal(before["physics"], after["physics"])
    assert not torch.equal(before["head"], after["head"])
    assert all(torch.equal(before[name], saved[name]) and not before[name].requires_grad for name in before)
    assert torch.equal(after["physics"], model.regions["physics"](after["features"]))


def test_torch_live_probes_support_auxiliary_losses():
    torch = pytest.importorskip("torch")
    model = make_pipeline().to_torch(dtype=torch.float64)
    x = torch.tensor([[0.0]], dtype=torch.float64)
    _, values = model.forward_with_intermediates(x, detach=False)
    values["physics"].sum().backward()
    assert any(p.grad is not None for p in model.regions["features"].parameters())
    assert all(p.grad is None for p in model.regions["head"].parameters())


def test_torch_named_variable_views_preserve_gradients():
    torch = pytest.importorskip("torch")

    def variables(x: nf.Real[-1, 1]):
        return {"shifted": x + 2, "other": [x - 1]}

    def combine(a: nf.Real[1, 3], b: nf.Real[-2, 0]):
        return a + b

    model = nf.pipeline(nf.region(variables), nf.region(combine)).to_torch(dtype=torch.float64)
    x = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    _, values = model.forward_with_intermediates(x, detach=False, structured=True)
    values["variables"]["shifted"].sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_torch_runtime_controls_clear_gradients_and_roundtrip_checkpoints():
    torch = pytest.importorskip("torch")
    model = make_pipeline().to_torch(dtype=torch.float64)
    model(torch.tensor([[0.0]], dtype=torch.float64)).sum().backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    frozen = [p.detach().clone() for p in model.regions["physics"].parameters()]
    model.set_trainable("physics", False).set_differentiable("physics", False)
    assert all(p.grad is None for p in model.regions["physics"].parameters())
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(frozen, model.regions["physics"].parameters()))
    state = model.state_dict()
    restored = make_pipeline().to_torch(dtype=torch.float64)
    restored.load_state_dict(state)
    assert restored.get_extra_state() == model.get_extra_state()
    assert all(not p.requires_grad for p in restored.regions["physics"].parameters())
    with pytest.raises(KeyError, match="unknown region"):
        restored.set_trainable("missing")
    with pytest.raises(TypeError, match="bool"):
        restored.set_differentiable("head", 0)


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_torch_exports_match_numpy_and_accept_single_samples(arch):
    torch = pytest.importorskip("torch")
    net = make_pipeline(arch)
    model = net.to_torch(dtype=torch.float64)
    for x in (np.array([0.25]), np.array([[0], [0.5]]), np.empty((0, 1))):
        y, values = model.forward_with_intermediates(torch.tensor(x, dtype=torch.float64))
        expected, probes = net.forward_with_intermediates(x)
        assert np.allclose(y.detach().numpy(), expected)
        assert all(np.allclose(values[name].numpy(), probes[name]) for name in probes)


def test_jax_frozen_parameters_are_excluded_and_input_gradients_flow():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    net = make_pipeline(frozen=True)
    apply, params = net.to_jax()
    assert set(params) == {"features", "head"}
    x = jnp.array([[0.0], [0.5]])
    assert np.allclose(jax.jit(apply)(params, x), net.forward(np.asarray(x)))
    assert np.allclose(jax.grad(lambda z: apply(params, z).sum())(x), 6)
    gradients = jax.grad(lambda p: apply(p, x).sum())(params)
    assert any(np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(gradients["features"]))
    updated = jax.tree_util.tree_map(lambda p, g: p - 0.001 * g - 0.001 * p, params, gradients)
    _, values = jax.jit(apply.with_intermediates)(updated, x)
    assert np.allclose(values["physics"], 3 * values["features"])


def test_jax_stop_gradient_and_live_observations():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    apply, params = make_pipeline(stopped=True).to_jax()
    x = jnp.array([[0.0]])
    assert np.allclose(jax.grad(lambda z: apply(params, z).sum())(x), 0)
    grads = jax.grad(lambda p: apply(p, x).sum())(params)
    assert all(not np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads["features"]))
    assert all(not np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads["physics"]))
    assert any(np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads["head"]))
    y, probes = jax.jit(apply.with_intermediates)(params, x)
    assert np.array_equal(y, probes["head"])

    apply, params = make_pipeline().to_jax()
    grads = jax.grad(lambda p: apply.with_intermediates(p, x, detach=False)[1]["physics"].sum())(params)
    assert any(np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads["features"]))
    assert all(not np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads["head"]))


def test_jax_named_variable_views_preserve_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def variables(x: nf.Real[-1, 1]):
        return {"shifted": x + 2, "other": [x - 1]}

    apply, params = nf.pipeline(nf.region(variables)).to_jax()
    x = jnp.array([[0.0]])
    inspect = jax.jit(lambda p, x: apply.with_intermediates(p, x, structured=True, detach=False))
    assert np.allclose(jax.grad(lambda z: inspect(params, z)[1]["variables"]["shifted"].sum())(x), 1)


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_jax_exports_match_numpy_single_and_batched(arch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    net = make_pipeline(arch)
    apply, params = net.to_jax()
    for x in (np.array([0.25]), np.array([[0], [0.5]]), np.empty((0, 1))):
        y, values = jax.jit(apply.with_intermediates)(params, jnp.asarray(x))
        expected, probes = net.forward_with_intermediates(x)
        assert np.allclose(y, expected)
        assert all(np.allclose(values[name], probes[name]) for name in probes)
