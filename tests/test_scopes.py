"""Lexical scopes keep code boundaries and route only their defined values."""

import numpy as np
import pytest

import neurify as nf
from neurify import scope as scope_alias


def scoped(x: nf.Real[-1, 1]):
    with nf.scope("physics", trainable=False, expose=("energy",)):
        energy = x + 2
        dead = energy * 2
    with nf.scope("decision"):
        score = 3 * energy
    return score + x


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_scopes_compile_from_one_function(arch):
    net = nf.compile(scoped, arch=arch)
    assert isinstance(net, nf.ScopedNetwork)
    assert net.scope_names == ("physics", "decision")
    assert not net.regions["physics"].trainable
    assert net.exact and nf.verify(net).ok
    assert net(0.5) == 8
    X = np.array([[-1], [0], [1]], dtype=float)
    y, values = net.forward_with_intermediates(X)
    assert np.array_equal(y, 4 * X + 6)
    assert set(values["physics"]) == {"energy"}
    assert np.array_equal(values["physics"]["energy"], X[:, 0] + 2)
    assert np.array_equal(values["decision"]["score"], 3 * values["physics"]["energy"])


def test_context_is_a_noop_in_normal_python_and_does_not_swallow_exceptions():
    assert scoped(0.5) == 8
    with pytest.raises(RuntimeError, match="inside"):
        with nf.scope("test"):
            raise RuntimeError("inside")


def test_code_before_after_scopes_and_reassigned_locals():
    def f(x: nf.Real[0, 2]):
        y = x + 1
        with scope_alias("first", expose=("y",)):
            y += 2
        z = y * 3
        with nf.scope("second", expose=("y", "z")):
            y = z - 1
            z = y + 2
        result = y + z + x
        return result

    net = nf.compile(f)
    assert nf.verify(net).ok
    assert net(1) == f(1)
    _, values = net.forward_with_intermediates([1])
    assert values["first"]["y"] == 4
    assert values["second"]["y"] == 11
    assert values["second"]["z"] == 13


def test_loops_branches_and_exposed_activations_inside_a_scope():
    def f(n: nf.Int[0, 4]):
        with nf.scope("loop", expose=("total", "diagnostic")):
            total = 0
            for i in range(n):
                total += i
            if n > 2:
                diagnostic = total + 10
            else:
                diagnostic = -total
        return total, diagnostic

    net = nf.compile(f)
    assert nf.verify(net).ok
    _, values = net.forward_with_intermediates([[0], [2], [4]])
    assert np.array_equal(values["loop"]["total"], [0, 1, 6])
    assert np.array_equal(values["loop"]["diagnostic"], [0, -1, 16])


def test_exposing_does_not_reorder_program_variables():
    def f(x: nf.Int[0, 2]):
        with nf.scope("values", expose=("z", "a")):
            z = x + 1
            a = x - 1
        return a, z

    net = nf.compile(f)
    y, obs = net.forward_with_intermediates([1], structured=False)
    assert np.array_equal(y, [0, 2])
    assert np.array_equal(obs["values"], [2, 0])
    assert nf.verify(net).ok


def test_unused_inputs_static_parameters_and_constant_reassignment():
    def f(x, unused, k=2):
        with nf.scope("constant", expose=("k",)):
            k = k + 1
        with nf.scope("score"):
            score = k * x
        return {"result": score, "k": k, "unused": unused}

    net = nf.compile(f, inputs={"x": nf.Real[-1, 1], "unused": nf.Real[0, 1]})
    assert net.n_in == 2
    assert nf.verify(net).ok
    assert net(0.5, 1, cast=True) == {"result": 1.5, "k": 3, "unused": 1}


def test_static_return_and_structured_fixed_wiring():
    def f(x: nf.Int[0, 2], k=(2, 3)):
        with nf.scope("body"):
            y = x + 1
        return {"pair": k, "result": [y, None, 7]}

    net = nf.compile(f)
    assert nf.verify(net).ok
    assert net(1, cast=True) == {"pair": (2, 3), "result": [2, None, 7]}


def test_zero_input_scope_and_empty_batches():
    def f():
        with nf.scope("constant", trainable=False):
            x = 3
        return x

    net = nf.compile(f)
    assert net() == 3 and nf.verify(net).ok
    assert len(net.steps) == 1  # returning a local is fixed wiring, not another trainable network
    assert np.array_equal(net.forward(np.empty((3, 0))), np.full((3, 1), 3))
    assert net.forward_with_intermediates(np.empty((0, 0)))[0].shape == (0, 1)


def test_constants_remain_float_when_the_python_value_is_float():
    def f(n: nf.Int[0, 3]):
        with nf.scope("convert"):
            value = n / 0.5
        with nf.scope("type"):
            result = isinstance(value, float)
        return result

    net = nf.compile(f)
    assert nf.verify(net).ok
    assert net(1, cast=True) is True


def test_snapshots_training_transforms_and_empty_exposure():
    def f(x: nf.Real[0, 1]):
        with nf.scope("silent", expose=()):
            y = x + 1
        return y

    net = nf.compile(f, trainable={"width": 8})
    assert net.forward_with_intermediates([0])[1] == {"silent": {}}
    assert net.forward_with_intermediates([[0]], structured=False)[1]["silent"].shape == (1, 0)
    assert nf.verify(net).ok
    original = nf.compile(scoped)
    prepared = original.prepare_for_training(width=8, depth=2)
    assert not prepared.regions["physics"].trainable
    assert nf.verify(prepared).ok


@pytest.mark.parametrize("name,options", [
    ("", {}), ("__neurify_reserved", {}), ("x", {"trainable": "false"}),
    ("x", {"differentiable": 1}), ("x", {"expose": "x"}), ("x", {"expose": ("x", "x")}),
])
def test_invalid_context_options(name, options):
    with pytest.raises((TypeError, ValueError)):
        nf.scope(name, **options)


def generated(source):
    env = {"nf": nf}
    exec(source, env)
    fn = env["f"]
    fn.__neurify_source__ = source
    return fn


@pytest.mark.parametrize("body,match", [
    ('with nf.scope("a"):\n        with nf.scope("b"):\n            y = x\n    return y', "top-level"),
    ('for i in range(2):\n        with nf.scope("a"):\n            x += 1\n    return x', "top-level"),
    ('with nf.scope("a"):\n        if x > 0:\n            return x\n        y = -x\n    return y', "early returns"),
    ('with nf.scope("a", expose=("x",)):\n        y = x + 1\n    return y', "assigned inside"),
    ('with nf.scope("a"):\n        y = x + 1\n    with nf.scope("a"):\n        z = y + 1\n    return z', "duplicate"),
    ('with nf.scope("a"):\n        y = [x]\n    return y', "scalars"),
    ('with nf.scope("a") as ctx:\n        y = x + 1\n    return y', "without an 'as'"),
    ('with nf.scope(str(x)):\n        y = x + 1\n    return y', "literals"),
    ('with nf.scope("a"):\n        y = x + 1\n    return x', "no live outputs"),
])
def test_unsupported_scope_constructs_fail_clearly(body, match):
    fn = generated("def f(x):\n    " + body + "\n")
    with pytest.raises((nf.CompileError, nf.TraceError), match=match):
        nf.compile(fn, inputs={"x": nf.Int[0, 2]})


def test_container_inputs_and_output_encodings_are_explicitly_unsupported():
    with pytest.raises(nf.CompileError, match="scalar"):
        nf.compile(scoped, inputs={"x": nf.Vec(nf.Real[-1, 1], 2)})
    with pytest.raises(nf.CompileError, match="output='raw'"):
        nf.compile(scoped, output="rank")


def test_exposed_variables_cannot_be_missing_on_some_paths():
    def f(x: nf.Int[0, 2]):
        with nf.scope("partial", expose=("diagnostic",)):
            if x > 0:
                diagnostic = x + 1
            result = x * 2
        return result

    with pytest.raises(nf.CompileError, match="defined on every path"):
        nf.compile(f)


def test_comprehension_targets_do_not_overwrite_bypassing_locals():
    def f(i: nf.Real[-1, 1]):
        with nf.scope("comprehension", differentiable=False):
            result = sum([i for i in range(3)])
        return i + result

    net = nf.compile(f)
    assert net.steps[0].writes == ("result",)
    assert nf.verify(net).ok
    assert net(0.5) == 3.5


def test_torch_stopped_scope_preserves_bypassing_gradient_path():
    torch = pytest.importorskip("torch")

    def f(x: nf.Real[-1, 1]):
        with nf.scope("stopped", differentiable=False):
            y = 3 * x
        return x + y

    model = nf.compile(f).to_torch(dtype=torch.float64)
    x = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    y, obs = model.forward_with_intermediates(x, detach=False)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert not obs["stopped"]["y"].requires_grad
    assert all(p.grad is None for p in model.regions["stopped"].parameters())


def test_torch_frozen_scope_and_fixed_return_do_not_add_trainable_parameters():
    torch = pytest.importorskip("torch")

    def f(x: nf.Real[-1, 1]):
        with nf.scope("frozen", trainable=False):
            y = 3 * x
        return y

    model = nf.compile(f).to_torch(dtype=torch.float64)
    assert not any(p.requires_grad for p in model.parameters())
    x = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 3))


def test_torch_live_scope_observations_and_training_changes():
    torch = pytest.importorskip("torch")
    net = nf.compile(scoped)
    model = net.to_torch(dtype=torch.float64)
    x = torch.tensor([[0.5]], dtype=torch.float64)
    _, before = model.forward_with_intermediates(x)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        y, live = model.forward_with_intermediates(x, detach=False)
        (y.square().mean() + live["decision"]["score"].square().mean()).backward()
        opt.step()
    _, after = model.forward_with_intermediates(x)
    assert torch.equal(before["physics"]["energy"], after["physics"]["energy"])
    assert not torch.equal(before["decision"]["score"], after["decision"]["score"])


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_backend_shapes_and_observations(arch):
    torch = pytest.importorskip("torch")
    net = nf.compile(scoped, arch=arch)
    model = net.to_torch(dtype=torch.float64)
    for x in (np.array([0.5]), np.array([[0], [0.5]]), np.empty((0, 1))):
        y, obs = model.forward_with_intermediates(torch.tensor(x, dtype=torch.float64))
        want, reference = net.forward_with_intermediates(x)
        assert np.allclose(y.detach().numpy(), want)
        assert np.allclose(obs["physics"]["energy"].numpy(), reference["physics"]["energy"])


def test_jax_scope_controls_jit_and_bypassing_derivatives():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def f(x: nf.Real[-1, 1]):
        with nf.scope("blocked", trainable=False, differentiable=False):
            y = 3 * x
        return x + y

    net = nf.compile(f)
    apply, params = net.to_jax()
    assert "blocked" not in params
    x = jnp.array([[0.5]])
    assert np.allclose(jax.jit(apply)(params, x), [[2]])
    assert np.allclose(jax.grad(lambda z: apply(params, z).sum())(x), 1)
    y, obs = jax.jit(apply.with_intermediates)(params, x)
    assert np.allclose(obs["blocked"]["y"], [1.5])


def test_jax_frozen_scope_preserves_input_derivatives():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def f(x: nf.Real[-1, 1]):
        with nf.scope("frozen", trainable=False):
            y = 3 * x
        return y

    apply, params = nf.compile(f).to_jax()
    assert params == {}
    x = jnp.array([[0.5]])
    assert np.allclose(jax.grad(lambda z: apply(params, z).sum())(x), 3)
