"""Function-compatible prediction and one-pass structured inspection."""

import inspect
import io

import numpy as np
import pytest

import neurify as nf


def scoped(x: nf.Real[-1, 1], /, y: nf.Real[0, 2] = 1, *, offset: nf.Real[0, 3] = 2):
    with nf.scope("physics", expose=("energy", "diagnostic")):
        energy = x + y
        diagnostic = energy + 10
        energy = diagnostic - 10
    with nf.scope("decision"):
        score = energy + offset
    return {"score": score, "details": (energy, None)}


def test_original_signature_defaults_and_output_tree():
    net = nf.compile(scoped)
    function = net.as_function()
    assert inspect.signature(net) == inspect.signature(scoped)
    assert inspect.signature(function) == inspect.signature(scoped)
    assert function.__name__ == scoped.__name__
    assert net(0.5) == scoped(0.5)
    assert function(0.5, 0.25, offset=1) == scoped(0.5, 0.25, offset=1)
    assert np.array_equal(net.flatten_inputs(0.5), [0.5, 1, 2])
    result = nf.verify(net)
    assert result.ok and result.n_skipped == 0


def test_output_and_all_exposed_values_together():
    net = nf.compile(scoped)
    result = net.inspect(0.5)
    assert isinstance(result, nf.Evaluation)
    assert result.output == scoped(0.5)
    assert result.exposed == {
        "physics": {"energy": 1.5, "diagnostic": 11.5},
        "decision": {"score": 3.5},
    }
    output, exposed = result
    assert output == result.output and exposed == result.exposed


@pytest.mark.parametrize("args,kwargs", [
    ((), {}), ((), {"x": 0.5}), ((0.5, 1, 2), {}),
    ((0.5, 1), {"y": 1}), ((0.5,), {"unknown": 1}),
])
def test_python_argument_rules_are_enforced(args, kwargs):
    net = nf.compile(scoped)
    for call in (net, net.as_function(), net.inspect):
        with pytest.raises(TypeError):
            call(*args, **kwargs)


def test_inspection_is_a_single_network_execution(monkeypatch):
    net = nf.compile(scoped)
    calls = []
    run = net._run

    def counted(X, dtype, capture):
        calls.append(capture)
        return run(X, dtype, capture)

    monkeypatch.setattr(net, "_run", counted)
    result = net.inspect(0.5)
    assert result.output["score"] == result.exposed["decision"]["score"]
    assert calls == [True]


def test_function_adapter_reads_current_weights_not_source_python():
    guard = {"fail": False}

    def f(x: nf.Real[-1, 1]):
        if guard["fail"]:
            raise RuntimeError("Python function executed")
        return x + 1

    net = nf.compile(f)
    function = net.as_function()
    guard["fail"] = True
    with pytest.raises(RuntimeError):
        f(0.5)
    assert function(0.5) == 1.5
    assert function.inspect(0.5) == nf.Evaluation(1.5, {})


def test_existing_adapter_and_inspection_follow_weight_updates():
    net = nf.compile(scoped)
    function = net.as_function()
    before = function.inspect(0.5)
    net.regions["physics"].network.steps[0].region.network.weights()["out"]["bias"] += 1
    after = function.inspect(0.5)
    assert after.exposed["physics"]["energy"] == before.exposed["physics"]["energy"] + 1
    assert after.output["score"] == before.output["score"] + 1
    assert before.exposed["physics"]["energy"] == 1.5


def test_static_values_are_fixed_and_can_be_omitted():
    def f(x: nf.Real[-1, 1], scale):
        return x * scale

    net = nf.compile(f, static={"scale": 3})
    assert net(0.5) == net(0.5, scale=3) == 1.5
    with pytest.raises(ValueError, match="static argument 'scale'"):
        net(0.5, scale=2)


def test_static_override_of_an_original_default():
    def f(x: nf.Real[-1, 1], scale=2):
        return x * scale

    net = nf.compile(f, static={"scale": 3})
    assert net(0.5) == 1.5
    with pytest.raises(ValueError, match="fixed at compile time"):
        net(0.5, scale=2)


def test_source_parameter_names_do_not_collide_with_adapter_options():
    def f(x: nf.Real[-1, 1], cast: nf.Bool = False, *, detach: nf.Bool = False):
        with nf.scope("flags"):
            value = x + cast + 2 * detach
        return value, cast, detach

    net = nf.compile(f)
    assert net(0.5, cast=True, detach=True) == (3.5, 1.0, 1.0)
    assert net.inspect(0.5, cast=True, detach=True).output == (3.5, 1.0, 1.0)
    assert net.as_function(cast=True)(0.5, cast=True, detach=True) == (3.5, True, True)


def test_numpy_broadcasting_preserves_batch_shape_for_output_and_exposure():
    net = nf.compile(scoped)
    x = np.array([[-0.5], [0.5]])
    y = np.array([0, 1, 2])
    result = net.inspect(x, y, offset=1)
    assert result.output["score"].shape == (2, 3)
    assert np.array_equal(result.output["score"], x + y + 1)
    assert np.array_equal(result.exposed["physics"]["energy"], x + y)
    empty = net.inspect(np.empty((0,)), offset=1)
    assert empty.output["score"].shape == (0,)
    assert empty.exposed["physics"]["energy"].shape == (0,)


def test_nested_inputs_return_trees_and_batched_container_arrays():
    def f(xs: nf.Vec(nf.Vec(nf.Real[-2, 2], 2), 2), cfg: {"bias": nf.Real[0, 1]}):
        return {"values": (xs[0][0] + cfg["bias"], xs[1][0] - xs[0][1]), "flag": True}

    net = nf.compile(f)
    xs = [[0, 1], [2, -1]]
    assert net(xs, cfg={"bias": 0.5}) == f(xs, {"bias": 0.5})
    assert net.as_function(cast=True)(xs, cfg={"bias": 0.5})["flag"] is True
    batch = np.array([xs, xs], dtype=float)
    values = net(batch, {"bias": np.array([0, 1])})
    assert np.array_equal(values["values"][0], [0, 1])
    assert np.array_equal(values["values"][1], [1, 1])
    with pytest.raises(ValueError, match="trailing shape"):
        net(np.zeros((3, 2)), {"bias": 0})
    with pytest.raises(ValueError, match="dictionary keys"):
        net(xs, {"bias": 0, "extra": 1})


def test_sequence_domain_uses_the_original_calling_interface():
    def f(xs: nf.Seq(nf.Int[0, 2], 3)):
        return max(xs)

    net = nf.compile(f, arch="transformer", attention="relu")
    assert net(xs=[0, 2, 1]) == 2
    assert np.array_equal(net(np.array([[0, 2, 1], [1, 0, 1]])), [2, 1])


def test_zero_input_structured_output():
    def f():
        with nf.scope("constant", expose=("answer",)):
            answer = 7
        return {"answer": answer, "extra": (None, 2)}

    net = nf.compile(f)
    result = net.inspect()
    assert result.output == f()
    assert result.exposed == {"constant": {"answer": 7}}


def test_helper_pipeline_has_the_same_two_interfaces():
    def features(x: nf.Real[0, 1], *, shift: nf.Real[0, 1] = 0.5):
        return {"energy": x + shift}

    def head(energy: nf.Real[0, 2]):
        return energy + 2

    net = nf.pipeline(nf.region(features), nf.region(head))
    assert net(0.5) == 3
    result = net.inspect(0.5, shift=0)
    assert result.output == 2.5
    assert result.exposed == {"features": {"energy": 0.5}, "head": 2.5}


def test_torch_function_interface_retains_gradients_and_current_weights():
    torch = pytest.importorskip("torch")
    net = nf.compile(scoped)
    model = net.to_torch(dtype=torch.float64)
    function = model.as_function()
    x = torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)
    result = function.inspect(x, offset=1)
    assert torch.allclose(result.output["score"], x + 2)
    assert not result.exposed["physics"]["energy"].requires_grad
    result.output["score"].sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    before = result.output["score"].detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(function(x, offset=1)["score"], before)
    assert torch.equal(model.inspect(x, offset=1).output["score"], model.call(x, offset=1)["score"])


def test_torch_live_function_observations_support_auxiliary_losses():
    torch = pytest.importorskip("torch")
    model = nf.compile(scoped).to_torch(dtype=torch.float64)
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    result = model.as_function(detach=False).inspect(x)
    result.exposed["physics"]["energy"].backward()
    assert x.grad == 1


def test_torch_complete_model_serialization_retains_function_calls():
    torch = pytest.importorskip("torch")

    def local(x: nf.Real[-1, 1]):
        return x + 1

    model = nf.compile(local).to_torch(dtype=torch.float64)
    serialized = io.BytesIO()
    torch.save(model, serialized)
    serialized.seek(0)
    restored = torch.load(serialized, weights_only=False)
    assert restored.call(0.5).item() == 1.5


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_torch_function_and_inspection_match_numpy(arch):
    torch = pytest.importorskip("torch")
    net = nf.compile(scoped, arch=arch)
    model = net.to_torch(dtype=torch.float64)
    actual = model.inspect(0.5)
    expected = net.inspect(0.5)
    assert actual.output["score"].item() == expected.output["score"]
    assert actual.exposed["physics"]["energy"].item() == expected.exposed["physics"]["energy"]


def test_jax_function_interface_and_observations_are_jittable():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    net = nf.compile(scoped)
    apply, params = net.to_jax()
    x = jnp.array([0.5, -0.5])
    result = jax.jit(lambda p, z: apply.inspect(p, z, offset=1))(params, x)
    assert np.allclose(result.output["score"], [2.5, 1.5])
    assert np.allclose(result.exposed["physics"]["energy"], [1.5, 0.5])
    assert np.allclose(jax.grad(lambda z: apply.call(params, z)["score"].sum())(x), 1)
    assert np.allclose(apply.as_function(params)(0.5)["score"], 3.5)


def test_jax_live_function_observations_support_auxiliary_losses():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    apply, params = nf.compile(scoped).to_jax()

    def loss(p, x):
        return apply.as_function(p, detach=False).inspect(x).exposed["physics"]["energy"].sum()

    gradients = jax.grad(loss)(params, jnp.array([0.5]))
    assert any(np.any(np.asarray(g)) for g in jax.tree_util.tree_leaves(gradients["physics"]))


def test_torch_residual_function_calls_observe_the_trained_correction():
    torch = pytest.importorskip("torch")

    def f(x: nf.Real[-1, 1]):
        return {"value": x + 1}

    model = nf.compile(f).to_torch_residual(hidden=(4,), dtype=torch.float64)
    function = model.as_function()
    assert function(x=0.5)["value"].item() == 1.5
    with torch.no_grad():
        model.delta[-1].bias.add_(1)
    result = function.inspect(x=0.5)
    assert result.output["value"].item() == 2.5 and result.exposed == {}


def test_jax_residual_function_calls_use_current_parameter_tree():
    pytest.importorskip("jax")

    def f(x: nf.Real[-1, 1]):
        return {"value": x + 1}

    apply, params = nf.compile(f).to_jax_residual(hidden=(4,))
    assert np.allclose(apply.call(params, x=0.5)["value"], 1.5)
    updated = list(params)
    updated[-1] = dict(params[-1], b=params[-1]["b"] + 1)
    assert np.allclose(apply.inspect(updated, x=0.5).output["value"], 2.5)
