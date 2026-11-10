"""Exposed locals are shared activations; probes never add prediction heads."""

import numpy as np
import pytest

import neurify as nf


def compile_chain(expose=("a",), arch="mlp"):
    source = '''def f(x: nf.Real[-1, 1]):
    with nf.scope("calculation", expose=EXPOSE):
        a = x + 2
        b = 3 * a
    return b
'''.replace("EXPOSE", repr(expose))
    namespace = {"nf": nf}
    exec(source, namespace)
    fn = namespace["f"]
    fn.__neurify_source__ = source
    return nf.compile(fn, arch=arch)


def arrays(tree):
    if isinstance(tree, np.ndarray):
        return [tree]
    if isinstance(tree, dict):
        return [a for value in tree.values() for a in arrays(value)]
    if isinstance(tree, (list, tuple)):
        return [a for value in tree for a in arrays(value)]
    return []


@pytest.mark.parametrize("arch", nf.ARCHS)
@pytest.mark.parametrize("exposure", [("a",), ("b",), ("a", "b"), ()])
def test_exposure_changes_only_observation_metadata(arch, exposure):
    unobserved = compile_chain(None, arch)
    observed = compile_chain(exposure, arch)
    assert observed.num_params == unobserved.num_params
    assert observed.depth == unobserved.depth
    assert observed.widths == unobserved.widths
    left = arrays(unobserved.regions["calculation"].network.weights())
    right = arrays(observed.regions["calculation"].network.weights())
    assert len(left) == len(right)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right))
    X = np.array([[-1], [-0.5], [0], [0.5], [1]], dtype=float)
    assert np.array_equal(observed.forward(X), unobserved.forward(X))
    assert np.array_equal(observed.forward(X), observed.forward_with_intermediates(X)[0])
    assert observed.activation_names["calculation"] == ("a", "b")
    assert nf.verify(observed).ok


def test_probe_is_not_an_additional_scope_output_or_head():
    net = compile_chain()
    block = net.regions["calculation"].network
    assert net.num_params == 4
    assert block.graph.num_outputs == block.n_out == 1  # only b, not a+b
    assert len(block.steps) == 2
    assert block.steps[0].writes == ("a",)
    assert block.steps[1].reads == ("a",)
    assert block.steps[1].writes == ("b",)


def test_changing_the_producer_changes_both_probe_and_prediction():
    net = compile_chain()
    function = net.as_function()
    initial = function.inspect(x=0.5)
    producer = net.regions["calculation"].network.steps[0].region.network
    producer.weights()["out"]["bias"] += 1
    changed = function.inspect(x=0.5)
    assert initial.output == 7.5 and initial.exposed["calculation"]["a"] == 2.5
    assert changed.output == 10.5 and changed.exposed["calculation"]["a"] == 3.5
    assert changed.output == 3 * changed.exposed["calculation"]["a"]
    assert initial.exposed["calculation"]["a"] == 2.5


def test_dead_variable_exposure_fails_instead_of_adding_a_head():
    source = '''def f(x: nf.Real[-1, 1]):
    with nf.scope("block", expose=("unused",)):
        unused = x + 10
        y = 2 * x
    return y
'''
    namespace = {"nf": nf}
    exec(source, namespace)
    namespace["f"].__neurify_source__ = source
    with pytest.raises(nf.CompileError, match="no retained activation.*unused"):
        nf.compile(namespace["f"])


def test_reassigned_name_means_its_final_definition():
    def f(x: nf.Real[-1, 1]):
        with nf.scope("block", expose=("a",)):
            a = x + 2
            a = 2 * a
            b = 3 * a
        return b

    net = nf.compile(f)
    result = net.inspect(0.5)
    assert result.exposed["block"]["a"] == 5
    assert result.output == 15


def test_dead_final_reassignment_cannot_mislabel_an_older_activation():
    def f(x: nf.Real[-1, 1]):
        with nf.scope("block", expose=("a",)):
            a = x + 2
            b = 3 * a
            a = 0
        return b

    with pytest.raises(nf.CompileError, match="no retained activation"):
        nf.compile(f)


def test_compound_statement_internal_values_need_a_retained_boundary():
    def f(x: nf.Int[0, 2]):
        with nf.scope("branch", expose=("a",)):
            if x > 0:
                a = x + 2
                b = 3 * a
            else:
                a = x - 2
                b = 3 * a
        return b

    with pytest.raises(nf.CompileError, match="no retained activation"):
        nf.compile(f)


def test_preparation_keeps_activation_boundaries_and_probe_topology():
    net = compile_chain().prepare_for_training(width=8, depth=2)
    assert net.activation_names["calculation"] == ("a", "b")
    assert nf.verify(net).ok
    result = net.inspect(0.5)
    assert result.output == 3 * result.exposed["calculation"]["a"]


def test_mixed_return_expressions_do_not_add_a_copy_head_for_the_activation():
    def f(x: nf.Real[-1, 1]):
        with nf.scope("values", expose=("a",)):
            a = x + 2
        return {"a": a, "b": 3 * a}

    net = nf.compile(f)
    consumer = net.regions["__neurify_return"].network
    assert consumer.n_out == 1  # b is computed; a is fixed return wiring
    consumer.weights()["out"]["bias"] += 1
    result = net.inspect(0.5)
    assert result.output == {"a": 2.5, "b": 8.5}
    assert result.output["a"] == result.exposed["values"]["a"]


def test_capture_context_is_call_local_and_resets_after_failure():
    from neurify.torch_capture import capture_call, current_capture

    class Module:
        def __call__(self, x):
            request = current_capture(self)
            if request is not None:
                request.values["activation"] = x
            if x < 0:
                raise ValueError("failed forward")
            return x * 3

    module = Module()
    assert capture_call(module, 2, detach=False, structured=True) == (6, {"activation": 2})
    assert current_capture(module) is None
    with pytest.raises(ValueError, match="failed forward"):
        capture_call(module, -1, detach=False, structured=True)
    assert current_capture(module) is None
    assert module(2) == 6


def test_capture_context_isolated_between_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from neurify.torch_capture import capture_call, current_capture

    barrier = Barrier(2)

    class Module:
        def __call__(self, x):
            request = current_capture(self)
            request.values["activation"] = x
            barrier.wait(timeout=5)
            return x * 3

    module = Module()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda x: capture_call(module, x, detach=False, structured=True), [2, 5]))
    assert results == [(6, {"activation": 2}), (15, {"activation": 5})]
    assert current_capture(module) is None


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_torch_live_probe_is_an_ancestor_of_the_loss(shape):
    torch = pytest.importorskip("torch")
    model = compile_chain().to_torch(dtype=torch.float64)
    function = model.as_function(detach=False)
    x = torch.full(shape, 0.5, dtype=torch.float64, requires_grad=True)
    result = function.inspect(x=x)
    a = result.exposed["calculation"]["a"]
    assert a.shape == x.shape
    a.retain_grad()
    result.output.sum().backward()
    assert torch.equal(a.grad, torch.full_like(a, 3))
    assert torch.equal(x.grad, torch.full_like(x, 3))


def test_torch_producer_hook_changes_the_activation_consumed_downstream():
    torch = pytest.importorskip("torch")
    model = compile_chain().to_torch(dtype=torch.float64)
    producer = model.regions["calculation"].layers[0]
    handle = producer.register_forward_hook(lambda module, inputs, output: output + 1)
    try:
        result = model.inspect(x=0.5)
        assert result.exposed["calculation"]["a"].item() == 3.5
        assert result.output.item() == 10.5
    finally:
        handle.remove()


def test_torch_snapshot_collection_does_not_change_output_gradients():
    torch = pytest.importorskip("torch")
    model = compile_chain().to_torch(dtype=torch.float64)
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    prediction = model.call(x=x)
    before = torch.autograd.grad(prediction, tuple(model.parameters()) + (x,))
    result = model.inspect(x=x)
    after = torch.autograd.grad(result.output, tuple(model.parameters()) + (x,))
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert not result.exposed["calculation"]["a"].requires_grad


def test_torch_inspection_preserves_normal_module_hooks():
    torch = pytest.importorskip("torch")
    model = compile_chain().to_torch(dtype=torch.float64)
    handle = model.register_forward_hook(lambda module, inputs, output: output + 1)
    try:
        assert model.call(x=0.5).item() == 8.5
        assert model.inspect(x=0.5).output.item() == 8.5
    finally:
        handle.remove()


def test_torch_live_flat_scope_probes_fail_clearly():
    torch = pytest.importorskip("torch")
    model = compile_chain().to_torch()
    with pytest.raises(ValueError, match="structured=True"):
        model.forward_with_intermediates(torch.tensor([0.5]), detach=False, structured=False)


def test_torch_structured_region_probe_is_consumed_by_the_next_region():
    torch = pytest.importorskip("torch")

    def features(x: nf.Real[-1, 1]):
        return {"a": x + 2}

    def head(a: nf.Real[1, 3]):
        return 3 * a

    model = nf.pipeline(nf.region(features), nf.region(head)).to_torch(dtype=torch.float64)
    result = model.as_function(detach=False).inspect(x=torch.tensor(0.5, dtype=torch.float64, requires_grad=True))
    a = result.exposed["features"]["a"]
    derivative = torch.autograd.grad(result.output, a)[0]
    assert derivative.item() == 3


def test_jax_producer_update_changes_the_shared_activation_and_output():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    apply, params = compile_chain().to_jax()
    updated = dict(params)
    layers = list(params["calculation"])
    producer = dict(layers[0])
    producer["out"] = dict(producer["out"], b=producer["out"]["b"] + 1)
    layers[0] = producer
    updated["calculation"] = layers
    result = jax.jit(lambda p, x: apply.inspect(p, x))(updated, jnp.array(0.5))
    assert np.allclose(result.output, 10.5)
    assert np.allclose(result.exposed["calculation"]["a"], 3.5)
