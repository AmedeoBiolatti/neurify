"""Regression tests for errors at the Python/Rust API boundary."""

import pytest

from neurify import _core


def graph_with_two_inputs():
    graph = _core.Graph()
    x = graph.input("x", "int", 0, 1)
    graph.input("y", "int", 0, 1)
    graph.output(x)
    return graph


def test_invalid_node_reports_compile_error():
    with pytest.raises(_core.CompileError, match="does not exist"):
        _core.Graph().node_op(0)


def test_output_specs_report_intermediate_domain_contracts():
    graph = graph_with_two_inputs()
    assert graph.output_specs() == [("int", 0.0, 1.0)]
    assert graph.output_specs(exhaustive_points=0) == [("int", 0.0, 1.0)]


@pytest.mark.parametrize("values", [[1], [0, 1, 0]])
def test_incomplete_batch_reports_compile_error(values):
    graph = graph_with_two_inputs()
    net, _ = _core.compile(graph)
    for evaluate in (graph.eval_batch, net.forward, net.normalize_activations):
        with pytest.raises(_core.CompileError, match="complete rows"):
            evaluate(values)


def test_complete_and_empty_batches():
    graph = graph_with_two_inputs()
    net, _ = _core.compile(graph)
    assert net.forward([0, 1, 1, 0]) == [0, 1]
    assert graph.eval_batch([0, 1, 1, 0]) == ([0, 1], [False, False])
    assert net.forward([]) == []
    assert graph.eval_batch([]) == ([], [])


def test_constant_network_batch():
    graph = _core.Graph()
    graph.output(graph.constant(7, "int"))
    net, _ = _core.compile(graph)
    assert net.forward([]) == [7]
    assert graph.eval_batch([]) == ([7], [False])
    for evaluate in (graph.eval_batch, net.forward, net.normalize_activations):
        with pytest.raises(_core.CompileError, match="zero-input"):
            evaluate([1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_margin_reports_compile_error(value):
    with pytest.raises(_core.CompileError, match="margin"):
        _core.compile(graph_with_two_inputs(), margin=value)


@pytest.mark.parametrize("name", ["attn_tol", "ln_tol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, 1])
def test_invalid_transformer_tolerances(name, value):
    with pytest.raises(_core.CompileError, match=name):
        _core.compile(graph_with_two_inputs(), arch="transformer", layernorm=True, **{name: value})
