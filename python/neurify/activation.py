"""Materialized statement activations, shared by observation and consumers."""

from __future__ import annotations

import ast
from dataclasses import replace
import math

import numpy as np

from . import _core
from ._host import Host
from .regions import Region, _output_contract
from .scopes import ScopedNetwork, _Step, _defined_after, _domain, _helper, _names


def compile_activation_block(body, reads, writes, exposed, specs, env, options, label):
    from .compiler import _param_domains, compile as compile_network

    initial = {n: specs[n] for n in reads}
    available, defined, entries, last = list(reads), set(reads), [], {}
    for index, statement in enumerate(body):
        assigned = _names([statement], ast.Store)
        defined = _defined_after([statement], defined)
        entries.append((statement, tuple(available), assigned, set(defined)))
        available.extend(sorted(assigned - set(available)))
        last.update({name: index for name in assigned})
    live, planned = set(writes), []
    for index in range(len(entries) - 1, -1, -1):
        statement, incoming, assigned, defined = entries[index]
        outputs = tuple(sorted(assigned & live))
        if not outputs:
            # Exposure does not keep dead work alive or add an output head.
            continue
        if not set(outputs) <= defined:
            raise _core.CompileError("activation boundary variables must be defined on every path")
        dependencies = _names([statement], ast.Load)
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            dependencies |= assigned & set(incoming)
        dependencies = tuple(n for n in incoming if n in dependencies)
        live = (live - assigned) | set(dependencies)
        planned.append((index, statement, dependencies, outputs))
    planned.reverse()

    retained = {name: source_index for source_index, _, _, outputs in planned for name in outputs if last[name] == source_index}
    missing = set(exposed) - set(retained)
    if missing:
        names = ", ".join(sorted(missing))
        raise _core.CompileError(
            f"scope '{label}' has no retained activation for {names}; "
            "expose only values used by a later statement or the return. "
            "Exposure cannot resurrect dead code or add a prediction head."
        )

    # Only the scope's real live-outs are graph outputs. Internal observations
    # never appear in this reference graph or in its trainable readout.
    result = ast.Dict(keys=[ast.Constant(n) for n in writes], values=[ast.Name(n, ast.Load()) for n in writes])
    reference = _helper(body + [ast.Return(result)], reads, env, "activation_reference")
    params = _param_domains(reference, {n: _domain(initial[n]) for n in reads}, None)
    host = Host()
    handle = host.handle(reference)
    graph, out_spec, warnings = _core.trace(
        reference.__neurify_source__, reference.__name__, "<neurify-activation>", 1,
        handle, params, host, max_loop_iters=options["max_loop_iters"],
        max_inline_depth=options["max_inline_depth"], max_table=options["max_table"],
        exhaustive_points=options["exhaustive_points"],
    )

    current, steps, capture, available_activations = dict(initial), [], {}, {}
    for source_index, statement, dependencies, outputs in planned:
        result = ast.Dict(keys=[ast.Constant(n) for n in outputs], values=[ast.Name(n, ast.Load()) for n in outputs])
        helper = _helper([statement, ast.Return(result)], dependencies, env, f"activation_{source_index}")
        dynamic = tuple(n for n in dependencies if current[n][0] != "static")
        net = compile_network(helper, inputs={n: _domain(current[n]) for n in dependencies}, **options)
        if any(spec[0] != "leaf" for _, spec in net.out_spec[1]):
            raise _core.CompileError("materialized scope activations must be scalars (Real/Int/Bool)")
        contract = _output_contract(net)
        if contract is None:
            raise _core.CompileError("cannot infer activation bounds; rebuild the extension for IR output specifications")
        lo, hi, _, _ = contract
        for column, (name, leaf) in enumerate(net.out_spec[1]):
            kind = leaf[1]
            current[name] = (kind, float(lo[column]), float(hi[column])) if kind != "bool" else ("bool",)
            if retained.get(name) == source_index:
                available_activations[name] = len(steps)
                if name in exposed:
                    capture.setdefault(len(steps), []).append(name)
        steps.append(_Step(Region(f"__neurify_statement_{source_index}", net), dynamic, outputs, (), False))

    return ActivationBlock(
        steps, [(name, None) for name in writes], params, graph, out_spec, reference,
        warnings, capture, tuple(sorted(available_activations)), {n: current[n] for n in writes},
    )


class ActivationBlock(ScopedNetwork):
    """A scope with actual, fixed-wiring activations between live statements."""

    _is_activation_block = True

    def __init__(self, steps, wiring, params, graph, out_spec, fn, warnings, capture, names, output_specs):
        super().__init__(steps, wiring, params, graph, out_spec, fn, warnings)
        self._capture = {index: tuple(values) for index, values in capture.items()}
        self._activation_names = names
        self._output_specs = output_specs
        self._activation_specs = {
            name: dict(self.steps[index].region.network.out_spec[1])[name]
            for index, names in self._capture.items() for name in names
        }

    @property
    def activation_names(self):
        return self._activation_names

    def output_contract(self):
        names = [n for n, _ in self.out_spec[1]]
        kinds, lo, hi = [], [], []
        for name in names:
            spec = self._output_specs[name]
            kinds.append(spec[0])
            lo.append(0.0 if spec[0] == "bool" else spec[1])
            hi.append(1.0 if spec[0] == "bool" else spec[2])
        return np.asarray(lo), np.asarray(hi), kinds, False

    def _activation_run(self, X, dtype, capture):
        X = np.asarray(X, dtype=dtype)
        if X.ndim < 1 or X.shape[-1] != self.n_in:
            raise ValueError(f"expected trailing input dimension {self.n_in}")
        batch = X.shape[:-1]
        rows = math.prod(batch)
        env = {name: X[..., i] for i, name in enumerate(self._input_names)}
        probes = {}
        for index, step in enumerate(self.steps):
            values = np.stack([env[n] for n in step.reads], axis=-1) if step.reads else np.empty(batch + (0,), dtype=dtype)
            if rows:
                output = step.region.network.forward(values.reshape(rows, step.region.network.n_in), dtype=dtype)
            else:
                output = np.empty((0, step.region.network.n_out), dtype=dtype)
            output = output.reshape(batch + (step.region.network.n_out,))
            env.update({name: output[..., i] for i, name in enumerate(step.writes)})
            if capture:
                probes.update({name: env[name] for name in self._capture.get(index, ())})
        output = np.stack([env[name] for name, _ in self._wiring], axis=-1)
        return output, probes

    def _run(self, X, dtype, capture):
        output, _ = self._activation_run(X, dtype, False)
        return output, {}

    def forward_with_activations(self, X, dtype=np.float64):
        return self._activation_run(X, dtype, True)

    def weights(self):
        """The actual per-statement parameter dictionaries; probes have none."""
        return {"body": "activations", "statements": [s.region.network.weights() for s in self.steps]}

    def prepare_for_training(self, **kwargs):
        return ActivationBlock(
            [replace(s, region=replace(s.region, network=s.region.network.prepare_for_training(**kwargs))) for s in self.steps],
            self._wiring, self._params, self.graph, self.out_spec, self.fn, [], self._capture,
            self.activation_names, self._output_specs,
        )

    def to_torch(self, dtype=None):
        from .export_activation_torch import ActivationModule
        from .interface import attach_torch_interface

        return attach_torch_interface(ActivationModule(self, dtype), self)

    def to_jax(self, dtype=None):
        from .export_activation_jax import to_jax
        from .interface import attach_jax_interface

        apply, params = to_jax(self, dtype)
        return attach_jax_interface(apply, self, dtype), params
