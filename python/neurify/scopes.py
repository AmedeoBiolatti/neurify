"""Lexical scope controls, lowered to separately compiled, wired code blocks."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import inspect
from types import MappingProxyType, ModuleType

import numpy as np

from . import _core
from ._host import Host, function_source
from .domains import Bool, Int, Real, Static
from .regions import Region, _batch, _output_contract, _policy


@dataclass(frozen=True)
class Scope:
    """A no-op Python context, interpreted as a compilation boundary."""

    name: str
    trainable: bool = True
    differentiable: bool = True
    expose: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or self.name.startswith("__neurify_"):
            raise ValueError("scope name must be non-empty and cannot start with '__neurify_'")
        _policy(self.trainable, "trainable")
        _policy(self.differentiable, "differentiable")
        if self.expose is not None:
            if isinstance(self.expose, str):
                raise TypeError("expose must be a sequence of variable names, not a string")
            names = tuple(self.expose)
            if not all(isinstance(n, str) and n.isidentifier() for n in names) or len(set(names)) != len(names):
                raise ValueError("expose must contain unique variable names")
            object.__setattr__(self, "expose", names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def scope(name, *, trainable=True, differentiable=True, expose=None):
    """Mark a named code scope for compilation; ordinary Python stays unchanged.

    Freeze weights with ``trainable=False``; stop gradients of values defined
    in the block with ``differentiable=False``. ``expose`` selects existing
    shared scalar activations; it never adds prediction heads or dead code.
    """
    return Scope(name, trainable, differentiable, expose)


def _environment(fn):
    env = dict(fn.__globals__)
    env.update(inspect.getclosurevars(fn).nonlocals)
    return env


def _resolve(node, env):
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve(node.value, env)
        if isinstance(parent, ModuleType):
            return vars(parent).get(node.attr)
    return None


def _is_scope(node, env):
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and _resolve(item.context_expr.func, env) is scope
        for item in node.items
    )


def has_scopes(fn):
    """Recognize the actual scope helper, including imported aliases."""
    source, _, _ = function_source(fn)
    tree = ast.parse(source)
    env = _environment(fn)
    return any(_is_scope(node, env) for node in ast.walk(tree))


def _parse_scope(node, env):
    if len(node.items) != 1 or node.items[0].optional_vars is not None:
        raise _core.CompileError("use one nf.scope context per with statement, without an 'as' target")
    call = node.items[0].context_expr
    if not isinstance(call, ast.Call) or _resolve(call.func, env) is not scope:
        raise _core.CompileError("only nf.scope contexts are supported in scoped functions")
    if any(keyword.arg is None for keyword in call.keywords):
        raise _core.CompileError("scope options must be literal keyword arguments")
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    except (ValueError, TypeError, SyntaxError):
        raise _core.CompileError("scope names and options must be compile-time literals") from None
    try:
        return scope(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise _core.CompileError(str(error)) from None


def _names(body, context):
    nodes = [node for statement in body for node in ast.walk(statement)]
    comprehension_targets = {
        id(target) for node in nodes if isinstance(node, ast.comprehension)
        for target in ast.walk(node.target)
    }
    return {
        node.id for node in nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, context)
        and not (context is ast.Store and id(node) in comprehension_targets)
    }


def _defined_after(body, incoming):
    """Conservative definite assignment at a boundary, including zero-trip loops."""
    defined = set(incoming)
    for statement in body:
        if isinstance(statement, ast.Assign):
            defined |= _names([ast.Expr(value=target) for target in statement.targets], ast.Store)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            defined |= _names([ast.Expr(value=statement.target)], ast.Store)
        elif isinstance(statement, ast.If):
            defined = _defined_after(statement.body, defined) & _defined_after(statement.orelse, defined)
        # Augmented assignment needs an already defined target. Loops may
        # execute zero times; they do not establish additional definite names.
    return defined


def _immutable(value):
    return value is None or isinstance(value, (bool, int, float, str)) or (
        isinstance(value, tuple) and all(_immutable(item) for item in value)
    )


def _domain(spec):
    if spec[0] == "real":
        return Real(spec[1], spec[2])
    if spec[0] == "int":
        return Int(spec[1], spec[2])
    if spec[0] == "bool":
        return Bool()
    if spec[0] == "static" and _immutable(spec[1]):
        return Static(spec[1])
    raise _core.CompileError("scope boundaries currently support scalar Real/Int/Bool values and immutable Static values")


def _helper(body, reads, env, index):
    name = f"__neurify_scope_step_{index}"
    while name in env:
        name += "_"
    definition = ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in reads], vararg=None,
                           kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=body, decorator_list=[], returns=None, type_comment=None,
    )
    tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    source = ast.unparse(tree) + "\n"
    namespace = dict(env)
    exec(compile(source, "<neurify-scope>", "exec"), namespace)
    fn = namespace[name]
    fn.__neurify_source__ = source
    return fn


def _wire_return(expr, available):
    """Flat fixed wiring for return values; None means an expression to compile."""
    if isinstance(expr, ast.Name) and expr.id in available:
        return [(expr.id, None)]
    if isinstance(expr, ast.Constant) and expr.value is None:
        return []
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (bool, int, float)):
        return [(None, float(expr.value))]
    if isinstance(expr, (ast.Tuple, ast.List, ast.Dict)):
        if isinstance(expr, ast.Dict):
            if any(key is None for key in expr.keys):
                return None
            try:
                keys = [ast.literal_eval(key) for key in expr.keys]
                if len(set(keys)) != len(keys):
                    return None
            except (ValueError, TypeError):
                return None
            items = expr.values
        else:
            items = expr.elts
        parts = [_wire_return(item, available) for item in items]
        return [ref for part in parts for ref in part] if all(part is not None for part in parts) else None
    return None


def _return_plan(expr, spec, specs, reserved):
    """Keep returned locals as wires, compiling only computed return leaves."""
    expressions, outputs, occupied = [], [], set(reserved)

    def count(s):
        if s[0] == "leaf":
            return 1
        if s[0] == "none":
            return 0
        return sum(count(child) for child in ([c for _, c in s[1]] if s[0] == "dict" else s[1]))

    def visit(value, tree):
        if isinstance(value, ast.Name) and value.id in specs:
            if specs[value.id][0] != "static":
                return [(value.id, None)]
            literal = ast.parse(repr(specs[value.id][1]), mode="eval").body
            fixed = _wire_return(literal, set())
            if fixed is not None:
                return fixed
        if isinstance(value, ast.Constant):
            fixed = _wire_return(value, set())
            if fixed is not None:
                return fixed
        if isinstance(value, (ast.List, ast.Tuple)) and tree[0] in ("list", "tuple"):
            if len(value.elts) == len(tree[1]) and not any(isinstance(v, ast.Starred) for v in value.elts):
                return [wire for child, child_spec in zip(value.elts, tree[1]) for wire in visit(child, child_spec)]
        if isinstance(value, ast.Dict) and tree[0] == "dict":
            try:
                keys = [ast.literal_eval(key) for key in value.keys]
            except (ValueError, TypeError):
                keys = None
            if keys == [key for key, _ in tree[1]] and len(set(keys)) == len(keys):
                return [wire for child, (_, child_spec) in zip(value.values, tree[1]) for wire in visit(child, child_spec)]
        width = count(tree)
        if not width:
            return []
        names = []
        for _ in range(width):
            name = f"__neurify_return_value_{len(occupied)}"
            while name in occupied:
                name += "_"
            occupied.add(name)
            names.append(name)
        expressions.append(value)
        outputs.extend(names)
        return [(name, None) for name in names]

    wiring = visit(expr, spec)
    return wiring, expressions, tuple(outputs)


@dataclass(frozen=True)
class _Step:
    region: Region
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    exposed: tuple[str, ...]
    named: bool
    final: bool = False


def compile_scoped(fn, inputs, static, options, trainable, block_configs=None):
    from .compiler import _param_domains, compile as compile_network
    from .activation import compile_activation_block

    block_configs = dict(block_configs or {})

    if options["output"] != "raw" or options["groups"] is not None or options["weights"] is not None:
        raise _core.CompileError("scoped compilation currently requires output='raw' without groups or weights")
    source, file, first_line = function_source(fn)
    tree = ast.parse(source)
    definition = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn.__name__), None)
    if definition is None:
        raise _core.CompileError("scopes require a Python function definition")
    env = _environment(fn)
    if not definition.body or not isinstance(definition.body[-1], ast.Return):
        raise _core.CompileError("scoped functions require a single final return statement")
    for statement in definition.body:
        if isinstance(statement, ast.Return):
            if statement is not definition.body[-1]:
                raise _core.CompileError("early returns across scopes are not yet supported")
            continue
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                raise _core.CompileError("define helper functions outside a scoped function")
            if isinstance(node, ast.Return):
                raise _core.CompileError("early returns across scopes are not yet supported")
            if isinstance(node, ast.With) and (node is not statement or not _is_scope(node, env)):
                raise _core.CompileError("scopes must be top-level function statements; nested contexts are not yet supported")
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.Delete)):
                raise _core.CompileError("global/nonlocal assignment and deletion are not supported across scopes")
    params = _param_domains(fn, inputs, static)
    specs = dict(params)
    for spec in specs.values():
        _domain(spec)

    # Trace the complete program without scope syntax for verification and
    # Python output structure. This graph is not used as a trained substitute.
    reference_body = []
    for statement in definition.body:
        reference_body.extend(statement.body if isinstance(statement, ast.With) else [statement])
    reference = _helper(reference_body, [name for name, _ in params], env, "reference")
    host = Host()
    handle = host.handle(reference)
    graph, out_spec, warnings = _core.trace(
        reference.__neurify_source__, reference.__name__, file, first_line, handle, params, host,
        max_loop_iters=options["max_loop_iters"], max_inline_depth=options["max_inline_depth"],
        max_table=options["max_table"], exhaustive_points=options["exhaustive_points"],
    )
    if not graph.num_outputs:
        raise _core.CompileError("scoped functions must return at least one numeric value")
    final_expr = definition.body[-1].value
    if final_expr is None:
        raise _core.CompileError("scoped functions must return numeric values")

    blocks, pending, names = [], [], set()
    for statement in definition.body[:-1]:
        if isinstance(statement, ast.With):
            if pending:
                blocks.append((None, pending))
                pending = []
            policy = _parse_scope(statement, env)
            if policy.name in names:
                raise _core.CompileError(f"duplicate scope name '{policy.name}'")
            names.add(policy.name)
            blocks.append((policy, statement.body))
        else:
            pending.append(statement)
    if pending:
        blocks.append((None, pending))

    unknown = set(block_configs) - names
    if unknown:
        raise ValueError(f"unknown blocks {sorted(unknown)}; available scopes: {sorted(names)}")

    available = list(specs)
    defined = set(specs)
    all_locals = set(available) | _names(definition.body, ast.Store)
    entries = []
    for policy, body in blocks:
        assigned = _names(body, ast.Store)
        defined = _defined_after(body, defined)
        entries.append((policy, body, tuple(available), assigned, set(defined)))
        available.extend(sorted(assigned - set(available)))
    live = _names([ast.Expr(value=final_expr)], ast.Load) & all_locals
    planned = []
    for policy, body, available_at_entry, assigned, defined in reversed(entries):
        exposed = policy.expose if policy else ()
        if exposed is not None and not set(exposed) <= assigned:
            raise _core.CompileError(f"scope '{policy.name}' can only expose variables assigned inside that scope")
        writes = assigned & live
        if not (writes | set(exposed or ())) <= defined:
            missing = ", ".join(sorted((writes | set(exposed or ())) - defined))
            raise _core.CompileError(f"scope boundary variables must be defined on every path: {missing}")
        ordered = tuple(sorted(writes))
        # Including overwritten incoming locals is conservative and preserves
        # augmented/conditional assignments without losing their old values.
        reads = (_names(body, ast.Load) | (assigned & set(available_at_entry))) & set(available_at_entry)
        reads = tuple(n for n in available_at_entry if n in reads)
        live = (live - assigned) | set(reads)
        planned.append((policy, body, reads, ordered))
    planned.reverse()

    steps = []
    for index, (policy, body, reads, writes) in enumerate(planned):
        if not writes:
            if policy:
                raise _core.CompileError(f"scope '{policy.name}' has no live outputs; exposure cannot retain dead code")
            continue
        dynamic_reads = tuple(n for n in reads if specs[n][0] != "static")
        exposed = (writes if policy.expose is None else policy.expose) if policy else ()
        if policy:
            config = block_configs.get(policy.name)
            block_options = dict(options)
            if config is not None and config.arch is not None:
                block_options["arch"] = config.arch
            net = compile_activation_block(body, reads, writes, exposed, specs, env, block_options, policy.name)
            if config is not None:
                net = config._build(net)
                policy = replace(
                    policy,
                    trainable=policy.trainable if config.trainable is None else config.trainable,
                    differentiable=policy.differentiable if config.differentiable is None else config.differentiable,
                )
        else:
            result = ast.Dict(keys=[ast.Constant(n) for n in writes], values=[ast.Name(n, ast.Load()) for n in writes])
            helper = _helper(body + [ast.Return(result)], reads, env, index)
            net = compile_network(helper, inputs={n: _domain(specs[n]) for n in reads}, **options)
        if any(spec[0] != "leaf" for _, spec in net.out_spec[1]):
            raise _core.CompileError("variables crossing scope boundaries or exposed by a scope must be scalars")
        contract = _output_contract(net)
        if contract is None:
            raise _core.CompileError("cannot infer scope output bounds; rebuild the extension for IR output specifications")
        lo, hi, _, _ = contract
        for column, (name, output_spec) in enumerate(net.out_spec[1]):
            kind = output_spec[1]
            specs[name] = (kind, float(lo[column]), float(hi[column])) if kind != "bool" else ("bool",)
        label = policy.name if policy else f"__neurify_code_{index}"
        region = Region(label, net, policy.trainable if policy else True, policy.differentiable if policy else True)
        steps.append(_Step(region, dynamic_reads, writes, tuple(exposed), policy is not None))
    wiring, expressions, return_names = _return_plan(final_expr, out_spec, specs, set(available) | set(env))
    if expressions:
        result = ast.Tuple(elts=expressions, ctx=ast.Load())
        reads = tuple(n for n in available if n in _names([ast.Expr(value=result)], ast.Load))
        helper = _helper([ast.Return(result)], reads, env, "return")
        net = compile_network(helper, inputs={n: _domain(specs[n]) for n in reads}, **options)
        steps.append(_Step(Region("__neurify_return", net), tuple(n for n in reads if specs[n][0] != "static"), return_names, (), False))
    result = ScopedNetwork(steps, wiring, params, graph, out_spec, fn, warnings, block_configs)
    if trainable:
        result = result.prepare_for_training(**(trainable if isinstance(trainable, dict) else {}))
    return result


class ScopedNetwork:
    """One function compiled into separately wired code scopes."""

    def __init__(self, steps, wiring, params, graph, out_spec, fn, warnings, block_configs=None):
        from .blocks import _block_info

        self.steps = tuple(steps)
        self._regions = tuple(step.region for step in steps)
        self.regions = MappingProxyType({r.name: r for r in self._regions})
        self.scope_names = tuple(s.region.name for s in steps if s.named)
        self.block_configs = MappingProxyType(dict(block_configs or {}))
        self.region_names = tuple(self.regions)
        self._wiring, self._params = wiring, params
        self.graph = self.source_graph = graph
        self.out_spec = self.source_out_spec = out_spec
        self.fn = fn
        self.n_in, self.n_out = graph.num_inputs, graph.num_outputs
        self._input_names = tuple(name for name, spec in params if spec[0] != "static")
        self.warnings = list(warnings) + [w for r in self._regions for w in r.network.warnings]
        self.report = {
            "arch": "scoped", "method": "scopes", "depth": self.depth, "widths": self.widths,
            "params": self.num_params, "units": sum(r.network.report["units"] for r in self._regions),
            "margin_sites": sum(r.network.report["margin_sites"] for r in self._regions),
            "margin": max((r.network.report["margin"] for r in self._regions), default=1e-3),
            "approx": [a for r in self._regions for a in r.network.report.get("approx", [])],
            "exact": all(r.network.exact for r in self._regions), "exhaustive_points": 0,
            "scopes": list(self.scope_names),
            "blocks": [_block_info(s) for s in self.steps],
        }

    @property
    def arch(self):
        return "scoped"

    @property
    def depth(self):
        return sum(r.network.depth for r in self._regions)

    @property
    def widths(self):
        return [w for r in self._regions for w in r.network.widths]

    @property
    def num_params(self):
        return sum(r.network.num_params for r in self._regions)

    @property
    def exact(self):
        return self.report["exact"]

    @property
    def input_names(self):
        return list(self._input_names)

    @property
    def blocks(self):
        """Explicit source blocks, keyed by scope name."""
        return MappingProxyType({name: self.regions[name] for name in self.scope_names})

    @property
    def activation_names(self):
        return MappingProxyType({
            s.region.name: s.region.network.activation_names
            for s in self.steps if s.named and getattr(s.region.network, "_is_activation_block", False)
        })

    def __repr__(self):
        return f"<neurify.ScopedNetwork scopes={self.scope_names} params={self.num_params} exact={self.exact}>"

    def flatten_inputs(self, *args, **kwargs):
        from .interface import flatten_single

        return flatten_single(self, args, kwargs)

    @property
    def __signature__(self):
        return inspect.signature(self.fn)

    def _run(self, X, dtype, capture):
        X, single = _batch(X, self.n_in, dtype)
        env = {name: X[:, i:i + 1] for i, name in enumerate(self._input_names)}
        observed, values = {}, None
        for step in self.steps:
            inputs = np.concatenate([env[n] for n in step.reads], axis=-1) if step.reads else np.empty((len(X), 0), dtype=dtype)
            if capture and step.named:
                values, activations = step.region.network.forward_with_activations(inputs, dtype=dtype)
            else:
                values = step.region.network.forward(inputs, dtype=dtype) if len(X) else np.empty((0, step.region.network.n_out), dtype=dtype)
            if not step.final:
                env.update({n: values[:, i:i + 1] for i, n in enumerate(step.writes)})
            if capture and step.named:
                observed[step.region.name] = {n: (activations[n][0] if single else activations[n]).copy() for n in step.exposed}
        if self._wiring is not None:
            values = np.concatenate([
                env[name] if name is not None else np.full((len(X), 1), value, dtype=dtype)
                for name, value in self._wiring
            ], axis=-1)
        return (values[0] if single else values), observed

    def forward(self, X, dtype=np.float64):
        return self._run(X, dtype, False)[0]

    def forward_with_intermediates(self, X, dtype=np.float64, *, structured=True):
        _policy(structured, "structured")
        output, observed = self._run(X, dtype, True)
        if not structured:
            observed = {
                name: np.stack(list(values.values()), axis=-1) if values else np.empty(output.shape[:-1] + (0,), dtype=dtype)
                for name, values in observed.items()
            }
        return output, observed

    def __call__(self, *args, **kwargs):
        from .interface import prediction_call

        return prediction_call(self, args, kwargs)

    def as_function(self, *, cast=False):
        from .interface import numpy_function

        return numpy_function(self, cast=cast)

    def inspect(self, *args, **kwargs):
        return self.as_function().inspect(*args, **kwargs)

    def reference(self, X, margin=None):
        X, _ = _batch(X, self.n_in, np.float64)
        if not len(X):
            return np.empty((0, self.n_out)), np.empty(0, dtype=bool)
        values, near = self.graph.eval_batch(X.ravel().tolist(), margin)
        if not self.n_in:
            return np.repeat(np.asarray(values)[None, :], len(X), axis=0), np.full(len(X), near[0], dtype=bool)
        return np.asarray(values).reshape(len(X), self.n_out), np.asarray(near, dtype=bool)

    source_reference = reference

    def prepare_for_training(self, **kwargs):
        return ScopedNetwork(
            [replace(s, region=replace(s.region, network=s.region.network.prepare_for_training(**kwargs))) for s in self.steps],
            self._wiring, self._params, self.graph, self.out_spec, self.fn, self.warnings, self.block_configs,
        )

    def summary(self):
        lines = [f"neurify.ScopedNetwork(inputs={self.n_in}, outputs={self.n_out}, params={self.num_params})"]
        for info in self.report["blocks"]:
            correction = f"correction_hidden={info['residual_hidden']}, " if info["residual_hidden"] is not None else ""
            lines.append(
                f"  {info['name']}: arch={info['arch']}, mode={info['mode']}, "
                f"depth={info['depth']}, widths={info['widths']}, "
                f"{correction}"
                f"params={info['params']}, trainable_params={info['trainable_params']}, "
                f"trainable={info['trainable']}, differentiable={info['differentiable']}, "
                f"expose={tuple(info['expose'])}"
            )
        return "\n".join(lines)

    def to_torch(self, dtype=None):
        from .export_scopes_torch import ScopedModule
        from .interface import attach_torch_interface

        return attach_torch_interface(ScopedModule(self, dtype), self)

    def to_jax(self, dtype=None):
        from .export_scopes_jax import to_jax
        from .interface import attach_jax_interface

        apply, params = to_jax(self, dtype)
        return attach_jax_interface(apply, self, dtype), params
