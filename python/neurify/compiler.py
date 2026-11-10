"""compile(): Python function -> exact neural network."""

from __future__ import annotations

import functools
import inspect

from . import _core
from ._host import Host, function_source
from .domains import Bool, Int, Real, Static, to_spec
from .network import Network

ARCHS = ("mlp", "resmlp", "glu", "transformer")
ATTENTIONS = ("softmax", "relu", "hard")
METHODS = ("depth", "size", "trainable", "sparse")
OUTPUTS = ("raw", "rank", "onehot")


def _param_domains(fn, inputs, static):
    sig = inspect.signature(fn)
    try:
        ann = inspect.get_annotations(fn, eval_str=True)
    except Exception:  # pragma: no cover - unevaluable annotations
        ann = {}
    inputs = dict(inputs or {})
    static = dict(static or {})
    params = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"*args / **kwargs parameters are not supported ('{name}')")
        if name in static:
            d = Static(static[name])
        elif name in inputs:
            d = inputs[name]
        elif name in ann:
            d = ann[name]
            if d is float:
                d = Real()
            elif d is int:
                d = Int()
            elif d is bool:
                d = Bool()
        elif p.default is not p.empty:
            d = Static(p.default)
        else:
            raise TypeError(
                f"no domain for parameter '{name}': pass inputs={{'{name}': neurify.Int(lo, hi)}} "
                f"or annotate it (e.g. {name}: neurify.Real[-1, 1])"
            )
        params.append((name, to_spec(d)))
    return params


def compile(
    fn=None,
    inputs: dict | None = None,
    *,
    static: dict | None = None,
    arch: str = "mlp",
    method: str = "depth",
    margin: float = 1e-3,
    strict: bool = False,
    max_loop_iters: int = 1000,
    max_inline_depth: int = 64,
    max_table: int = 4096,
    exhaustive_points: int = 65536,
    robust: bool = True,
    attention: str = "softmax",
    attn_tol: float = 1e-6,
    mlp: str = "relu",
    layernorm: bool = False,
    ln_tol: float = 1e-6,
    output: str = "raw",
    slope: float = 0.0,
    repr: str = "scalar",
    onehot_max: int = 64,
    trainable: bool | dict = False,
    groups: list | None = None,
    weights: list | None = None,
    blocks: dict | None = None,
) -> Network:
    """Compile a plain Python function into an exactly-equivalent network.

    Args:
        fn: the function. Its source is parsed and interpreted symbolically
            (by the Rust frontend); helpers it calls are inlined.
        inputs: ``{param: domain}`` (see ``neurify.domains``); annotations
            like ``x: Int[0, 9]`` work too.
        static: ``{param: value}`` compile-time constant arguments.
        arch: ``"mlp"`` (ReLU MLP), ``"resmlp"`` (residual ReLU MLP),
            ``"glu"`` (residual ReGLU; exact products of real values) or
            ``"transformer"`` (token 0 runs the program; ``Seq`` inputs live
            on tokens 1..n and are read through attention).
        method: gadget/schedule objective: ``"depth"``, ``"size"`` or
            ``"trainable"`` (well-conditioned, symmetric gadgets).
        margin: band width for real-valued comparisons that cannot be exact
            over a continuum (reported in ``net.report``).
        strict: refuse anything that is not exact on the whole domain.
        robust: noise-robust discrete gadgets (flat around every admissible
            integer / table value), so rounding errors never accumulate --
            required for deep programs and float32. ``False`` uses cheaper
            interpolating gadgets (fewer units, non-zero gradients).
        max_loop_iters: unrolling bound for ``while`` / symbolic ``range``;
            loops not provably done within it are rejected.
        attention: transformer attention: ``"softmax"`` (standard; one-hot
            within ``attn_tol`` per read, reported as approximate),
            ``"relu"`` (unnormalised relu(1-(i-j)^2) scores: exact) or
            ``"hard"`` (argmax: exact, no gradient through the selection).
        mlp: transformer MLP blocks: ``"relu"`` or ``"glu"``.
        layernorm: opt-in pre-LayerNorm (stock transformer layout). Exactness
            is approximate: every LayerNorm read is within relative error
            ``ln_tol`` (realised with mirrored dimensions and anchors).
        slope: robust gadgets only, in [0, 1]: slope at admissible values as
            a fraction of the interpolating slope. 0 (default): flat plateaus,
            rounding errors are removed and no gradient flows through
            discrete operations. Larger: still exact at every admissible
            value, errors are damped by ``slope`` per gadget, and gradients
            flow (better for fine-tuning).
        repr: value representation. ``"scalar"``: one number per value.
            ``"onehot"``: values with at most ``onehot_max`` distinct values
            are vectors of 0/1 indicators; unary operations are free
            regroupings and every n-ary operation is one robust AND per
            reachable argument tuple. Every unit sees an input range of about
            the number of arguments instead of the full value range, so the
            network tolerates much larger weight updates (see
            ``trainability_report``), at the cost of more units.
        output: how the outputs are presented. ``"raw"``: the program's
            values. ``"rank"`` / ``"onehot"``: the flat output vector is read
            as scores of one decision and only their order is kept --
            ``rank_i = #{j: s_i > s_j}`` or ``onehot_i = [s_i is a maximum]``.
            The argmax (with ties) is unchanged but the output range is
            O(1), which matters for fine-tuning programs whose scores span
            many orders of magnitude.
        trainable: ``True`` or kwargs for ``Network.prepare_for_training``.
        groups: with ``output="rank"`` / ``"onehot"``: sizes of consecutive
            groups of the flat outputs, each encoded on its own (default: one
            group). Used by ``neurify.merge``.
        weights: one weight per group (groups of equal size n): the outputs
            are the n exact combinations ``sum_g weights[g] * enc_g``.
        blocks: ``{scope_name: neurify.Block(...)}`` overrides for named
            scopes: architecture, additional capacity, residual correction,
            weight freezing, and gradient-flow policy.

    Can be used as a decorator: ``@neurify.compile(inputs=...)``.
    """
    if fn is None:
        return functools.partial(
            compile, inputs=inputs, static=static, arch=arch, method=method, margin=margin, strict=strict,
            max_loop_iters=max_loop_iters, max_inline_depth=max_inline_depth, max_table=max_table,
            exhaustive_points=exhaustive_points, robust=robust, attention=attention, attn_tol=attn_tol,
            mlp=mlp, layernorm=layernorm, ln_tol=ln_tol, output=output, slope=slope, repr=repr, onehot_max=onehot_max,
            trainable=trainable, groups=groups, weights=weights, blocks=blocks,
        )
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    if attention not in ATTENTIONS:
        raise ValueError(f"attention must be one of {ATTENTIONS}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}")
    from .scopes import compile_scoped, has_scopes
    from .blocks import _block_configs

    blocks = _block_configs(blocks)

    if has_scopes(fn):
        return compile_scoped(fn, inputs, static, dict(
            arch=arch, method=method, margin=margin, strict=strict,
            max_loop_iters=max_loop_iters, max_inline_depth=max_inline_depth,
            max_table=max_table, exhaustive_points=exhaustive_points, robust=robust,
            attention=attention, attn_tol=attn_tol, mlp=mlp, layernorm=layernorm,
            ln_tol=ln_tol, output=output, slope=slope, repr=repr, onehot_max=onehot_max,
            groups=groups, weights=weights,
        ), trainable, blocks)
    if blocks:
        raise ValueError("blocks requires a function with named neurify scopes")
    params = _param_domains(fn, inputs, static)
    host = Host()
    h = host.handle(fn)
    src, file, line = function_source(fn)
    graph, out_spec, warnings = _core.trace(
        src, fn.__name__, file, line, h, params, host,
        max_loop_iters=max_loop_iters, max_inline_depth=max_inline_depth,
        max_table=max_table, exhaustive_points=exhaustive_points,
    )
    source_graph = graph
    source_out_spec = out_spec
    graph = graph.encode_outputs(output, True, list(groups or []), None if weights is None else [float(w) for w in weights])
    core, report = _core.compile(
        graph, arch=arch, method=method, margin=margin, strict=strict,
        max_table=max_table, exhaustive_points=exhaustive_points, robust=robust,
        attention=attention, attn_tol=attn_tol, mlp=mlp, layernorm=layernorm, ln_tol=ln_tol,
        slope=slope, repr=repr, onehot_max=onehot_max,
    )
    report["output"] = output
    if groups is not None:
        report["groups"] = list(groups)
    if weights is not None:
        report["weights"] = [float(w) for w in weights]
    if output != "raw" or weights is not None:
        leaf = "int" if (output == "rank" or (output == "onehot" and weights is not None)) else "bool"
        if output == "raw" or any(float(w) != int(w) for w in (weights or [])):
            leaf = "real"
        out_spec = ("list", [("leaf", leaf)] * graph.num_outputs)
    net = Network(core, report, graph, fn, params, out_spec, warnings, source_graph=source_graph,
                  source_out_spec=source_out_spec)
    if trainable:
        net = net.prepare_for_training(**(trainable if isinstance(trainable, dict) else {}))
    return net
