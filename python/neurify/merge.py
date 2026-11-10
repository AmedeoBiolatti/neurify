"""merge(): several heuristics with the same inputs -> one exact network.

All heuristics are compiled as *one* program, so every subexpression they
compute the same way (helpers, features, intermediate boards) is built once
and shared by the hash-consed IR. Each heuristic's outputs are encoded on
their own (``output="rank"``: ranks 0..n-1 within that heuristic, so scores on
different scales become comparable) and then combined exactly:

* ``"stack"``: all encoded outputs, heuristic after heuristic (G * n);
* ``"priority"``: the first heuristic decides, the second breaks its ties,
  and so on: ``sum_g n^(G-1-g) * rank_g`` (exact, since ranks are < n);
* ``"vote"``: ``sum_g rank_g`` (Borda count);
* a list of G weights: ``sum_g w_g * rank_g``.

All combinations are linear in the encoded outputs, so they are points of
the same space: fine-tuning can move from one to another, and zero weights
between the heuristics' units (the "bridges") let training reuse what one
heuristic computes in another's decisions.
"""

from __future__ import annotations

import inspect

from . import _core
from ._host import Host, function_source
from .compiler import _param_domains, compile

COMBINES = ("stack", "priority", "vote")


def _num_outputs(fn, inputs, static, max_loop_iters, max_inline_depth, max_table, exhaustive_points) -> int:
    params = _param_domains(fn, inputs, static)
    host = Host()
    h = host.handle(fn)
    src, file, line = function_source(fn)
    graph, _, _ = _core.trace(
        src, fn.__name__, file, line, h, params, host,
        max_loop_iters=max_loop_iters, max_inline_depth=max_inline_depth,
        max_table=max_table, exhaustive_points=exhaustive_points,
    )
    return graph.num_outputs


def merged_function(fns, name: str = "merged"):
    """A Python function returning the concatenated outputs of ``fns``
    (lists), with the signature and annotations of ``fns[0]``."""
    fns = list(fns)
    if not fns:
        raise ValueError("merge needs at least one function")
    sig = inspect.signature(fns[0])
    names = list(sig.parameters)
    for f in fns[1:]:
        if list(inspect.signature(f).parameters) != names:
            raise TypeError(f"{f.__name__} has parameters {list(inspect.signature(f).parameters)}, expected {names}")
    args = ", ".join(names)
    body = "".join(f"    out = out + list(_h{i}({args}))\n" for i in range(len(fns)))
    src = f"def {name}({args}):\n    out = []\n{body}    return out\n"
    env = {f"_h{i}": f for i, f in enumerate(fns)}
    exec(src, env)  # noqa: S102 - generated from parameter names only
    fn = env[name]
    fn.__neurify_source__ = src
    fn.__signature__ = sig
    try:
        fn.__annotations__ = dict(inspect.get_annotations(fns[0], eval_str=True))
    except Exception:  # pragma: no cover
        fn.__annotations__ = dict(getattr(fns[0], "__annotations__", {}))
    return fn


def merge(
    fns,
    inputs: dict | None = None,
    *,
    combine="priority",
    output: str = "rank",
    static: dict | None = None,
    max_loop_iters: int = 1000,
    max_inline_depth: int = 64,
    max_table: int = 4096,
    exhaustive_points: int = 65536,
    **kwargs,
):
    """Compile several heuristics with the same parameters into one network.

    ``fns``: functions returning lists of scores (e.g. one score per move).
    ``combine``: ``"stack"``, ``"priority"``, ``"vote"`` or a list of weights
    (see module doc). ``output``: encoding of each heuristic's scores before
    combining (``"rank"`` default; ``"raw"`` only with explicit weights or
    ``"stack"``). Other keyword arguments go to ``neurify.compile``.

    ``net.report["groups"]`` holds the number of outputs of each heuristic.
    """
    fns = list(fns)
    fn = merged_function(fns)  # checks the signatures first
    sizes = [_num_outputs(f, inputs, static, max_loop_iters, max_inline_depth, max_table, exhaustive_points) for f in fns]
    G = len(fns)
    if isinstance(combine, str):
        if combine not in COMBINES:
            raise ValueError(f"combine must be one of {COMBINES} or a list of weights")
        if combine != "stack" and len(set(sizes)) != 1:
            raise ValueError(f"combine='{combine}' needs heuristics with equally many outputs, got {sizes}")
        if combine != "stack" and output == "raw":
            raise ValueError(f"combine='{combine}' needs encoded outputs (output='rank' or 'onehot')")
        n = sizes[0]
        base = n if output == "rank" else 2
        weights = {
            "stack": None,
            "priority": [base ** (G - 1 - g) for g in range(G)],
            "vote": [1] * G,
        }[combine]
    else:
        weights = [float(w) for w in combine]
        if len(weights) != G:
            raise ValueError(f"{len(weights)} weights for {G} heuristics")
    net = compile(
        fn, inputs, static=static, output=output, groups=sizes, weights=weights,
        max_loop_iters=max_loop_iters, max_inline_depth=max_inline_depth, max_table=max_table,
        exhaustive_points=exhaustive_points, **kwargs,
    )
    net.report["combine"] = combine if isinstance(combine, str) else list(weights)
    net.report["heuristics"] = [f.__name__ for f in fns]
    return net
