"""Empirical exactness check of a compiled network."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .network import Network, flatten_outputs, unflatten_inputs
from .interface import invoke_original


@dataclass
class VerifyResult:
    ok: bool
    n_checked: int
    n_skipped: int
    max_abs_err: float
    exhaustive: bool
    failures: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        how = "all" if self.exhaustive else "sampled"
        s = (
            f"{'OK' if self.ok else 'FAILED'}: {self.n_checked} {how} points, max |err| = {self.max_abs_err:.3g}"
        )
        if self.n_skipped:
            s += f", {self.n_skipped} skipped (in margin bands or outside the domain)"
        for f in self.failures[:5]:
            s += f"\n  at {f['input']}: expected {f['expected']}, got {f['got']}"
        return s


def _sample(specs, n, rng, exhaustive_limit):
    sizes = []
    for _, kind, lo, hi in specs:
        if kind == "real" or not (math.isfinite(lo) and math.isfinite(hi)):
            sizes = None
            break
        sizes.append(int(hi - lo) + 1)
    if sizes is not None and math.prod(sizes) <= exhaustive_limit:
        axes = [np.arange(lo, hi + 1) for _, _, lo, hi in specs]
        X = np.array(list(itertools.product(*axes)), dtype=np.float64).reshape(-1, len(specs))
        return X, True
    cols = []
    for _, kind, lo, hi in specs:
        if kind == "real":
            if math.isfinite(lo) and math.isfinite(hi):
                c = rng.uniform(lo, hi, n)
                c[: min(2, n)] = [lo, hi][: min(2, n)]
            else:
                c = np.clip(rng.normal(0, 10, n), lo, hi)
        else:
            a = lo if math.isfinite(lo) else -100
            b = hi if math.isfinite(hi) else 100
            c = rng.integers(int(a), int(b) + 1, n).astype(np.float64)
        cols.append(c)
    return np.stack(cols, axis=1), False


def verify(
    net: Network,
    fn=None,
    *,
    n: int = 2000,
    seed: int = 0,
    exhaustive_limit: int = 100_000,
    python_checks: int = 2000,
    atol: float | None = None,
    rtol: float | None = None,
    dtype=np.float64,
) -> VerifyResult:
    """Compare the network against the program.

    The reference is the traced IR (exact semantics, fast). If ``fn`` is
    given (default: the compiled function), up to ``python_checks`` points
    are also checked against real Python execution. Small finite domains are
    enumerated exhaustively. Inputs within the margin band of real-valued
    decision boundaries, and inputs on which Python raises, are skipped.
    """
    if atol is None:
        atol = 1e-6 if not net.report.get("approx") else 1e-4
    if rtol is None:
        rtol = 1e-6 if not net.report.get("approx") else 1e-4
    fn = net.fn if fn is None else fn
    rng = np.random.default_rng(seed)
    specs = net.graph.input_specs()
    if not specs:
        X = np.zeros((1, 0))
        exhaustive = True
    else:
        X, exhaustive = _sample(specs, n, rng, exhaustive_limit)
    margin = net.report["margin"] if net.report["margin_sites"] else None
    Y_ref, near = net.reference(X, margin)
    Y_src = Y_ref if net.source_graph is net.graph else net.source_reference(X)[0]
    skip = near.copy()
    if fn is not None and len(X):
        idx = np.arange(len(X)) if len(X) <= python_checks else rng.choice(len(X), python_checks, replace=False)
        for i in idx:
            it = iter(X[i])
            args = {name: unflatten_inputs(spec, it) for name, spec in net._params}
            try:
                out = invoke_original(fn, args)
            except Exception:
                skip[i] = True
                continue
            flat: list = []
            flatten_outputs(net.source_out_spec, out, flat)
            if not np.all(np.abs(np.asarray(flat) - Y_src[i]) <= 1e-6 + 1e-6 * np.abs(Y_src[i])):
                raise AssertionError(
                    f"frontend bug: traced program disagrees with Python at {args}: {flat} vs {Y_src[i].tolist()}"
                )
    Y = net.forward(X, dtype=dtype).astype(np.float64).reshape(len(X), -1)
    err = np.abs(Y - Y_ref)
    bad = ~np.all(err <= atol + rtol * np.abs(Y_ref), axis=1) & ~skip  # NaN counts as failure
    failures = [
        {"input": X[i].tolist(), "expected": Y_ref[i].tolist(), "got": Y[i].tolist()} for i in np.flatnonzero(bad)[:20]
    ]
    checked = ~skip
    return VerifyResult(
        ok=not bad.any(),
        n_checked=int(checked.sum()),
        n_skipped=int(skip.sum()),
        max_abs_err=float(np.nanmax(np.where(np.isnan(err), np.inf, err)[checked])) if checked.any() and err.size else 0.0,
        exhaustive=exhaustive,
        failures=failures,
    )
