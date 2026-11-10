"""Compiled network wrapper: evaluation, structure, transforms, export."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core


def _dense(d: dict) -> dict:
    r, c = d["shape"]
    return {
        "weight": np.asarray(d["weight"], dtype=np.float64).reshape(r, c),
        "bias": np.asarray(d["bias"], dtype=np.float64),
    }


def flatten_inputs(spec: tuple, value: Any, out: list) -> None:
    tag = spec[0]
    if tag in ("real", "int", "bool"):
        out.append(float(value))
    elif tag in ("list", "tuple"):
        items = list(value)
        if len(items) != len(spec[1]):
            raise ValueError(f"expected {len(spec[1])} elements, got {len(items)}")
        for s, v in zip(spec[1], items):
            flatten_inputs(s, v, out)
    elif tag == "dict":
        for k, s in spec[1]:
            flatten_inputs(s, value[k], out)
    elif tag == "static":
        pass
    else:  # pragma: no cover
        raise ValueError(tag)


def unflatten_inputs(spec: tuple, it) -> Any:
    tag = spec[0]
    if tag == "real":
        return float(next(it))
    if tag == "int":
        return int(round(next(it)))
    if tag == "bool":
        return bool(round(next(it)))
    if tag in ("list", "seq"):
        return [unflatten_inputs(s, it) for s in spec[1]]
    if tag == "tuple":
        return tuple(unflatten_inputs(s, it) for s in spec[1])
    if tag == "dict":
        return {k: unflatten_inputs(s, it) for k, s in spec[1]}
    if tag == "static":
        return spec[1]
    raise ValueError(tag)  # pragma: no cover


def unflatten_outputs(spec: tuple, it, cast: bool) -> Any:
    tag = spec[0]
    if tag == "leaf":
        v = float(next(it))
        if cast and spec[1] == "int":
            return int(round(v))
        if cast and spec[1] == "bool":
            return bool(round(v))
        return v
    if tag == "none":
        return None
    if tag == "list":
        return [unflatten_outputs(s, it, cast) for s in spec[1]]
    if tag == "tuple":
        return tuple(unflatten_outputs(s, it, cast) for s in spec[1])
    if tag == "dict":
        return {k: unflatten_outputs(s, it, cast) for k, s in spec[1]}
    raise ValueError(tag)  # pragma: no cover


def flatten_outputs(spec: tuple, value: Any, out: list) -> None:
    tag = spec[0]
    if tag == "leaf":
        out.append(float(value))
    elif tag in ("list", "tuple"):
        for s, v in zip(spec[1], value):
            flatten_outputs(s, v, out)
    elif tag == "dict":
        for k, s in spec[1]:
            flatten_outputs(s, value[k], out)


def featurize(X, index, value):
    """Fixed input features: column j is X[:, index[j]], or the indicator
    [round(X[:, index[j]]) == value[j]] where value[j] is not NaN."""
    F = X[:, index]
    eq = ~np.isnan(value)
    if eq.any():
        F = np.where(eq, (np.round(F) == value).astype(F.dtype), F)
    return F


def _np_ln(l, x):
    if l is None:
        return x
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return l["gain"] * (x - m) / np.sqrt(v + l["eps"]) + l["bias"]


def _np_mlp(b, x):
    a = np.maximum(x @ b["w_in"]["weight"].T + b["w_in"]["bias"], 0.0)
    if "w_val" in b:
        a = a * (x @ b["w_val"]["weight"].T + b["w_val"]["bias"])
    return a @ b["w_out"]["weight"].T + b["w_out"]["bias"]


def _np_attention(a, x):
    B, T, _ = x.shape
    H, K = a["heads"], a["d_k"]
    q = (x @ a["w_q"]["weight"].T + a["w_q"]["bias"]).reshape(B, T, H, K)
    k = (x @ a["w_k"]["weight"].T + a["w_k"]["bias"]).reshape(B, T, H, K)
    v = x @ a["w_v"]["weight"].T + a["w_v"]["bias"]  # B, T, H
    sc = np.einsum("bthk,buhk->bhtu", q, k)
    if a["mode"] == "relu":
        w = np.maximum(sc, 0.0)
    elif a["mode"] == "hard":
        w = (np.arange(T) == sc.argmax(-1)[..., None]).astype(x.dtype)
    else:
        e = np.exp(sc - sc.max(-1, keepdims=True))
        w = e / e.sum(-1, keepdims=True)
    o = np.einsum("bhtu,buh->bth", w, v)
    return o @ a["w_o"]["weight"].T + a["w_o"]["bias"]


def _np_transformer(w, u):
    s = np.stack([u @ e["weight"].T + e["bias"] for e in w["tok_embed"]], axis=1)  # B, T, d
    for L in w["layers"]:
        if "attn" in L:
            s = s + _np_attention(L["attn"], _np_ln(L.get("ln1"), s))
        if "mlp" in L:
            s = s + _np_mlp(L["mlp"], _np_ln(L.get("ln2"), s))
    h = _np_ln(w["ln_f"], s[:, 0])
    return h @ w["readout"]["weight"].T + w["readout"]["bias"]


class Network:
    """An exactly-compiled neural network.

    Calling it like the original function evaluates the network; ``forward``
    works on batches of flat input vectors. Transforms (``balance``,
    ``widen``, ``deepen``, ...) return new networks computing the same
    function. ``to_torch`` / ``to_jax`` export it for training.
    """

    def __init__(self, core, report, graph, fn, params, out_spec, warnings, source_graph=None, source_out_spec=None):
        self._core = core
        self.report: dict = dict(report)
        self.graph = graph  # what the network computes (outputs possibly re-encoded)
        self.source_graph = graph if source_graph is None else source_graph  # the program as traced
        self.source_out_spec = out_spec if source_out_spec is None else source_out_spec  # its output structure
        self.fn = fn
        self._params = params  # [(name, spec)]
        self.out_spec = out_spec
        self.warnings: list[str] = list(warnings) + list(self.report.get("warnings", []))
        self._w = None

    # -------------------------------------------------------------- info

    @property
    def arch(self) -> str:
        return self._core.arch

    @property
    def depth(self) -> int:
        return self._core.depth

    @property
    def widths(self) -> list[int]:
        return list(self._core.widths)

    @property
    def d_model(self):
        return self._core.d_model

    @property
    def num_params(self) -> int:
        return self._core.params

    @property
    def heads(self) -> list[int]:
        """Attention heads per layer (transformer only)."""
        return list(self._core.heads)

    @property
    def n_tokens(self) -> int:
        """Tokens: 1 main token + sequence positions (transformer only)."""
        return self._core.n_tokens

    @property
    def n_in(self) -> int:
        return self._core.n_in

    @property
    def n_out(self) -> int:
        return self._core.n_out

    @property
    def exact(self) -> bool:
        """True iff the network equals the program on the whole declared domain."""
        return bool(self.report["exact"])

    @property
    def input_names(self) -> list[str]:
        return [s[0] for s in self.graph.input_specs()]

    def summary(self) -> str:
        r = self.report
        lines = [
            f"neurify.Network(arch={self.arch}, method={r['method']})",
            f"  inputs : {self.n_in} {self.input_names}",
            f"  outputs: {self.n_out}",
            f"  depth  : {self.depth}   widths: {self.widths}"
            + (f"   d_model: {self.d_model}" if self.d_model is not None else ""),
            f"  params : {self.num_params}   units: {r['units']}   lookups: {r['lookups']}",
        ]
        if self.arch == "transformer":
            lines.insert(4, f"  tokens : {self.n_tokens}   heads/layer: {self.heads}   attention: {r['attention']}")
        if r["exhaustive_points"]:
            lines.append(f"  exhaustive analysis over {r['exhaustive_points']} input points")
        if self.exact:
            lines.append("  exact  : yes, on the whole declared domain")
        else:
            if r["margin_sites"]:
                lines.append(
                    f"  exact  : outside a {r['margin']:g}-band around {r['margin_sites']} "
                    "real-valued decision boundaries"
                )
            for a in r.get("approx", []):
                lines.append(f"  approx : {a}")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<neurify.Network arch={self.arch} depth={self.depth} widths={self.widths} "
            f"params={self.num_params} exact={self.exact}>"
        )

    # -------------------------------------------------------------- eval

    def weights(self) -> dict:
        """All weights as numpy arrays (float64)."""
        if self._w is None:
            d = self._core.to_dict()
            w = {
                "arch": d["arch"],
                "body": d["body"],
                "in_offset": np.asarray(d["in_offset"]),
                "in_scale": np.asarray(d["in_scale"]),
                "feat_index": np.asarray([f[0] for f in d["features"]], dtype=np.int64),
                "feat_value": np.asarray([f[1] for f in d["features"]], dtype=np.float64),
            }
            if d["body"] == "mlp":
                w["hidden"] = [_dense(h) for h in d["hidden"]]
                w["out"] = _dense(d["out"])
            elif d["body"] == "transformer":
                w["block"] = d["block"]
                w["anchor"] = d["anchor"]
                w["tok_embed"] = [_dense(e) for e in d["tok_embed"]]
                ln = lambda x: {"gain": np.asarray(x["gain"]), "bias": np.asarray(x["bias"]), "eps": x["eps"]}  # noqa: E731
                layers = []
                for l in d["layers"]:
                    L = {}
                    for k in ("ln1", "ln2"):
                        if k in l:
                            L[k] = ln(l[k])
                    if "attn" in l:
                        a = l["attn"]
                        L["attn"] = {"mode": a["mode"], "heads": a["heads"], "d_k": a["d_k"],
                                     **{k: _dense(a[k]) for k in ("w_q", "w_k", "w_v", "w_o")}}
                    if "mlp" in l:
                        L["mlp"] = {k: _dense(v) for k, v in l["mlp"].items()}
                    layers.append(L)
                w["layers"] = layers
                w["ln_f"] = ln(d["ln_f"]) if "ln_f" in d else None
                w["readout"] = _dense(d["readout"])
            else:
                w["block"] = d["block"]
                w["embed"] = _dense(d["embed"])
                w["blocks"] = [{k: _dense(v) for k, v in b.items()} for b in d["blocks"]]
                w["readout"] = _dense(d["readout"])
            self._w = w
        return self._w

    def _weights_as(self, dtype) -> dict:
        def cast(x):
            if isinstance(x, np.ndarray):
                return x.astype(dtype)
            if isinstance(x, dict):
                return {k: v if k.startswith("feat_") else cast(v) for k, v in x.items()}
            if isinstance(x, list):
                return [cast(v) for v in x]
            return x

        return cast(self.weights())

    def flatten_inputs(self, *args, **kwargs) -> np.ndarray:
        from .interface import flatten_single

        return flatten_single(self, args, kwargs)

    @property
    def __signature__(self):
        import inspect

        return inspect.signature(self.fn)

    def forward(self, X, dtype=np.float64) -> np.ndarray:
        """Evaluate on raw inputs. X: (batch, n_in) or (n_in,). ``dtype`` sets
        the arithmetic precision (``np.float32`` mimics float32 training)."""
        X = np.asarray(X, dtype=dtype)
        single = X.ndim == 1
        X = X.reshape(-1, self.n_in) if self.n_in else np.zeros((max(1, len(X)), 0), dtype)
        w = self.weights() if dtype == np.float64 else self._weights_as(dtype)
        u = (featurize(X, w["feat_index"], w["feat_value"]) - w["in_offset"]) / w["in_scale"]
        if w["body"] == "mlp":
            h = u
            for d in w["hidden"]:
                h = np.maximum(h @ d["weight"].T + d["bias"], 0.0)
            y = h @ w["out"]["weight"].T + w["out"]["bias"]
        elif w["body"] == "transformer":
            y = _np_transformer(w, u)
        else:
            s = u @ w["embed"]["weight"].T + w["embed"]["bias"]
            for b in w["blocks"]:
                a = np.maximum(s @ b["w_in"]["weight"].T + b["w_in"]["bias"], 0.0)
                if "w_val" in b:
                    a = a * (s @ b["w_val"]["weight"].T + b["w_val"]["bias"])
                s = s + a @ b["w_out"]["weight"].T + b["w_out"]["bias"]
            y = s @ w["readout"]["weight"].T + w["readout"]["bias"]
        return y[0] if single else y

    def __call__(self, *args, **kwargs):
        from .interface import prediction_call

        return prediction_call(self, args, kwargs)

    def as_function(self, *, cast=False):
        """A callable with the original parameters and structured outputs."""
        from .interface import numpy_function

        return numpy_function(self, cast=cast)

    def inspect(self, *args, **kwargs):
        """Return Evaluation(output, exposed) using the original arguments."""
        return self.as_function().inspect(*args, **kwargs)

    def reference(self, X, margin: float | None = None):
        """Evaluate the traced IR (exact program semantics) on a batch.
        Returns (Y, near_margin_flags)."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, max(self.n_in, 1) if self.n_in else 1)
        flat = X.ravel().tolist() if self.n_in else []
        y, near = self.graph.eval_batch(flat, margin)
        return np.asarray(y).reshape(-1, self.n_out), np.asarray(near, dtype=bool)

    def source_reference(self, X, margin: float | None = None):
        """Like ``reference`` but for the program's own (un-encoded) outputs."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, max(self.n_in, 1) if self.n_in else 1)
        flat = X.ravel().tolist() if self.n_in else []
        y, near = self.source_graph.eval_batch(flat, margin)
        return np.asarray(y).reshape(len(X), -1), np.asarray(near, dtype=bool)

    # -------------------------------------------------------- transforms

    def _derive(self, core) -> "Network":
        n = Network(core, self.report, self.graph, self.fn, self._params, self.out_spec, [], self.source_graph,
                    self.source_out_spec)
        n.warnings = list(self.warnings)
        n.report = dict(
            self.report, depth=core.depth, widths=list(core.widths), params=core.params, d_model=core.d_model,
            heads=list(core.heads),
        )
        return n

    def normalize_inputs(self) -> "Network":
        """Consume inputs rescaled to [-1, 1] internally (raw inputs still accepted)."""
        c = self._core.copy()
        c.normalize_inputs()
        return self._derive(c)

    def balance(self, sweeps: int = 3) -> "Network":
        """Equalise per-unit in/out weight norms (ReLU positive homogeneity)."""
        c = self._core.copy()
        c.balance(sweeps)
        return self._derive(c)

    def widen(self, extra: int, extra_dims: int = 0, scale: float = 1.0, seed: int = 0) -> "Network":
        """Add `extra` spare units per layer (random in, zero out); residual nets
        can also get `extra_dims` spare stream dimensions."""
        c = self._core.copy()
        c.widen(extra, extra_dims, scale, seed)
        return self._derive(c)

    def deepen(self, n: int, width: int = 8, scale: float = 1.0, seed: int = 0) -> "Network":
        """Insert `n` identity layers / zero-output residual blocks."""
        c = self._core.copy()
        c.deepen(n, width, scale, seed)
        return self._derive(c)

    def normalize_activations(self, X=None, n: int = 2048, seed: int = 0) -> "Network":
        """Rescale units so their largest activation on sample inputs is O(1)
        (incoming / 2^j, outgoing * 2^j: exact). ``X``: raw input batch;
        default: ``n`` samples of the declared domain. Big internal
        activations (big-M constants, sentinel values) otherwise make
        gradient steps blow up. Not available for transformers (no-op)."""
        if X is None:
            from .verify import _sample

            X, _ = _sample(self.graph.input_specs(), n, np.random.default_rng(seed), n)
        c = self._core.copy()
        if not c.normalize_activations(np.asarray(X, dtype=np.float64).ravel().tolist()):
            self.warnings.append("normalize_activations: not supported for this architecture; skipped")
        return self._derive(c)

    def prepare_for_training(
        self,
        width: int | None = None,
        depth: int | None = None,
        extra_dims: int = 0,
        scale: float = 1.0,
        seed: int = 0,
        normalize: bool = True,
        scaling: str | None = "weights",
        samples=None,
    ) -> "Network":
        """Function-preserving initialisation for fine-tuning:

        1. normalise inputs to [-1, 1];
        2. ``scaling="weights"`` (default): balance per-unit in/out weight
           norms, spreading the program's dynamic range evenly over layers;
           ``"activations"``: rescale units to O(1) activations on
           ``samples`` (pushes large magnitudes into outgoing weights, which
           conditions badly for programs with big constants); ``None``:
           leave as compiled;
        3. reach the target depth / width with spare capacity (identity
           layers, random-in / zero-out units, spare stream dims).
        """
        net = self.normalize_inputs() if normalize else self
        if scaling == "activations":
            net = net.normalize_activations(samples, seed=seed)
        elif scaling == "weights":
            net = net.balance()
        elif scaling is not None:
            raise ValueError("scaling must be 'activations', 'weights' or None")
        if depth is not None and depth > net.depth:
            net = net.deepen(depth - net.depth, width=width or 8, scale=scale, seed=seed)
        if width is not None and net.widths:
            extra = max(0, width - min(net.widths))
            if extra or extra_dims:
                net = net.widen(extra, extra_dims, scale, seed + 1)
        elif extra_dims:
            net = net.widen(0, extra_dims, scale, seed + 1)
        return net

    # ------------------------------------------------------------ export

    def to_torch(self, dtype=None):
        """A torch.nn.Module computing this network (float32 by default)."""
        from .export_torch import to_torch
        from .interface import attach_torch_interface

        return attach_torch_interface(to_torch(self, dtype), self)

    def to_torch_residual(self, hidden=(256, 256), onehot_max: int = 16, dtype=None):
        """``y = exact(x) + delta(features(x))`` as a torch module: the
        compiled network frozen, plus a standard MLP (last layer zero, so it
        starts exact) on one-hot / normalised inputs. The recommended way to
        fine-tune with standard optimisers and learning rates: exact
        networks of programs with large internal ranges do not survive plain
        Adam steps (see ``neurify.trainability_report``), this model can only
        add to the program. For decision programs combine it with
        ``compile(output="rank")``."""
        from .export_torch import to_torch_residual
        from .interface import attach_torch_interface

        return attach_torch_interface(to_torch_residual(self, hidden, onehot_max, dtype), self)

    def to_jax_residual(self, hidden=(256, 256), onehot_max: int = 16, seed: int = 0, dtype=None):
        """JAX version of ``to_torch_residual``: (apply, params), params = the
        trainable branch only."""
        from .export_jax import to_jax_residual
        from .interface import attach_jax_interface

        apply, params = to_jax_residual(self, hidden, onehot_max, seed, dtype)
        return attach_jax_interface(apply, self, dtype), params

    def to_jax(self, dtype=None):
        """(apply_fn, params): ``apply_fn(params, x)`` is jit/grad-compatible."""
        from .export_jax import to_jax
        from .interface import attach_jax_interface

        apply, params = to_jax(self, dtype)
        return attach_jax_interface(apply, self, dtype), params


__all__ = ["Network"]
_ = _core
