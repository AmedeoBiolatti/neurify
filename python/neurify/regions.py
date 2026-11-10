"""Named compilation boundaries for selective training and observation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import itertools
import math
from types import MappingProxyType

import numpy as np

from .network import Network


def _policy(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


@dataclass(frozen=True)
class Region:
    """One independently compiled part of a program.

    ``trainable=False`` freezes its parameters, while preserving derivatives
    with respect to its inputs. ``differentiable=False`` stops derivatives at
    its output. Neither option changes its initial numerical function.
    """

    name: str
    network: Network
    trainable: bool = True
    differentiable: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("region name must be a non-empty string")
        if not isinstance(self.network, Network) and not getattr(self.network, "_is_activation_block", False):
            raise TypeError("region network must be a compiled neurify.Network")
        _policy(self.trainable, "trainable")
        _policy(self.differentiable, "differentiable")

    def __call__(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def with_controls(self, *, trainable=None, differentiable=None):
        """Return a region with updated controls and the same initial network."""
        return replace(
            self,
            trainable=self.trainable if trainable is None else _policy(trainable, "trainable"),
            differentiable=self.differentiable if differentiable is None else _policy(differentiable, "differentiable"),
        )


def region(fn=None, *, name=None, trainable=True, differentiable=True, **compile_options):
    """Compile a named helper, or wrap an already compiled network.

    Can also be used as ``@neurify.region(trainable=False)``. Compiler options
    (``inputs``, ``arch``, ``slope``, etc.) apply to this region only.
    """
    if fn is None:
        return functools.partial(
            region, name=name, trainable=trainable, differentiable=differentiable, **compile_options
        )
    if isinstance(fn, Network) or getattr(fn, "_is_activation_block", False):
        if compile_options:
            raise TypeError("compiler options cannot be applied to an already compiled network")
        net = fn
    else:
        from .compiler import compile

        net = compile(fn, **compile_options)
    if name is None:
        name = getattr(net.fn, "__name__", None)
    return Region(name, net, trainable, differentiable)


def _batch(X, n_in, dtype):
    X = np.asarray(X, dtype=dtype)
    single = X.ndim == 1
    if X.ndim not in (1, 2) or X.shape[-1] != n_in:
        raise ValueError(f"expected inputs with shape ({n_in},) or (batch, {n_in}), got {X.shape}")
    return (X[None, :] if single else X), single


def _leaf_kinds(spec):
    if spec[0] == "leaf":
        return [spec[1]]
    if spec[0] == "none":
        return []
    items = [s for _, s in spec[1]] if spec[0] == "dict" else spec[1]
    return [kind for s in items for kind in _leaf_kinds(s)]


def _structured(spec, values):
    """Rebuild an output tree using array/tensor views, preserving autograd."""
    index = 0

    def build(s):
        nonlocal index
        tag = s[0]
        if tag == "leaf":
            value = values[..., index]
            index += 1
            return value
        if tag == "none":
            return None
        if tag == "dict":
            return {key: build(item) for key, item in s[1]}
        items = [build(item) for item in s[1]]
        return tuple(items) if tag == "tuple" else items

    return build(spec)


def _from_leaves(spec, leaves):
    """Rebuild a tree using the existing activation objects, without views."""
    if spec[0] == "leaf":
        return next(leaves)
    if spec[0] == "none":
        return None
    if spec[0] == "dict":
        return {key: _from_leaves(child, leaves) for key, child in spec[1]}
    values = [_from_leaves(child, leaves) for child in spec[1]]
    return tuple(values) if spec[0] == "tuple" else values


def _dense_bounds(dense, lo, hi):
    w = dense["weight"]
    # Avoid 0 * infinity for unbounded, unused inputs.
    a = np.zeros_like(w)
    b = np.zeros_like(w)
    with np.errstate(invalid="ignore", over="ignore"):
        np.multiply(w, lo, out=a, where=w != 0)
        np.multiply(w, hi, out=b, where=w != 0)
        return np.minimum(a, b).sum(axis=1) + dense["bias"], np.maximum(a, b).sum(axis=1) + dense["bias"]


def _interval_outputs(net):
    """Conservative bounds of the initialized network; None if unavailable."""
    if getattr(net, "_is_activation_block", False):
        return net.output_contract()
    if hasattr(net.graph, "output_specs"):
        specs = net.graph.output_specs()
        return (
            np.array([s[1] for s in specs]), np.array([s[2] for s in specs]),
            [s[0] for s in specs], False,
        )
    w = net.weights()
    if w["body"] == "transformer":
        if w["ln_f"] is not None or any(
            "attn" in layer or ("mlp" in layer and layer.get("ln2") is not None) for layer in w["layers"]
        ):
            return None
    specs = net.graph.input_specs()
    lo = np.array([specs[i][2] if np.isnan(v) else 0.0 for i, v in zip(w["feat_index"], w["feat_value"])])
    hi = np.array([specs[i][3] if np.isnan(v) else 1.0 for i, v in zip(w["feat_index"], w["feat_value"])])
    with np.errstate(invalid="ignore", over="ignore"):
        a = (lo - w["in_offset"]) / w["in_scale"]
        b = (hi - w["in_offset"]) / w["in_scale"]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if w["body"] == "mlp":
            for dense in w["hidden"]:
                lo, hi = _dense_bounds(dense, lo, hi)
                lo, hi = np.maximum(lo, 0), np.maximum(hi, 0)
            lo, hi = _dense_bounds(w["out"], lo, hi)
        else:
            transformer = w["body"] == "transformer"
            embed = w["tok_embed"][0] if transformer else w["embed"]
            blocks = [layer["mlp"] for layer in w["layers"] if "mlp" in layer] if transformer else w["blocks"]
            lo, hi = _dense_bounds(embed, lo, hi)
            for block in blocks:
                al, ah = _dense_bounds(block["w_in"], lo, hi)
                al, ah = np.maximum(al, 0), np.maximum(ah, 0)
                if "w_val" in block:
                    bl, bh = _dense_bounds(block["w_val"], lo, hi)
                    products = np.array([al * bl, al * bh, ah * bl, ah * bh])
                    al, ah = products.min(axis=0), products.max(axis=0)
                dl, dh = _dense_bounds(block["w_out"], al, ah)
                lo, hi = lo + dl, hi + dh
            lo, hi = _dense_bounds(w["readout"], lo, hi)
    if np.isnan(lo).any() or np.isnan(hi).any():
        return None
    return lo, hi, _leaf_kinds(net.out_spec), False


def _output_contract(net):
    specs = net.graph.input_specs()
    count = 1
    for _, kind, lo, hi in specs:
        if kind == "real" or not (math.isfinite(lo) and math.isfinite(hi)):
            return _interval_outputs(net)
        count *= int(hi - lo) + 1
        if count > 65536 or count * net.graph.num_nodes > 20_000_000:
            return _interval_outputs(net)
    axes = [range(int(lo), int(hi) + 1) for _, _, lo, hi in specs]
    X = np.array(list(itertools.product(*axes)), dtype=float).reshape(count, len(specs))
    values, _ = net.graph.eval_batch(X.ravel().tolist())
    values = np.asarray(values).reshape(count, net.n_out)
    kinds = []
    for col in values.T:
        kinds.append("bool" if np.isin(col, [0, 1]).all() else "int" if (col == np.round(col)).all() else "real")
    return values.min(axis=0), values.max(axis=0), kinds, True


def _check_boundary(previous, current):
    if previous.network.n_out != current.network.n_in:
        raise ValueError(
            f"region '{previous.name}' produces {previous.network.n_out} values, "
            f"but '{current.name}' expects {current.network.n_in} inputs"
        )
    contract = _output_contract(previous.network)
    if contract is None:
        return f"input-domain compatibility of '{previous.name}' -> '{current.name}' is not proven"
    lo, hi, kinds, enumerated = contract
    for i, (name, kind, lower, upper) in enumerate(current.network.graph.input_specs()):
        fits = lower <= lo[i] and hi[i] <= upper
        integral = kind == "real" or kinds[i] != "real" or (
            math.isfinite(lo[i]) and lo[i] == hi[i] == round(lo[i])
        )
        boolean = kind != "bool" or (0 <= lo[i] and hi[i] <= 1)
        if not (fits and integral and boolean):
            message = (
                f"region '{previous.name}' output {i} may fall outside '{current.name}' "
                f"input '{name}' domain {kind}[{lower}, {upper}]"
            )
            if enumerated:
                raise ValueError(message)
            return message + "; domain compatibility is not proven"
    return None


class Pipeline:
    """A sequence of named networks with preserved compilation boundaries.

    Each region's flat outputs, in return-structure order, become the next
    region's flat inputs in parameter order. Intermediate helper annotations
    must include all values the preceding region can produce.

    ``verify`` compares the composed network with the composed IR references.
    To also check tracing against Python source, verify each region's network.
    """

    def __init__(self, regions):
        self._regions = tuple(regions)
        if not self._regions or not all(isinstance(r, Region) for r in self._regions):
            raise TypeError("pipeline requires one or more neurify.Region objects")
        names = [r.name for r in self._regions]
        if len(set(names)) != len(names):
            raise ValueError("pipeline region names must be unique")
        self.regions = MappingProxyType(dict(zip(names, self._regions)))
        boundary_warnings = [
            warning for a, b in zip(self._regions, self._regions[1:])
            if (warning := _check_boundary(a, b)) is not None
        ]
        first, last = self._regions[0].network, self._regions[-1].network
        self.graph = self.source_graph = first.graph  # input specs for verification
        self._params = first._params
        self.out_spec = self.source_out_spec = last.out_spec
        self.fn = None  # verify the composed IR; source checks are per region
        self.warnings = [w for r in self._regions for w in r.network.warnings] + boundary_warnings
        self.report = {
            "arch": "pipeline", "method": "regions", "depth": self.depth,
            "params": self.num_params, "widths": self.widths,
            "units": sum(r.network.report["units"] for r in self._regions),
            "margin_sites": sum(r.network.report["margin_sites"] for r in self._regions),
            "margin": max(r.network.report["margin"] for r in self._regions),
            "approx": [a for r in self._regions for a in r.network.report.get("approx", [])] + boundary_warnings,
            "exact": all(r.network.exact for r in self._regions) and not boundary_warnings,
            "exhaustive_points": 0,
            "regions": names,
        }

    @property
    def region_names(self):
        return tuple(self.regions)

    @property
    def n_in(self):
        return self._regions[0].network.n_in

    @property
    def n_out(self):
        return self._regions[-1].network.n_out

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
    def arch(self):
        return "pipeline"

    def summary(self):
        lines = [f"neurify.Pipeline(inputs={self.n_in}, outputs={self.n_out}, params={self.num_params})"]
        for r in self._regions:
            lines.append(
                f"  {r.name}: {r.network.arch}, {r.network.n_in} -> {r.network.n_out}, "
                f"trainable={r.trainable}, differentiable={r.differentiable}"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)

    def flatten_inputs(self, *args, **kwargs):
        return self._regions[0].network.flatten_inputs(*args, **kwargs)

    @property
    def __signature__(self):
        import inspect

        return inspect.signature(self._regions[0].network.fn)

    def _run(self, X, dtype, capture):
        values, single = _batch(X, self.n_in, dtype)
        observations = {}
        for r in self._regions:
            values = r.network.forward(values, dtype=dtype) if len(values) else np.empty((0, r.network.n_out), dtype=dtype)
            if capture:
                observations[r.name] = (values[0] if single else values).copy()
        return (values[0] if single else values), observations

    def forward(self, X, dtype=np.float64):
        return self._run(X, dtype, False)[0]

    def forward_with_intermediates(self, X, dtype=np.float64, *, structured=False):
        """Return ``(output, {region_name: flat_outputs})`` from this network.

        Arrays are independent snapshots. Torch/JAX exports expose the same
        observations from their actual, potentially trained, parameters.
        """
        _policy(structured, "structured")
        output, observations = self._run(X, dtype, True)
        if structured:
            observations = {
                r.name: _structured(r.network.out_spec, observations[r.name]) for r in self._regions
            }
        return output, observations

    def __call__(self, *args, **kwargs):
        from .interface import prediction_call

        return prediction_call(self, args, kwargs)

    def as_function(self, *, cast=False):
        from .interface import numpy_function

        return numpy_function(self, cast=cast)

    def inspect(self, *args, **kwargs):
        return self.as_function().inspect(*args, **kwargs)

    def reference(self, X, margin=None):
        values, _ = _batch(X, self.n_in, np.float64)
        near = np.zeros(len(values), dtype=bool)
        for r in self._regions:
            net = r.network
            if not len(values):
                values = np.empty((0, net.n_out))
                continue
            flat, flags = net.graph.eval_batch(values.ravel().tolist(), margin)
            if net.n_in == 0:
                values = np.repeat(np.asarray(flat)[None, :], len(values), axis=0)
                near |= flags[0]
            else:
                values = np.asarray(flat).reshape(len(values), net.n_out)
                near |= np.asarray(flags, dtype=bool)
        return values, near

    source_reference = reference

    def prepare_for_training(self, **kwargs):
        """Apply function-preserving training transforms to every region."""
        return Pipeline([replace(r, network=r.network.prepare_for_training(**kwargs)) for r in self._regions])

    def to_torch(self, dtype=None):
        from .export_regions_torch import RegionPipeline
        from .interface import attach_torch_interface

        return attach_torch_interface(RegionPipeline(self, dtype), self)

    def to_jax(self, dtype=None):
        from .export_regions_jax import to_jax
        from .interface import attach_jax_interface

        apply, params = to_jax(self, dtype)
        return attach_jax_interface(apply, self, dtype), params


def pipeline(*regions):
    """Compose ``region`` objects in order; accepts varargs or one iterable."""
    if len(regions) == 1 and not isinstance(regions[0], Region):
        try:
            regions = tuple(regions[0])
        except TypeError:
            raise TypeError("pipeline requires neurify.Region objects or an iterable of them") from None
    return Pipeline(regions)
