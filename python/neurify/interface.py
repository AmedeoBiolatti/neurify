"""Function-shaped calls and a single-pass output/observation interface."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np


class Evaluation(NamedTuple):
    """The structured output and all exposed values from one network run."""

    output: Any
    exposed: dict


def _equal_static(a, b):
    if isinstance(b, Mapping):
        return isinstance(a, Mapping) and set(a) == set(b) and all(_equal_static(a[k], b[k]) for k in b)
    if isinstance(b, (list, tuple)):
        return isinstance(a, type(b)) and len(a) == len(b) and all(_equal_static(x, y) for x, y in zip(a, b))
    try:
        return bool(np.all(a == b))
    except (TypeError, ValueError, RuntimeError):
        return False


def _shape(spec):
    if spec[0] in ("real", "int", "bool"):
        return ()
    if spec[0] not in ("list", "tuple", "seq") or not spec[1]:
        return None
    shapes = [_shape(child) for child in spec[1]]
    if shapes[0] is None or any(child != shapes[0] for child in shapes):
        return None
    return (len(shapes),) + shapes[0]


def _flatten(spec, value, name, leaves):
    tag = spec[0]
    if tag in ("real", "int", "bool"):
        leaves.append(value)
    elif tag == "static":
        if not _equal_static(value, spec[1]):
            raise ValueError(f"static argument '{name}' is fixed at compile time")
    elif tag == "dict":
        keys = {key for key, _ in spec[1]}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"argument '{name}' must have the declared dictionary keys")
        for key, child in spec[1]:
            _flatten(child, value[key], f"{name}[{key!r}]", leaves)
    elif tag in ("list", "tuple", "seq"):
        shape = _shape(spec)
        array_shape = getattr(value, "shape", None)
        if array_shape is not None and shape is not None:
            depth = len(shape)
            if len(array_shape) < depth or tuple(array_shape[-depth:]) != shape:
                raise ValueError(f"argument '{name}' needs trailing shape {shape}, got {tuple(array_shape)}")
            for i, child in enumerate(spec[1]):
                selection = (slice(None),) * (len(array_shape) - depth) + (i,) + (slice(None),) * (depth - 1)
                _flatten(child, value[selection], f"{name}[{i}]", leaves)
        else:
            try:
                items = list(value)
            except TypeError:
                raise ValueError(f"argument '{name}' must have {len(spec[1])} elements") from None
            if len(items) != len(spec[1]):
                raise ValueError(f"argument '{name}' must have {len(spec[1])} elements, got {len(items)}")
            for i, (child, item) in enumerate(zip(spec[1], items)):
                _flatten(child, item, f"{name}[{i}]", leaves)
    else:
        raise ValueError(f"unsupported input spec '{tag}'")


def _exposure_spec(net):
    if hasattr(net, "steps"):
        items = []
        for step in net.steps:
            if step.named:
                specs = dict(step.region.network.out_spec[1])
                if getattr(step.region.network, "_is_activation_block", False):
                    specs.update(step.region.network._activation_specs)
                items.append((step.region.name, ("dict", [(n, specs[n]) for n in step.exposed])))
        return ("dict", items)
    if hasattr(net, "_regions"):
        return ("dict", [(r.name, r.network.out_spec) for r in net._regions])
    return ("dict", [])


class FunctionBinding:
    """Source signature and input/output trees, independent of model weights."""

    def __init__(self, net):
        fn = net.fn
        if fn is None and hasattr(net, "_regions"):
            fn = net._regions[0].network.fn
        self.name = fn.__name__
        self.qualname = fn.__qualname__
        self.module = fn.__module__
        self.doc = fn.__doc__
        self.signature = inspect.signature(fn)
        self.params = net._params
        self.output = net.out_spec
        self.exposure = _exposure_spec(net)
        self.n_out = net.n_out

    def bind(self, args, kwargs):
        bound = self.signature.bind_partial(*args, **kwargs)
        explicit = set(bound.arguments)
        bound.apply_defaults()
        leaves = []
        for name, spec in self.params:
            if spec[0] == "static":
                if name in explicit and not _equal_static(bound.arguments[name], spec[1]):
                    raise ValueError(f"static argument '{name}' is fixed at compile time")
                continue
            if name not in bound.arguments:
                raise TypeError(f"missing required argument: '{name}'")
            _flatten(spec, bound.arguments[name], name, leaves)
        return leaves


def _tree(spec, leaf):
    tag = spec[0]
    if tag == "leaf":
        return leaf(spec[1])
    if tag == "none":
        return None
    if tag == "dict":
        return {key: _tree(child, leaf) for key, child in spec[1]}
    values = [_tree(child, leaf) for child in spec[1]]
    return tuple(values) if tag == "tuple" else values


class FunctionInterface:
    """Call a network with its source function's parameters and output tree.

    ``inspect`` returns Evaluation(output, exposed) from a single execution.
    The adapter retains the live model, so it observes subsequent training.
    """

    def __init__(self, binding, evaluate, observe, *, backend="numpy", model=None, dtype=None, cast=False, detach=True, preserve_batch=False):
        from .regions import _policy

        self.binding = binding
        self.evaluate, self.observe = evaluate, observe
        self.backend, self.model, self.dtype = backend, model, dtype
        self.preserve_batch = preserve_batch
        self.cast = _policy(cast, "cast")
        self.detach = _policy(detach, "detach")
        self.__signature__ = binding.signature
        self.__name__, self.__qualname__ = binding.name, binding.qualname
        self.__module__, self.__doc__ = binding.module, binding.doc

    def _pack(self, leaves):
        if self.backend == "numpy":
            arrays = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in leaves]) if leaves else []
            batch = tuple(arrays[0].shape) if arrays else ()
            values = np.stack(arrays, axis=-1).reshape(-1, len(arrays)) if arrays else np.empty((1, 0))
        elif self.backend == "torch":
            import torch

            parameter = next(self.model.parameters(), None)
            if parameter is None:
                parameter = next(self.model.buffers(), None)
            dtype = parameter.dtype if parameter is not None else torch.float32
            device = parameter.device if parameter is not None else None
            arrays = torch.broadcast_tensors(*[torch.as_tensor(v, dtype=dtype, device=device) for v in leaves]) if leaves else []
            batch = tuple(arrays[0].shape) if arrays else ()
            values = torch.stack(arrays, dim=-1) if arrays else torch.empty((0,) if self.preserve_batch else (1, 0), dtype=dtype, device=device)
            if arrays and not self.preserve_batch:
                values = values.reshape(-1, len(arrays))
        else:
            import jax.numpy as jnp

            arrays = jnp.broadcast_arrays(*[jnp.asarray(v, dtype=self.dtype) for v in leaves]) if leaves else []
            batch = tuple(arrays[0].shape) if arrays else ()
            values = jnp.stack(arrays, axis=-1) if arrays else jnp.empty((0,) if self.preserve_batch else (1, 0), dtype=self.dtype)
            if arrays and not self.preserve_batch:
                values = values.reshape(-1, len(arrays))
        return values, batch

    def _leaf(self, value, kind, batch):
        if tuple(value.shape) != batch:
            value = value.reshape(batch)
        if self.cast and kind in ("int", "bool"):
            if self.backend == "numpy":
                value = np.round(value).astype(np.int64 if kind == "int" else bool)
            elif self.backend == "torch":
                import torch

                value = value.round().to(torch.int64 if kind == "int" else torch.bool)
            else:
                import jax.numpy as jnp

                value = jnp.round(value).astype(jnp.int64 if kind == "int" else jnp.bool_)
        return value.item() if self.backend == "numpy" and not batch else value

    def _output(self, values, batch):
        index = 0

        def leaf(kind):
            nonlocal index
            value = self._leaf(values[..., index], kind, batch)
            index += 1
            return value

        return _tree(self.binding.output, leaf)

    def _exposed(self, spec, value, batch):
        tag = spec[0]
        if tag == "leaf":
            return self._leaf(value, spec[1], batch)
        if tag == "none":
            return None
        if tag == "dict":
            return {key: self._exposed(child, value[key], batch) for key, child in spec[1]}
        values = [self._exposed(child, item, batch) for child, item in zip(spec[1], value)]
        return tuple(values) if tag == "tuple" else values

    def __call__(self, *args, **kwargs):
        values, batch = self._pack(self.binding.bind(args, kwargs))
        if self.backend == "numpy" and math.prod(batch) == 0:
            output = np.empty((0, self.binding.n_out))
        else:
            output = self.evaluate(values)
        return self._output(output, batch)

    def inspect(self, *args, **kwargs):
        values, batch = self._pack(self.binding.bind(args, kwargs))
        if self.backend == "numpy" and math.prod(batch) == 0:
            output = np.empty((0, self.binding.n_out))
            exposed = _tree(self.binding.exposure, lambda kind: np.empty((0,)))
        else:
            output, exposed = self.observe(values, self.detach)
        return Evaluation(self._output(output, batch), self._exposed(self.binding.exposure, exposed, batch))


def numpy_function(net, *, cast=False):
    binding = FunctionBinding(net)

    def observe(x, detach):
        if hasattr(net, "forward_with_intermediates"):
            return net.forward_with_intermediates(x, structured=True)
        return net.forward(x), {}

    return FunctionInterface(binding, net.forward, observe, model=net, cast=cast)


def prediction_call(net, args, kwargs):
    function = numpy_function(net)
    # Retain the old inference convenience without stealing an actual source
    # parameter named "cast". New options belong on as_function instead.
    if "cast" in kwargs and "cast" not in function.__signature__.parameters:
        kwargs = dict(kwargs)
        function = numpy_function(net, cast=kwargs.pop("cast"))
    return function(*args, **kwargs)


def flatten_single(net, args, kwargs):
    leaves = FunctionBinding(net).bind(args, kwargs)
    if any(np.asarray(value).ndim != 0 for value in leaves):
        raise ValueError("flatten_inputs expects one example; use function calls for batched arguments")
    return np.asarray(leaves, dtype=np.float64)


class _TorchObserver:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, snapshot):
        if hasattr(self.model, "forward_with_intermediates"):
            return self.model.forward_with_intermediates(x, detach=snapshot, structured=True)
        return self.model(x), {}


class _TorchFunctionFactory:
    def __init__(self, binding, model):
        self.binding, self.model = binding, model

    def __call__(self, *, cast=False, detach=True):
        return FunctionInterface(
            self.binding, self.model, _TorchObserver(self.model),
            backend="torch", model=self.model, cast=cast, detach=detach,
            preserve_batch=getattr(self.model, "_preserve_function_batch", False),
        )


def attach_torch_interface(model, net):
    model.as_function = _TorchFunctionFactory(FunctionBinding(net), model)
    model.call = model.as_function()
    model.inspect = model.call.inspect
    return model


def attach_jax_interface(apply, net, dtype=None):
    import jax.numpy as jnp

    binding = FunctionBinding(net)

    def as_function(params, /, *, cast=False, detach=True):
        def evaluate(x):
            return apply(params, x)

        def observe(x, snapshot):
            if hasattr(apply, "with_intermediates"):
                return apply.with_intermediates(params, x, detach=snapshot, structured=True)
            return evaluate(x), {}

        return FunctionInterface(
            binding, evaluate, observe, backend="jax", dtype=dtype or jnp.float32,
            cast=cast, detach=detach, preserve_batch=getattr(apply, "_preserve_function_batch", False),
        )

    def call(params, /, *args, **kwargs):
        return as_function(params)(*args, **kwargs)

    def inspect_call(params, /, *args, **kwargs):
        return as_function(params).inspect(*args, **kwargs)

    apply.as_function, apply.call, apply.inspect = as_function, call, inspect_call
    return apply


def invoke_original(fn, values):
    """Call a Python reference while respecting positional-only parameters."""
    positional, keywords = [], {}
    for name, parameter in inspect.signature(fn).parameters.items():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(values[name])
        else:
            keywords[name] = values[name]
    return fn(*positional, **keywords)
