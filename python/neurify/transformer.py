"""Explicit classical transformer targets and function-specific lowerings.

This target is separate from the existing ``arch='transformer'`` compiler.
An exact shallow lowering must be registered for the source function.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from weakref import WeakKeyDictionary

import numpy as np

from . import _core


@dataclass(frozen=True, kw_only=True)
class Transformer:
    """Classical pre-LN/ReLU target; four uniform FFNs by default.

    ``None`` widths are inferred from the lowering's live register and FFN
    requirements. Shape constraints are hard limits, never a request to fit
    or distill an approximation. No PyTorch import is needed for this config.
    """
    blocks: int = 4
    heads: int = 4
    d_model: int | None = None
    ffn_width: int | None = None
    uniform_ffn: bool = True

    def __post_init__(self):
        for name in ('blocks', 'heads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('d_model', 'ffn_width'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f'{name} must be a positive integer or None')
        if type(self.uniform_ffn) is not bool:
            raise TypeError('uniform_ffn must be a bool')
        if self.d_model is not None and self.d_model % self.heads:
            raise ValueError('d_model must be divisible by heads')

    def resolve(self, *, required_blocks, min_d_model, ffn_widths):
        """Resolve a lowering's proven minimums to concrete uniform shapes."""
        widths = tuple(ffn_widths)
        if type(required_blocks) is not int or required_blocks <= 0:
            raise ValueError('required_blocks must be a positive integer')
        if len(widths) != required_blocks or any(type(w) is not int or w <= 0 for w in widths):
            raise ValueError('ffn_widths must give a positive minimum for every block')
        if type(min_d_model) is not int or min_d_model <= 0:
            raise ValueError('min_d_model must be a positive integer')
        if required_blocks != self.blocks:
            raise _core.CompileError(f'lowering requires {required_blocks} blocks; target requests {self.blocks}')
        alignment = math.lcm(2, self.heads)
        d_model = self.d_model or math.ceil(min_d_model / alignment) * alignment
        if d_model < min_d_model or d_model % alignment:
            raise _core.CompileError(f'target d_model={d_model} cannot fit the {min_d_model}-dimension balanced register bank')
        if self.ffn_width is not None:
            allocated = (self.ffn_width,) * self.blocks
        elif self.uniform_ffn:
            allocated = (math.ceil(max(widths) / 16) * 16,) * self.blocks
        else:
            allocated = tuple(math.ceil(w / 16) * 16 for w in widths)
        if any(have < need for have, need in zip(allocated, widths)):
            raise _core.CompileError(f'target FFN width cannot fit required widths {list(widths)}')
        return {'blocks': self.blocks, 'heads': self.heads, 'd_model': d_model,
                'ffn_widths': allocated, 'uniform_ffn': len(set(allocated)) == 1}


_LOWERINGS = WeakKeyDictionary()


def transformer_lowering(fn, *, exact=False):
    """Register a trusted function-specific builder for ``Transformer``.

    The decorated builder receives ``target``, ``reference``, ``inputs`` and
    ``static`` as keyword arguments and returns a Torch module accepting flat
    batches. ``exact=True`` is the author's domain-wide semantic assertion,
    not an automatic proof. Shape checks and ``neurify.verify`` remain active.
    Builders must reject unsupported input/static overrides themselves.
    """
    if not callable(fn):
        raise TypeError('fn must be callable')
    if type(exact) is not bool:
        raise TypeError('exact must be a bool')

    def register(builder):
        if not callable(builder):
            raise TypeError('lowering must be callable')
        if fn in _LOWERINGS:
            raise ValueError('a transformer lowering is already registered for this function')
        _LOWERINGS[fn] = (builder, exact)
        return builder
    return register


def _validate_model(model, target):
    import torch
    from torch import nn

    if not isinstance(model, nn.Module):
        raise TypeError('transformer lowering must return a torch.nn.Module')
    layers = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    if len(layers) != target.blocks:
        raise _core.CompileError(f'lowering produced {len(layers)} encoder blocks; target requires {target.blocks}')
    widths, streams = [], []
    for layer in layers:
        streams.append(layer.self_attn.embed_dim)
        widths.append(layer.linear1.out_features)
        if layer.self_attn.num_heads != target.heads:
            raise _core.CompileError('lowering attention-head count does not match target')
        if not layer.norm_first or not isinstance(layer.norm1, nn.LayerNorm) or not isinstance(layer.norm2, nn.LayerNorm):
            raise _core.CompileError('classical target requires pre-LayerNorm in every encoder block')
        if layer.activation_relu_or_gelu != 1:
            raise _core.CompileError('classical target currently requires ReLU FFNs')
        if layer.self_attn.dropout or layer.dropout.p or layer.dropout1.p or layer.dropout2.p:
            raise _core.CompileError('exact initial evaluation requires zero dropout')
    if len(set(streams)) != 1 or (target.d_model is not None and streams[0] != target.d_model):
        raise _core.CompileError('lowering residual width does not match target')
    if (target.uniform_ffn and len(set(widths)) != 1) or (target.ffn_width is not None and any(w != target.ffn_width for w in widths)):
        raise _core.CompileError('lowering FFN widths do not match target')
    if any(p.dtype != torch.float64 for p in model.parameters()):
        raise _core.CompileError('lowering must construct float64 weights; export float32 with to_torch')
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise _core.CompileError('lowering must construct finite weights')
    return layers


def _flat_outputs(value, rows, columns):
    import torch
    if not isinstance(value, torch.Tensor):
        raise TypeError('transformer lowering must return a flat output tensor')
    if value.numel() != rows * columns:
        raise _core.CompileError('lowering output shape does not match the source function and input batch')
    return value.reshape(rows, columns)


class TransformerNetwork:
    """Function interface, IR verification and Torch export for a lowering.

    Named-scope probes, JAX export and native-network transforms are not yet
    supported by this experimental backend. The original function's input
    signature and output tree are retained.
    """
    def __init__(self, model, reference, target, exact):
        layers = _validate_model(model, target)
        self._model = model.eval().cpu()
        self._cache = {np.dtype(np.float64): self._model}
        self._reference = reference
        self.target = target
        for name in ('fn', 'graph', 'source_graph', 'out_spec', 'source_out_spec', '_params', 'n_in', 'n_out'):
            setattr(self, name, getattr(reference, name))
        self.arch = 'classical_transformer'
        self.depth = len(layers)
        self.widths = [m.linear1.out_features for m in layers]
        self.d_model = layers[0].self_attn.embed_dim
        self.heads = [m.self_attn.num_heads for m in layers]
        self.num_params = sum(p.numel() for p in model.parameters())
        self.exact = exact and reference.exact
        self.warnings = list(reference.warnings)
        self.report = dict(reference.report, arch=self.arch, depth=self.depth, widths=self.widths,
                           d_model=self.d_model, heads=self.heads, params=self.num_params,
                           exact=self.exact, exactness_source='registered_lowering',
                           uniform_ffn=len(set(self.widths)) == 1, scope_probes=False)

    @property
    def __signature__(self):
        import inspect
        return inspect.signature(self.fn)

    @property
    def input_names(self):
        return [name for name, *_ in self.graph.input_specs()]

    def flatten_inputs(self, *args, **kwargs):
        from .interface import flatten_single
        return flatten_single(self, args, kwargs)

    def forward(self, X, dtype=np.float64):
        import torch
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError('classical forward supports float64 or float32')
        x = np.asarray(X, dtype=dtype)
        single = x.ndim == 1
        if x.ndim not in (1, 2) or x.shape[-1] != self.n_in:
            raise ValueError(f'expected (..., {self.n_in}) flat input columns')
        x = x.reshape(-1, self.n_in)
        for column, (_, kind, lo, hi) in enumerate(self.graph.input_specs()):
            values = x[:, column]
            if np.any(~np.isfinite(values)) or np.any(values < lo) or np.any(values > hi):
                raise ValueError('input outside the declared domain')
            if kind != 'real' and np.any(values != np.rint(values)):
                raise ValueError('integer/Boolean inputs must be integral')
        if dtype not in self._cache:
            self._cache[dtype] = copy.deepcopy(self._model).to(dtype=torch.float32)
        model = self._cache[dtype]
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                batch = x[start:start + 256]
                value = _flat_outputs(model(torch.from_numpy(batch)), len(batch), self.n_out)
                outputs.append(value.detach().cpu().numpy())
        result = np.concatenate(outputs) if outputs else np.empty((0, self.n_out), dtype=dtype)
        return result[0] if single else result

    def __call__(self, *args, **kwargs):
        from .interface import prediction_call
        return prediction_call(self, args, kwargs)

    def as_function(self, *, cast=False):
        from .interface import numpy_function
        return numpy_function(self, cast=cast)

    def inspect(self, *args, **kwargs):
        return self.as_function().inspect(*args, **kwargs)

    def reference(self, X, margin=None):
        return self._reference.reference(X, margin)

    def source_reference(self, X, margin=None):
        return self._reference.source_reference(X, margin)

    def to_torch(self, dtype=None):
        import torch
        from torch import nn
        from .interface import attach_torch_interface

        class FlatOutputs(nn.Module):
            def __init__(self, model, n_out):
                super().__init__()
                self.model, self.n_out = model, n_out

            def forward(self, x):
                return _flat_outputs(self.model(x), len(x), self.n_out)

        model = FlatOutputs(copy.deepcopy(self._model).to(dtype=dtype or torch.float32), self.n_out)
        return attach_torch_interface(model, self)

    def summary(self):
        return (f'neurify.TransformerNetwork(blocks={self.depth}, d_model={self.d_model}, heads={self.heads})\n'
                f'  FFN widths: {self.widths}; uniform={len(set(self.widths)) == 1}\n'
                f'  parameters: {self.num_params}; exact={self.exact} (registered lowering)')


def compile_transformer(fn, target, *, inputs, static, strict, reference_options):
    from .compiler import compile
    if fn not in _LOWERINGS:
        raise _core.CompileError('no classical Transformer lowering is registered for this function; '
                                 "arch='transformer' uses the existing generic compiler")
    builder, exact = _LOWERINGS[fn]
    if strict and not exact:
        raise _core.CompileError('strict=True requires an explicitly exact registered lowering')
    reference = compile(fn, inputs=inputs, static=static, arch='resmlp', strict=strict, **reference_options)
    model = builder(target=target, reference=reference, inputs=inputs, static=static)
    return TransformerNetwork(model, reference, target, exact)
