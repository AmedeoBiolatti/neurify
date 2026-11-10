"""Residual scope execution with shared corrected output activations."""

import math

import torch
from torch import nn

from .torch_capture import capture_call, current_capture


class ResidualActivationModule(nn.Module):
    _preserve_function_batch = True

    def __init__(self, block, dtype=None):
        super().__init__()
        dtype = dtype or torch.float32
        self.base = block.base.to_torch(dtype)
        self.n_in = block.n_in
        self._outputs = tuple(name for name, _ in block._wiring)
        self.activation_names = block.activation_names
        self.register_buffer("offset", torch.as_tensor(block.offset, dtype=dtype))
        self.register_buffer("scale", torch.as_tensor(block.scale, dtype=dtype))
        layers = []
        for index, weights in enumerate(block.correction_weights()):
            out_dim, in_dim = weights["w"].shape
            layer = nn.Linear(in_dim, out_dim, dtype=dtype)
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weights["w"], dtype=dtype))
                layer.bias.copy_(torch.as_tensor(weights["b"], dtype=dtype))
            layers.append(layer)
            if index < len(block.hidden):
                layers.append(nn.ReLU())
        self.delta = nn.Sequential(*layers)
        self._set_trainable(True)

    def _set_trainable(self, enabled):
        self.base.requires_grad_(False)
        self.delta.requires_grad_(enabled)

    def _run(self, x, capture):
        if x.ndim < 1 or x.shape[-1] != self.n_in:
            raise ValueError(f"expected trailing input dimension {self.n_in}")
        if capture:
            base, probes = self.base.forward_with_activations(x)
        else:
            base, probes = self.base(x), {}
        batch = tuple(x.shape[:-1])
        rows = math.prod(batch)
        features = ((x - self.offset) / self.scale).reshape(rows, self.n_in)
        values = base + self.delta(features).reshape(batch + (len(self._outputs),))
        # The corrected leaves themselves feed both downstream code and probes.
        leaves = {name: values[..., index] for index, name in enumerate(self._outputs)}
        if capture:
            probes.update({name: leaf for name, leaf in leaves.items() if name in probes})
        return torch.stack(list(leaves.values()), dim=-1), probes

    def forward(self, x):
        request = current_capture(self)
        output, probes = self._run(x, request is not None)
        if request is not None:
            request.values.update(probes)
        return output

    def forward_with_activations(self, x):
        return capture_call(self, x, detach=False, structured=True)
