"""Shared Torch activation tensors at live statement boundaries."""

import math

import torch
from torch import nn

from .torch_capture import capture_call, current_capture


class ActivationModule(nn.Module):
    _preserve_function_batch = True

    def __init__(self, block, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList([s.region.network.to_torch(dtype) for s in block.steps])
        self.n_in = block.n_in
        self._inputs = block._input_names
        self._routing = tuple((s.reads, s.writes, s.region.network.n_in, s.region.network.n_out) for s in block.steps)
        self._capture = block._capture
        self._outputs = tuple(n for n, _ in block._wiring)
        self.activation_names = block.activation_names

    def _run(self, x, capture):
        if x.ndim < 1 or x.shape[-1] != self.n_in:
            raise ValueError(f"expected trailing input dimension {self.n_in}")
        batch, env, probes = tuple(x.shape[:-1]), {}, {}
        rows = math.prod(batch)
        env.update({name: x[..., i] for i, name in enumerate(self._inputs)})
        for index, (layer, (reads, writes, n_in, n_out)) in enumerate(zip(self.layers, self._routing)):
            inputs = torch.stack([env[n] for n in reads], dim=-1) if reads else x.new_empty(batch + (0,))
            values = layer(inputs.reshape(rows, n_in)).reshape(batch + (n_out,))
            # These exact views are used by the next layer and live probes.
            env.update({name: values[..., i] for i, name in enumerate(writes)})
            if capture:
                probes.update({name: env[name] for name in self._capture.get(index, ())})
        return torch.stack([env[n] for n in self._outputs], dim=-1), probes

    def forward(self, x):
        request = current_capture(self)
        output, activations = self._run(x, request is not None)
        if request is not None:
            request.values.update(activations)
        return output

    def forward_with_activations(self, x):
        return capture_call(self, x, detach=False, structured=True)
