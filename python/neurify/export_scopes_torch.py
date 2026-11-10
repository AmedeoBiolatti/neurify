"""Torch execution of compiled scopes with explicit variable wiring."""

import torch
import math

from .export_regions_torch import RegionPipeline
from .regions import _policy
from .torch_capture import capture_call


class ScopedModule(RegionPipeline):
    _preserve_function_batch = True

    def __init__(self, scoped, dtype=None):
        super().__init__(scoped, dtype)
        self.scope_names = scoped.scope_names
        self.activation_names = dict(scoped.activation_names)
        self._input_names = scoped._input_names
        self._wiring = scoped._wiring
        self._routing = tuple((s.region.name, s.reads, s.writes, s.exposed, s.named, s.final,
                               s.region.network.n_in, s.region.network.n_out) for s in scoped.steps)

    def _run(self, x, capture, detach, structured=True):
        if x.ndim < 1 or x.shape[-1] != self.n_in:
            raise ValueError(f"expected trailing input dimension {self.n_in}, got {tuple(x.shape)}")
        batch = tuple(x.shape[:-1])
        rows = math.prod(batch)
        env = {name: x[..., i] for i, name in enumerate(self._input_names)}
        observed, values = {}, None
        for stage, (name, reads, writes, exposed, named, final, n_in, n_out) in zip(self.stages, self._routing):
            inputs = torch.stack([env[n] for n in reads], dim=-1) if reads else x.new_empty(batch + (0,))
            if named:
                if capture:
                    values, activations = stage.forward_with_activations(inputs)
                else:
                    values = stage(inputs)
            else:
                values = stage(inputs.reshape(rows, n_in)).reshape(batch + (n_out,))
            if not self._differentiable[name]:
                values = values.detach()
            if not final:
                env.update({n: values[..., i] for i, n in enumerate(writes)})
            if capture and named:
                observed[name] = {
                    n: activations[n].detach().clone() if detach else
                    activations[n].detach() if not self._differentiable[name] else activations[n]
                    for n in exposed
                }
        if self._wiring is not None:
            values = torch.stack([
                env[name] if name is not None else x.new_full(batch, value)
                for name, value in self._wiring
            ], dim=-1)
        if capture and not structured:
            observed = {
                name: torch.stack(list(probes.values()), dim=-1) if probes else values.new_empty(values.shape[:-1] + (0,))
                for name, probes in observed.items()
            }
        return values, observed

    def forward_with_intermediates(self, x, *, detach=True, structured=True):
        _policy(structured, "structured")
        detach = _policy(detach, "detach")
        if not structured and not detach:
            raise ValueError("live activation probes require structured=True; flat stacks are snapshots")
        return capture_call(self, x, detach=detach, structured=structured)
