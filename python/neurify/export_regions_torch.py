"""PyTorch export of named, separately compiled regions."""

from types import MappingProxyType
import math

import torch
from torch import nn

from .regions import _policy, _from_leaves
from .torch_capture import capture_call, current_capture


class RegionPipeline(nn.Module):
    """Region controls and observation of the actual trainable computation.

    Freezing parameters does not detach inputs. Stopping output derivatives
    does not disable gradients for downstream trainable regions.
    """

    _preserve_function_batch = True

    def __init__(self, pipeline, dtype=None):
        super().__init__()
        self.region_names = pipeline.region_names
        self.n_in, self.n_out = pipeline.n_in, pipeline.n_out
        self._out_specs = {r.name: r.network.out_spec for r in pipeline._regions}
        self._dimensions = tuple((r.network.n_in, r.network.n_out) for r in pipeline._regions)
        self._block_info = {info["name"]: dict(info) for info in pipeline.report.get("blocks", [])}
        self.stages = nn.ModuleList([r.network.to_torch(dtype) for r in pipeline._regions])
        self._trainable, self._differentiable = {}, {}
        for r in pipeline._regions:
            self.set_trainable(r.name, r.trainable)
            self.set_differentiable(r.name, r.differentiable)

    @property
    def regions(self):
        return MappingProxyType(dict(zip(self.region_names, self.stages)))

    @property
    def blocks(self):
        """Named source blocks (or named regions for a helper pipeline)."""
        names = getattr(self, "scope_names", self.region_names)
        return MappingProxyType({name: self.regions[name] for name in names})

    def _index(self, name):
        if name not in self.region_names:
            raise KeyError(f"unknown region '{name}'; expected one of {self.region_names}")
        return self.region_names.index(name)

    def set_trainable(self, name, enabled=True):
        """Freeze/unfreeze parameters. Freezing clears already accumulated grads."""
        index = self._index(name)
        enabled = _policy(enabled, "trainable")
        stage = self.stages[index]
        if hasattr(stage, "_set_trainable"):
            stage._set_trainable(enabled)
        else:
            stage.requires_grad_(enabled)
        for parameter in stage.parameters():
            if not enabled:
                parameter.grad = None
        self._trainable[name] = enabled
        return self

    def train_only(self, *names):
        """Train just the named blocks, freezing all others atomically.

        Calling with no names freezes every block. Gradient-flow policies
        are preserved, so frozen downstream blocks still pass derivatives.
        """
        for name in names:
            if not isinstance(name, str):
                raise TypeError("train_only expects block names as separate strings")
            self._index(name)
        selected = set(names)
        for name in self.region_names:
            self.set_trainable(name, name in selected)
        return self

    def summary(self):
        """Report current structure, trainable counts, and gradient controls."""
        lines = [f"neurify Torch model(inputs={self.n_in}, outputs={self.n_out})"]
        for name, stage in self.regions.items():
            info = self._block_info.get(name, {})
            total = sum(p.numel() for p in stage.parameters())
            trainable = sum(p.numel() for p in stage.parameters() if p.requires_grad)
            structure = (f"arch={info['arch']}, mode={info['mode']}, depth={info['depth']}, "
                         f"widths={info['widths']}, " if info else "")
            if info.get("residual_hidden") is not None:
                structure += f"correction_hidden={info['residual_hidden']}, "
            lines.append(f"  {name}: {structure}params={total}, trainable_params={trainable}, "
                         f"trainable={self._trainable[name]}, differentiable={self._differentiable[name]}")
        return "\n".join(lines)

    def set_differentiable(self, name, enabled=True):
        """Enable/stop gradient flow at the named region's output."""
        self._index(name)
        self._differentiable[name] = _policy(enabled, "differentiable")
        return self

    def get_extra_state(self):
        return {
            "version": 1,
            "trainable": dict(self._trainable),
            "differentiable": dict(self._differentiable),
        }

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("unsupported region control state version")
        for key in ("trainable", "differentiable"):
            if set(state[key]) != set(self.region_names):
                raise ValueError("checkpoint region names do not match this pipeline")
            for flag in state[key].values():
                _policy(flag, key)
        for name in self.region_names:
            self.set_trainable(name, state["trainable"][name])
            self.set_differentiable(name, state["differentiable"][name])

    def _run(self, x, capture, detach, structured=False):
        if x.ndim < 1 or x.shape[-1] != self.n_in:
            raise ValueError(f"expected trailing input dimension {self.n_in}, got {tuple(x.shape)}")
        batch, values = tuple(x.shape[:-1]), x
        rows = math.prod(batch)
        observations = {}
        for name, stage, (n_in, n_out) in zip(self.region_names, self.stages, self._dimensions):
            raw = stage(values.reshape(rows, n_in)).reshape(batch + (n_out,))
            if not self._differentiable[name]:
                raw = raw.detach()
            parts = tuple(raw[..., i] for i in range(n_out))
            # Always route through these leaves, including prediction-only
            # calls, so live structured probes are ancestors of the loss.
            values = torch.stack(parts, dim=-1)
            if capture:
                if structured:
                    leaves = (v.detach().clone() for v in parts) if detach else iter(parts)
                    observations[name] = _from_leaves(self._out_specs[name], iter(leaves))
                else:
                    observations[name] = raw.detach().clone() if detach else raw
        return values, observations

    def forward(self, x):
        request = current_capture(self)
        output, observations = self._run(
            x, request is not None, request.detach if request is not None else True,
            request.structured if request is not None else False,
        )
        if request is not None:
            request.values.update(observations)
        return output

    def forward_with_intermediates(self, x, *, detach=True, structured=False):
        """Return output and named region outputs from current weights.

        By default observations are detached, cloned snapshots. Use
        ``detach=False`` for auxiliary losses or activation gradients. A
        region with differentiable=False remains detached in either case.
        """
        _policy(structured, "structured")
        return capture_call(self, x, detach=_policy(detach, "detach"), structured=structured)
