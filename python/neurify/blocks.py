"""Configuration for named source-code blocks, independent of the source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True, kw_only=True)
class Block:
    """Override one named scope at compilation.

    Unspecified settings inherit the compile/scope defaults. ``extra_width``
    adds spare units to each compiled layer; ``extra_depth`` inserts identity
    layers in the block's last live statement. Neither changes initial values.
    ``residual=(32, 32)`` freezes the compiled base and learns a zero-output
    correction with those hidden widths. ``residual=()`` is a linear correction.
    """

    arch: str | None = None
    extra_width: int = 0
    extra_depth: int = 0
    residual: tuple[int, ...] | None = None
    trainable: bool | None = None
    differentiable: bool | None = None
    seed: int = 0

    def __post_init__(self):
        from .compiler import ARCHS

        if self.arch is not None and self.arch not in ARCHS:
            raise ValueError(f"block arch must be one of {ARCHS}")
        for name in ("extra_width", "extra_depth", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("trainable", "differentiable"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool or None")
        if self.residual is not None:
            if not isinstance(self.residual, (tuple, list)) or any(type(n) is not int or n <= 0 for n in self.residual):
                raise ValueError("residual must be a sequence of positive hidden widths (or an empty tuple)")
            object.__setattr__(self, "residual", tuple(self.residual))

    def _build(self, net):
        from .activation import ActivationBlock

        if self.extra_width or self.extra_depth:
            steps = []
            for index, step in enumerate(net.steps):
                body = step.region.network
                if self.extra_width:
                    body = body.widen(self.extra_width, seed=self.seed + index)
                if self.extra_depth and index == len(net.steps) - 1:
                    body = body.deepen(self.extra_depth, width=self.extra_width or 8, seed=self.seed + index)
                steps.append(replace(step, region=replace(step.region, network=body)))
            net = ActivationBlock(
                steps, net._wiring, net._params, net.graph, net.out_spec, net.fn,
                net.warnings, net._capture, net.activation_names, net._output_specs,
            )
        if self.residual is not None:
            from .residual_block import ResidualActivationBlock

            net = ResidualActivationBlock(net, self.residual, self.seed)
        return net


def _block_configs(blocks):
    if blocks is None:
        return {}
    if not isinstance(blocks, Mapping):
        raise TypeError("blocks must map scope names to neurify.Block configurations")
    result = dict(blocks)
    for name, config in result.items():
        if not isinstance(name, str) or not name.strip() or name.startswith("__neurify_"):
            raise ValueError("block names must be non-empty explicit scope names")
        if not isinstance(config, Block):
            raise TypeError(f"block '{name}' must be a neurify.Block configuration")
    return result


def _block_info(step):
    net = step.region.network
    base = getattr(net, "base", net)
    archs = tuple(dict.fromkeys(s.region.network.arch for s in base.steps)) if step.named else (net.arch,)
    residual = getattr(net, "hidden", None)
    trainable_params = net.num_params if residual is None else net.correction_params
    return {
        "name": step.region.name, "arch": "/".join(archs),
        "mode": "compiled" if residual is None else "residual",
        "depth": net.depth, "widths": list(net.widths), "params": net.num_params,
        "trainable_params": trainable_params if step.region.trainable else 0,
        "trainable": step.region.trainable, "differentiable": step.region.differentiable,
        "expose": list(step.exposed), "residual_hidden": residual,
    }
