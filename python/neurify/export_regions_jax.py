"""JAX export with frozen weights outside the optimizer's parameter tree."""

import jax
import jax.numpy as jnp
import math

from .regions import _policy, _from_leaves


def to_jax(pipeline, dtype=None):
    stages, params = [], {}
    for region in pipeline._regions:
        apply, initial = region.network.to_jax(dtype)
        stages.append((region, apply, None if region.trainable else initial))
        if region.trainable:
            params[region.name] = initial
    names = tuple(params)

    def run(p, x, capture, detach, structured=False):
        if set(p) != set(names):
            raise ValueError(f"expected parameter regions {names}, got {tuple(p)}")
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != pipeline.n_in:
            raise ValueError(f"expected trailing input dimension {pipeline.n_in}, got {x.shape}")
        batch, values = tuple(x.shape[:-1]), x
        rows = math.prod(batch)
        observations = {}
        for region, stage_apply, frozen in stages:
            weights = p[region.name] if region.trainable else frozen
            values = stage_apply(weights, values.reshape(rows, region.network.n_in)).reshape(batch + (region.network.n_out,))
            if not region.differentiable:
                values = jax.lax.stop_gradient(values)
            parts = tuple(values[..., i] for i in range(region.network.n_out))
            raw, values = values, jnp.stack(parts, axis=-1)
            if capture:
                if structured:
                    leaves = (jax.lax.stop_gradient(v) for v in parts) if detach else iter(parts)
                    observations[region.name] = _from_leaves(region.network.out_spec, iter(leaves))
                else:
                    observations[region.name] = jax.lax.stop_gradient(raw) if detach else raw
        return values, observations

    def apply(p, x):
        return run(p, x, False, True)[0]

    def with_intermediates(p, x, *, detach=True, structured=False):
        _policy(structured, "structured")
        return run(p, x, True, _policy(detach, "detach"), structured)

    apply.with_intermediates = with_intermediates
    apply.region_names = pipeline.region_names
    apply.trainable_regions = names
    apply._preserve_function_batch = True
    return apply, params
