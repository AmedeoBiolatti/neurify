"""JAX execution of scopes; untouched locals bypass scope output barriers."""

import jax
import jax.numpy as jnp
import math

from .regions import _policy


def to_jax(scoped, dtype=None):
    stages, params = [], {}
    for step in scoped.steps:
        region = step.region
        apply, initial = region.network.to_jax(dtype)
        stages.append((step, apply, None if region.trainable else initial))
        if region.trainable:
            params[region.name] = initial
    keys = tuple(params)

    def run(p, x, capture, detach, structured):
        if set(p) != set(keys):
            raise ValueError(f"expected parameter regions {keys}, got {tuple(p)}")
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != scoped.n_in:
            raise ValueError(f"expected trailing input dimension {scoped.n_in}, got {x.shape}")
        batch = tuple(x.shape[:-1])
        rows = math.prod(batch)
        env = {name: x[..., i] for i, name in enumerate(scoped._input_names)}
        observed, values = {}, None
        for step, apply, frozen in stages:
            inputs = jnp.stack([env[n] for n in step.reads], axis=-1) if step.reads else jnp.empty(batch + (0,), dtype=x.dtype)
            weights = p[step.region.name] if step.region.trainable else frozen
            if step.named:
                if capture:
                    values, activations = apply.with_activations(weights, inputs)
                else:
                    values = apply(weights, inputs)
            else:
                values = apply(weights, inputs.reshape(rows, step.region.network.n_in)).reshape(batch + (step.region.network.n_out,))
            if not step.region.differentiable:
                values = jax.lax.stop_gradient(values)
            if not step.final:
                env.update({n: values[..., i] for i, n in enumerate(step.writes)})
            if capture and step.named:
                selected = {
                    n: jax.lax.stop_gradient(activations[n]) if detach or not step.region.differentiable else activations[n]
                    for n in step.exposed
                }
                observed[step.region.name] = selected if structured else (
                    jnp.stack(list(selected.values()), axis=-1) if selected else jnp.empty(batch + (0,), dtype=x.dtype)
                )
        if scoped._wiring is not None:
            values = jnp.stack([
                env[name] if name is not None else jnp.full(batch, value, dtype=x.dtype)
                for name, value in scoped._wiring
            ], axis=-1)
        return values, observed

    def apply(p, x):
        return run(p, x, False, True, True)[0]

    def with_intermediates(p, x, *, detach=True, structured=True):
        detach = _policy(detach, "detach")
        structured = _policy(structured, "structured")
        if not structured and not detach:
            raise ValueError("live activation probes require structured=True; flat stacks are snapshots")
        return run(p, x, True, detach, structured)

    apply.with_intermediates = with_intermediates
    apply.region_names = scoped.region_names
    apply.scope_names = scoped.scope_names
    apply.trainable_regions = keys
    apply._preserve_function_batch = True
    return apply, params
