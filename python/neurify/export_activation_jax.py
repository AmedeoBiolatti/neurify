"""Fixed wiring between live statement activations in JAX."""

import math

import jax.numpy as jnp


def to_jax(block, dtype=None):
    layers, params = [], []
    for step in block.steps:
        apply, initial = step.region.network.to_jax(dtype)
        layers.append((step, apply))
        params.append(initial)

    def run(p, x, capture):
        if len(p) != len(layers):
            raise ValueError("activation parameter layers do not match the compiled network")
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != block.n_in:
            raise ValueError(f"expected trailing input dimension {block.n_in}")
        batch = tuple(x.shape[:-1])
        rows = math.prod(batch)
        env = {name: x[..., i] for i, name in enumerate(block._input_names)}
        probes = {}
        for index, ((step, apply), weights) in enumerate(zip(layers, p)):
            values = jnp.stack([env[n] for n in step.reads], axis=-1) if step.reads else jnp.empty(batch + (0,), dtype=x.dtype)
            values = apply(weights, values.reshape(rows, step.region.network.n_in)).reshape(batch + (step.region.network.n_out,))
            env.update({name: values[..., i] for i, name in enumerate(step.writes)})
            if capture:
                probes.update({name: env[name] for name in block._capture.get(index, ())})
        return jnp.stack([env[name] for name, _ in block._wiring], axis=-1), probes

    def apply(p, x):
        return run(p, x, False)[0]

    apply.with_activations = lambda p, x: run(p, x, True)
    apply._preserve_function_batch = True
    return apply, params
