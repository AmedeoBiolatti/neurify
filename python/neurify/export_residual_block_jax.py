"""JAX residual scopes keep the compiled base outside the parameter tree."""

import math

import jax.numpy as jnp


def to_jax(block, dtype=None):
    dtype = dtype or jnp.float32
    base_apply, frozen = block.base.to_jax(dtype)
    params = [{key: jnp.asarray(value, dtype) for key, value in weights.items()} for weights in block.correction_weights()]
    offset, scale = jnp.asarray(block.offset, dtype), jnp.asarray(block.scale, dtype)
    outputs = tuple(name for name, _ in block._wiring)

    def run(p, x, capture):
        if len(p) != len(params):
            raise ValueError("residual parameter layers do not match the block")
        x = jnp.asarray(x, dtype)
        if x.ndim < 1 or x.shape[-1] != block.n_in:
            raise ValueError(f"expected trailing input dimension {block.n_in}")
        if capture:
            base, probes = base_apply.with_activations(frozen, x)
        else:
            base, probes = base_apply(frozen, x), {}
        batch = tuple(x.shape[:-1])
        h = ((x - offset) / scale).reshape(math.prod(batch), block.n_in)
        for index, weights in enumerate(p):
            h = h @ weights["w"].T + weights["b"]
            if index < len(p) - 1:
                h = jnp.maximum(h, 0)
        values = base + h.reshape(batch + (block.n_out,))
        leaves = {name: values[..., index] for index, name in enumerate(outputs)}
        if capture:
            probes.update({name: leaf for name, leaf in leaves.items() if name in probes})
        return jnp.stack(list(leaves.values()), axis=-1), probes

    def apply(p, x):
        return run(p, x, False)[0]

    apply.with_activations = lambda p, x: run(p, x, True)
    apply._preserve_function_batch = True
    return apply, params
