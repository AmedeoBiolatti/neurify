"""Frozen compiled scope plus a zero-initialized correction at its live-outs."""

import numpy as np


class ResidualActivationBlock:
    _is_activation_block = True

    def __init__(self, base, hidden, seed):
        self.base, self.hidden, self.seed = base, tuple(hidden), seed
        bounds = [(lo, hi) for _, _, lo, hi in base.graph.input_specs()]
        self.offset = np.array([(lo + hi) / 2 if np.isfinite([lo, hi]).all() else 0 for lo, hi in bounds])
        self.scale = np.array([(hi - lo) / 2 if np.isfinite([lo, hi]).all() and hi > lo else 1 for lo, hi in bounds])
        widths = (base.n_in,) + self.hidden + (base.n_out,)
        self.correction_params = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
        self.report = dict(base.report, params=self.num_params, depth=self.depth,
                           units=base.report["units"] + sum(self.hidden),
                           residual={"hidden": self.hidden, "params": self.correction_params})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    @property
    def num_params(self):
        return self.base.num_params + self.correction_params

    @property
    def depth(self):
        # Match compiled-network depth: hidden layers/blocks, excluding the
        # affine readout. The two branches execute in parallel.
        return max(self.base.depth, len(self.hidden))

    def correction_weights(self):
        rng = np.random.default_rng(self.seed)
        widths = (self.n_in,) + self.hidden + (self.n_out,)
        return [
            {"w": np.zeros((b, a)) if i == len(widths) - 2 else rng.normal(0, (2 / max(a, 1)) ** 0.5, (b, a)),
             "b": np.zeros(b)}
            for i, (a, b) in enumerate(zip(widths, widths[1:]))
        ]

    def weights(self):
        return {"body": "scope_residual", "base": self.base.weights(), "delta": self.correction_weights()}

    def prepare_for_training(self, **kwargs):
        return type(self)(self.base.prepare_for_training(**kwargs), self.hidden, self.seed)

    def to_torch(self, dtype=None):
        from .export_residual_block_torch import ResidualActivationModule
        from .interface import attach_torch_interface

        return attach_torch_interface(ResidualActivationModule(self, dtype), self)

    def to_jax(self, dtype=None):
        from .export_residual_block_jax import to_jax
        from .interface import attach_jax_interface

        apply, params = to_jax(self, dtype)
        return attach_jax_interface(apply, self, dtype), params
