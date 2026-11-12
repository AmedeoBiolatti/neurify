# Classical transformer target (experimental, unreleased)

```python
import neurify as nf

target = nf.Transformer()
# Equivalent policy:
target = nf.Transformer(blocks=4, heads=4, uniform_ffn=True)
net = nf.compile(my_function, arch=target, strict=True)
```

The default is four stock pre-LayerNorm transformer encoder blocks, softmax
attention, ReLU and uniform FFNs. Residual and FFN widths are inferred from
the lowering's requirements. FFNs round up to multiples of 16. For example,
required widths `[182, 280, 335, 13]` become `[336, 336, 336, 336]`; they
become `[192, 288, 336, 16]` with `uniform_ffn=False`.

Explicit sizes are hard constraints:

```python
target = nf.Transformer(blocks=4, heads=4, d_model=192, ffn_width=384)
```

An unsupported function, incompatible block count, insufficient residual
width or insufficient FFN width raises `CompileError`. Configuration is
immutable and validated. No global setting changes the existing string
architectures; `arch="transformer"` retains the generic native compiler.

## Registering a lowering

The four-block finite-domain construction is not yet a generic optimization
pass for arbitrary Python. A backend author must register a lowering for
the source function explicitly:

```python
@nf.transformer_lowering(my_function, exact=True)
def lower(*, target, reference, inputs, static):
    # Validate supported input/static domains and construct the exact model.
    # requirements and build_exact_model are supplied by the backend author.
    shape = target.resolve(
        required_blocks=4,
        min_d_model=requirements.residual_width,
        ffn_widths=requirements.ffn_widths,
    )
    return build_exact_model(shape, dtype="float64")
```

The builder returns a Torch module taking flat batches. Each encoder must
be a `torch.nn.TransformerEncoderLayer` with the requested dimensions,
pre-LayerNorm, ReLU and zero dropout. Neurify validates the produced shape
and retains the original function's signature, input/output trees and
traced IR for `nf.verify`. Unsupported domain/static overrides must be
rejected by the lowering, not silently ignored. Duplicate registration is
rejected.

`exact=True` is a **trusted assertion by the lowering author** that the
constructed network matches the source over its declared domain. It is
not an automatically generated proof; shape validation cannot establish
semantic equivalence. The report records
`exactness_source="registered_lowering"`. Use native-oracle or exhaustive
checks where possible, plus `nf.verify(net)`. Ordinary floating-point
rounding still applies. `strict=True` rejects registrations that do not
make the exactness assertion.

Generic Neurify contains only the target and backend interface. Domain
specific source and lowerings can remain in their own private project.

## Calls and export

The resulting `TransformerNetwork` supports original-function calls,
broadcast input arrays, `forward` on flat arrays, `nf.verify`, `summary`,
and `to_torch`. Torch exports have the usual `.call` function interface.
Input bounds and integer domains are checked for NumPy calls. Float64 is
used for constructed weights; `to_torch()` exports float32 by default.

Named scope probes and training policies, JAX export, native network
transforms, output re-encoding and block overrides are not yet supported
by this experimental backend. A fused lowering should expose an unscoped
source function rather than claim it retains named-scope controls. Extra
capacity for training must not be confused with exact initial behavior.

For a complete application-independent example, run
`python examples/classical_transformer.py` with the Torch extra installed.
It preserves both Boolean cases exactly and checks that zero-output FFNs
receive a gradient. See [EXPERIMENTS.md](EXPERIMENTS.md) for the consolidation
decisions and the features retained outside the public library.
