from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

import neurify as nf


def test_default_target_is_four_uniform_blocks_and_infers_minimum_width():
    target = nf.Transformer()
    assert target.blocks == 4 and target.heads == 4 and target.uniform_ffn
    shape = target.resolve(required_blocks=4, min_d_model=176, ffn_widths=[182, 280, 335, 13])
    assert shape['d_model'] == 176
    assert shape['ffn_widths'] == (336,) * 4
    assert nf.Transformer(uniform_ffn=False).resolve(
        required_blocks=4, min_d_model=176, ffn_widths=[182, 280, 335, 13])['ffn_widths'] == (192, 288, 336, 16)
    with pytest.raises(FrozenInstanceError):
        target.blocks = 3


@pytest.mark.parametrize('options', [{'blocks': 0}, {'blocks': True}, {'heads': -1},
    {'d_model': 175}, {'ffn_width': 0}, {'uniform_ffn': 'yes'}])
def test_invalid_target_settings(options):
    with pytest.raises((TypeError, ValueError)):
        nf.Transformer(**options)


@pytest.mark.parametrize('target,requirements', [
    (nf.Transformer(blocks=3), dict(required_blocks=4, min_d_model=176, ffn_widths=[1, 1, 1, 1])),
    (nf.Transformer(d_model=128), dict(required_blocks=4, min_d_model=176, ffn_widths=[1, 1, 1, 1])),
    (nf.Transformer(ffn_width=16), dict(required_blocks=4, min_d_model=176, ffn_widths=[1, 17, 1, 1])),
])
def test_incompatible_constraints_fail_explicitly(target, requirements):
    with pytest.raises(nf.CompileError):
        target.resolve(**requirements)


def test_unregistered_function_and_unsupported_controls_do_not_fall_back():
    def f(x: nf.Bool):
        return x
    with pytest.raises(nf.CompileError, match='no classical Transformer lowering'):
        nf.compile(f, arch=nf.Transformer())
    with pytest.raises(ValueError, match='raw output'):
        nf.compile(f, arch=nf.Transformer(), output='rank')
    assert nf.compile(f, arch='transformer', strict=True).exact


def test_scope_policies_and_legacy_schedule_options_are_not_silently_ignored():
    def scoped(x: nf.Bool):
        with nf.scope('energy', trainable=False):
            y = x
        return y
    with pytest.raises(ValueError, match='scope controls'):
        nf.compile(scoped, arch=nf.Transformer())
    def plain(x: nf.Bool):
        return x
    for kwargs in ({'method': 'size'}, {'attention': 'hard'}, {'repr': 'onehot'}, {'trainable': True},
                   {'layernorm': True}, {'attn_tol': 0.1}, {'ln_tol': 0.1}, {'onehot_max': 128}):
        with pytest.raises(ValueError):
            nf.compile(plain, arch=nf.Transformer(), **kwargs)


def register_identity(*, exact=True, wrong_shape=False, attention_dropout=0., wrong_output=False,
                      nonfinite=False):
    torch = pytest.importorskip('torch')
    from torch import nn

    def identity(x: nf.Bool):
        return x

    @nf.transformer_lowering(identity, exact=exact)
    def build(*, target, reference, inputs, static):
        shape = target.resolve(required_blocks=4, min_d_model=4, ffn_widths=[1] * 4)

        class Identity(nn.Module):
            def __init__(self):
                super().__init__()
                d = shape['d_model']
                self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, target.heads,
                    dim_feedforward=shape['ffn_widths'][i], dropout=0., activation='relu',
                    batch_first=True, norm_first=True, dtype=torch.float64)
                    for i in range(3 if wrong_shape else 4)])
                with torch.no_grad():
                    for layer in self.layers:
                        for p in layer.parameters():
                            p.zero_()
                        for norm in (layer.norm1, layer.norm2):
                            norm.weight.fill_(math.sqrt(1 + norm.eps))
                    self.layers[0].self_attn.dropout = attention_dropout
                    if nonfinite:
                        self.layers[0].linear1.weight[0, 0] = float('nan')

            def forward(self, x):
                s = 2 * x[:, :1] - 1
                state = torch.cat([s, -s] * (shape['d_model'] // 2), dim=1).unsqueeze(1)
                for layer in self.layers:
                    state = layer(state)
                output = (state[:, 0, 0] + 1) / 2
                return output.repeat(2) if wrong_output else output

        return Identity()
    return identity


def test_registered_target_preserves_function_interface_verification_and_torch_export():
    import inspect
    torch = pytest.importorskip('torch')
    fn = register_identity()
    net = nf.compile(fn, arch=nf.Transformer(), strict=True)
    assert net.depth == 4 and net.widths == [16] * 4 and net.heads == [4] * 4
    assert net.exact and net.report['exactness_source'] == 'registered_lowering'
    assert inspect.signature(net) == inspect.signature(fn)
    assert net.input_names == ['x']
    np.testing.assert_array_equal(net.flatten_inputs(x=True), [1])
    assert nf.verify(net).ok
    np.testing.assert_array_equal(net(x=np.array([0, 1, 0])), [0, 1, 0])
    np.testing.assert_array_equal(net.forward([[0], [1]], dtype=np.float32), [[0], [1]])
    assert net.forward(np.empty((0, 1))).shape == (0, 1)
    assert net.inspect(True).exposed == {}
    with pytest.raises(ValueError, match='integral'):
        net(.5)
    with pytest.raises(ValueError, match='domain'):
        net(2)
    model = net.to_torch(dtype=torch.float64)
    assert sum(p.numel() for p in model.parameters()) == net.num_params
    assert model.call(x=True).item() == 1
    assert model(torch.tensor([[0.], [1.]], dtype=torch.float64)).shape == (2, 1)
    assert net.summary().startswith('neurify.TransformerNetwork')


def test_registration_claims_and_actual_shape_are_checked():
    fn = register_identity(exact=False)
    with pytest.raises(nf.CompileError, match='explicitly exact'):
        nf.compile(fn, arch=nf.Transformer(), strict=True)
    assert not nf.compile(fn, arch=nf.Transformer()).exact
    wrong = register_identity(wrong_shape=True)
    with pytest.raises(nf.CompileError, match='3 encoder blocks'):
        nf.compile(wrong, arch=nf.Transformer(), strict=True)
    with pytest.raises(ValueError, match='already registered'):
        nf.transformer_lowering(fn)(lambda **kw: None)


def test_registered_target_rejects_stochastic_or_nonfinite_initialization():
    for options, message in ((dict(attention_dropout=.1), 'zero dropout'),
                             (dict(nonfinite=True), 'finite weights')):
        with pytest.raises(nf.CompileError, match=message):
            nf.compile(register_identity(**options), arch=nf.Transformer(), strict=True)


def test_lowering_cannot_silently_change_batch_size_on_numpy_or_torch_export():
    torch = pytest.importorskip('torch')
    net = nf.compile(register_identity(wrong_output=True), arch=nf.Transformer(), strict=True)
    with pytest.raises(nf.CompileError, match='output shape'):
        net.forward([[0], [1]])
    with pytest.raises(nf.CompileError, match='output shape'):
        net.to_torch()(torch.tensor([[0.], [1.]]))
