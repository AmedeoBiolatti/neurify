import numpy as np
import pytest

import neurify as nf
from programs import int_branches, seq_prog, seq_real


@pytest.mark.parametrize("attention", nf.ATTENTIONS)
@pytest.mark.parametrize("layernorm", [False, True])
@pytest.mark.parametrize("fn", [seq_prog, seq_real], ids=lambda f: f.__name__)
def test_transformer_modes(fn, attention, layernorm):
    net = nf.compile(fn, arch="transformer", attention=attention, layernorm=layernorm, exhaustive_points=0)
    assert net.n_tokens == 7 if fn is seq_prog else 6
    assert sum(net.heads) > 0
    assert net.exact == (attention != "softmax" and not layernorm)
    res = nf.verify(net, n=1000)
    assert res.ok, str(res)
    if net.exact:
        assert res.max_abs_err < 1e-9


def test_seq_is_a_list_elsewhere():
    for arch in ("mlp", "resmlp", "glu"):
        net = nf.compile(seq_prog, arch=arch)
        assert net.exact and nf.verify(net, n=500).ok


def test_transformer_without_sequence_is_exact():
    net = nf.compile(int_branches, arch="transformer", mlp="glu")
    assert net.n_tokens == 1 and net.exact and nf.verify(net).ok


@pytest.mark.parametrize("layernorm", [False, True])
def test_transformer_prepare_for_training(layernorm):
    net = nf.compile(seq_prog, arch="transformer", attention="relu", layernorm=layernorm, exhaustive_points=0)
    prep = net.prepare_for_training(width=32, depth=net.depth + 2, extra_dims=8)
    assert prep.depth == net.depth + 2 and prep.d_model == net.d_model + 8
    assert nf.verify(prep, n=500).ok


def test_transformer_exports_and_fine_tune():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    net = nf.compile(seq_real, arch="transformer", attention="softmax", exhaustive_points=0)
    X = np.random.default_rng(0).uniform(0, 1, (256, 11))
    X[:, 10] = np.random.default_rng(1).integers(0, 5, 256)
    y = net.forward(X)
    m = net.to_torch(torch.float64)
    assert np.allclose(m(torch.tensor(X)).detach().numpy(), y, atol=1e-10)
    apply, params = net.to_jax(jnp.float64)
    assert np.allclose(np.asarray(apply(params, jnp.asarray(X))), y, atol=1e-10)
    # fine-tune: gradients reach attention, embeddings and MLP
    m32 = net.to_torch()
    Xt = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32) + 0.2 * Xt[:, :1]
    opt = torch.optim.Adam(m32.parameters(), lr=1e-3)
    loss0 = torch.nn.functional.mse_loss(m32(Xt), target).item()
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m32(Xt), target)
        loss.backward()
        opt.step()
    assert np.isfinite(loss.item()) and loss.item() < loss0
