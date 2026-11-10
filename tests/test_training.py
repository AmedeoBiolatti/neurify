import numpy as np
import pytest

import neurify as nf
from programs import int_branches, piecewise_real, poly


@pytest.mark.parametrize("arch", nf.ARCHS)
def test_prepare_for_training_preserves_function(arch):
    net = nf.compile(int_branches, arch=arch, method="trainable")
    prep = net.prepare_for_training(width=64, depth=net.depth + 3, extra_dims=8, seed=1)
    assert prep.depth == net.depth + 3
    assert min(prep.widths) >= 64 or arch != "mlp"
    assert prep.num_params > net.num_params
    assert nf.verify(prep).ok


def test_torch_export_matches_and_trains():
    torch = pytest.importorskip("torch")
    net = nf.compile(piecewise_real, arch="resmlp", trainable={"width": 32, "extra_dims": 4})
    model = net.to_torch(dtype=torch.float64)
    X = np.random.default_rng(0).uniform(-3, 3, (512, 2))
    with torch.no_grad():
        y = model(torch.tensor(X)).numpy()
    assert np.allclose(y, net.forward(X), atol=1e-9)

    # fine-tune towards a drifted target: the exact init starts at the program
    Xt = torch.tensor(X, dtype=torch.float32)
    ref = torch.tensor(net.forward(X), dtype=torch.float32)
    yt = ref + 0.3 * torch.sin(Xt[:, :1])  # task drift
    m32 = net.to_torch()
    opt = torch.optim.Adam(m32.parameters(), lr=1e-3)
    loss0 = torch.nn.functional.mse_loss(m32(Xt), yt).item()
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m32(Xt), yt)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * loss0


def test_jax_export_matches_and_differentiates():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    for arch in ("mlp", "glu"):
        net = nf.compile(poly if arch == "glu" else piecewise_real, arch=arch)
        apply, params = net.to_jax(dtype=jnp.float64)
        X = np.random.default_rng(1).uniform(-1, 1, (256, 2))
        y = np.asarray(jax.jit(apply)(params, jnp.asarray(X)))
        assert np.allclose(y, net.forward(X), atol=1e-9)
        g = jax.grad(lambda p: jnp.mean(apply(p, jnp.asarray(X)) ** 2))(params)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a: bool(jnp.all(jnp.isfinite(a))), g))
