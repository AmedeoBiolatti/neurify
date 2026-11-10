"""slope knob, precision-aware gadgets, trainability report, exact+delta export."""

import numpy as np
import pytest

import neurify as nf
from programs import bools, bubble, gcd, int_branches, loops


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("fn", [gcd, loops, int_branches, bubble])
@pytest.mark.parametrize("slope", [0.25, 0.5])
def test_slope_exact(fn, slope):
    net = nf.compile(fn, slope=slope)
    assert nf.verify(net).ok


def test_slope_validation():
    with pytest.raises(nf.CompileError):
        nf.compile(bools, slope=1.5)


def test_slope_adds_gradient_flow(torch):
    torch.set_num_threads(4)
    flat = nf.trainability_report(nf.compile(gcd), n=128, lrs=(1e-6,))
    soft = nf.trainability_report(nf.compile(gcd, slope=0.5), n=128, lrs=(1e-6,))
    assert soft.grad_coverage > flat.grad_coverage


@pytest.mark.parametrize("arch", ["mlp", "resmlp"])
def test_trainable_onehot_pairs_exact(arch):
    net = nf.compile(gcd, arch=arch, method="trainable")
    assert nf.verify(net).ok
    assert nf.verify(net, dtype=np.float32).ok


def test_trainable_tolerates_more_noise(torch):
    torch.set_num_threads(4)
    depth = nf.trainability_report(nf.compile(gcd), n=128, lrs=(1e-6,))
    train = nf.trainability_report(nf.compile(gcd, method="trainable"), n=128, lrs=(1e-6,))
    assert train.lr_pred > 100 * depth.lr_pred
    assert train.adam_curve[1e-6] > depth.adam_curve[1e-6]


def test_report_fields_and_str(torch):
    torch.set_num_threads(4)
    r = nf.trainability_report(nf.compile(bools), n=64, lrs=(1e-4, 1e-3))
    assert r.adam_tolerance == 1e-3  # small ranges: standard learning rates are fine
    assert 0 <= r.bottleneck < len(r.layers)
    assert "bottleneck" in str(r) and "He ratio" in str(r)


def test_exact_plus_delta_torch(torch):
    net = nf.compile(gcd)
    m = net.to_torch_residual(hidden=(32,))
    X = np.array([[12, 18], [7, 21], [24, 1], [5, 5]], float)
    y = m(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    assert np.allclose(y, net.reference(X)[0])
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == sum(p.numel() for p in m.delta.parameters())
    # one big step changes the output, but only through delta
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss = m(torch.tensor(X, dtype=torch.float32)).sum()
    loss.backward()
    opt.step()
    assert all(p.grad is None for p in m.exact.parameters())
    assert m.features.dim == 2  # 24 values > onehot_max: normalised scalars
    assert net.to_torch_residual(onehot_max=32).features.dim == 24 + 24


def test_exact_plus_delta_jax():
    jnp = pytest.importorskip("jax.numpy")
    net = nf.compile(gcd)
    apply, params = net.to_jax_residual(hidden=(16,))
    X = np.array([[12, 18], [7, 21]], float)
    assert np.allclose(np.asarray(apply(params, jnp.asarray(X))), net.reference(X)[0], atol=1e-5)


# ------------------------------------------------------------ one-hot values

from programs import EXACT_ALL_ARCHS  # noqa: E402


@pytest.mark.parametrize("fn", EXACT_ALL_ARCHS)
@pytest.mark.parametrize("arch", ["mlp", "resmlp"])
def test_onehot_exact(fn, arch):
    net = nf.compile(fn, arch=arch, repr="onehot")
    assert nf.verify(net).ok
    if all(s[1] != "real" for s in net.graph.input_specs()):  # real inputs round in float32
        assert nf.verify(net, dtype=np.float32).ok
    assert nf.verify(net.prepare_for_training(width=16)).ok


def test_onehot_input_features_exports(torch):
    jnp = pytest.importorskip("jax.numpy")
    net = nf.compile(gcd, repr="onehot")
    w = net.weights()
    assert (~np.isnan(w["feat_value"])).sum() == 48  # both inputs one-hot encoded
    X = np.array([[12, 18], [7, 21], [24, 1]], float)
    ref = net.reference(X)[0]
    assert np.allclose(net.to_torch()(torch.tensor(X, dtype=torch.float32)).detach().numpy(), ref)
    apply, params = net.to_jax()
    assert np.allclose(np.asarray(apply(params, jnp.asarray(X))), ref)


def test_onehot_tolerates_standard_learning_rates(torch):
    torch.set_num_threads(4)
    scalar = nf.trainability_report(nf.compile(gcd), n=128, lrs=(1e-4,))
    onehot = nf.trainability_report(nf.compile(gcd, repr="onehot"), n=128, lrs=(1e-4,))
    assert scalar.adam_curve[1e-4] < 0.5
    assert onehot.adam_curve[1e-4] >= 0.99


def test_repr_validation():
    with pytest.raises(nf.CompileError):
        nf.compile(bools, repr="binary")
