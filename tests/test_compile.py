import math

import numpy as np
import pytest

import neurify as nf
from programs import EXACT_ALL_ARCHS, poly, real_cmp


@pytest.mark.parametrize("arch", nf.ARCHS)
@pytest.mark.parametrize("method", nf.METHODS)
@pytest.mark.parametrize("fn", EXACT_ALL_ARCHS, ids=lambda f: f.__name__)
def test_exact(fn, arch, method):
    net = nf.compile(fn, arch=arch, method=method)
    assert net.exact
    res = nf.verify(net)  # also checks the traced IR against real Python execution
    assert res.ok, str(res)


def test_call_matches_python():
    from programs import bubble, dict_io

    net = nf.compile(dict_io)
    for w in (1, 2, 3):
        for flip in (False, True):
            for x in range(10):
                cfg = {"w": w, "flip": flip}
                assert net(cfg, x, cast=True) == dict_io(cfg, x)
    net = nf.compile(bubble, arch="resmlp")
    assert net([3, 1, 2, 0], 2, cast=True) == bubble([3, 1, 2, 0], 2)


def test_glu_products_exact_and_relu_refuses():
    net = nf.compile(poly, arch="glu")
    assert net.exact and nf.verify(net).ok
    with pytest.raises(nf.CompileError, match="not piecewise linear"):
        nf.compile(poly, arch="mlp")


def test_real_comparison_margin_reported_or_refused():
    net = nf.compile(real_cmp)
    assert not net.exact and net.report["margin_sites"] > 0
    assert nf.verify(net).ok  # exact outside the margin bands
    with pytest.raises(nf.CompileError, match="NotExact"):
        nf.compile(real_cmp, strict=True)


def test_real_comparison_exact_on_finite_domain():
    def f(n: nf.Int[0, 20]):
        x = n / 3
        return x * 2 if x > 2.5 else -x

    net = nf.compile(f, strict=True)
    assert net.exact and nf.verify(net).ok


def test_static_args_and_annotations():
    def f(x: nf.Int[0, 9], k):
        return x * k if k > 1 else -x

    net = nf.compile(f, static={"k": 3})
    assert net(4) == pytest.approx(12)


def test_missing_domain():
    def f(x):
        return x

    with pytest.raises(TypeError, match="no domain"):
        nf.compile(f)


def test_dynamic_dict_keys_rejected():
    def f(x: nf.Int[0, 3]):
        d = {}
        if x > 1:
            d["a"] = x
        return d.get("a", 0)

    with pytest.raises(nf.TraceError, match="dict keys depend"):
        nf.compile(f)


def test_dynamic_list_length_rejected():
    def f(x: nf.Int[0, 3]):
        ys = []
        if x > 1:
            ys.append(x)
        return len(ys)

    with pytest.raises(nf.TraceError, match="list length"):
        nf.compile(f)


def test_unbounded_loop_rejected():
    def f(x: nf.Int[0, 9]):
        while x != 0:
            x -= 2
        return x

    with pytest.raises(nf.TraceError, match="max_loop_iters"):
        nf.compile(f, max_loop_iters=50)


def test_deep_loop_exact_in_float32():
    def collatz(n: nf.Int[1, 27]):
        steps = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        return steps

    net = nf.compile(collatz, arch="resmlp", max_loop_iters=120)
    assert nf.verify(net).ok
    torch = pytest.importorskip("torch")
    x = torch.arange(1, 28, dtype=torch.float32)[:, None]
    y = net.to_torch()(x).detach().numpy().ravel()
    assert np.array_equal(y, [collatz(i) for i in range(1, 28)])


def test_reports():
    from programs import int_branches

    net = nf.compile(int_branches, method="size")
    s = net.summary()
    assert "exact  : yes" in s and net.num_params > 0 and net.depth == len(net.widths)


def test_float32_exact_with_large_constants():
    # a data-dependent select between values of very different magnitude
    def f(x: nf.Int[0, 9], y: nf.Int[0, 9]):
        s = x * 3 - y * 0.5 + 1e6 * (x == y)
        return s if x + y < 12 else -1e9

    for arch in ("mlp", "resmlp", "glu"):
        net = nf.compile(f, arch=arch)
        r = nf.verify(net, dtype=np.float32, atol=0, rtol=0)
        assert r.ok, str(r)
