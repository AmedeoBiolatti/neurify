"""neurify.merge: several heuristics -> one exact network with shared features."""

import numpy as np
import pytest

import neurify as nf

ILLEGAL = -1e9


def feat(x, y):
    return x * 3 + y


def h_a(x: nf.Int[0, 5], y: nf.Int[0, 5]):
    f = feat(x, y)
    return [f, 10 - f, x * y]


def h_b(x: nf.Int[0, 5], y: nf.Int[0, 5]):
    f = feat(x, y)
    return [f % 4, y, 2]


def ranks(s):
    s = np.asarray(s, float)
    return np.array([(s[i] > np.delete(s, i)).sum() for i in range(len(s))])


GRID = np.array([[x, y] for x in range(6) for y in range(6)], float)


def expected(combine):
    rows = []
    for x, y in GRID.astype(int):
        ra, rb = ranks(h_a(x, y)), ranks(h_b(x, y))
        if combine == "stack":
            rows.append(np.concatenate([ra, rb]))
        elif combine == "priority":
            rows.append(3 * ra + rb)
        elif combine == "vote":
            rows.append(ra + rb)
        else:
            rows.append(combine[0] * ra + combine[1] * rb)
    return np.array(rows, float)


@pytest.mark.parametrize("combine", ["stack", "priority", "vote", [1.0, 0.5]])
def test_merge_combinations_exact(combine):
    net = nf.merge([h_a, h_b], combine=combine)
    assert net.report["groups"] == [3, 3]
    exp = expected(combine)
    for dt in (np.float64, np.float32):
        assert np.array_equal(net.forward(GRID, dtype=dt).astype(float), exp)
    assert nf.verify(net).ok


def test_merge_shares_features():
    def big_a(x: nf.Int[0, 30], y: nf.Int[0, 30]):
        p = x * y
        return [p % 7, p % 5]

    def big_b(x: nf.Int[0, 30], y: nf.Int[0, 30]):
        p = x * y
        return [p % 3, p % 4]

    ua = nf.compile(big_a, output="rank").report["units"]
    ub = nf.compile(big_b, output="rank").report["units"]
    um = nf.merge([big_a, big_b], combine="stack").report["units"]
    assert um < ua + ub  # x * y is built once


def test_merge_validation():
    def other(x: nf.Int[0, 5], z: nf.Int[0, 5]):
        return [x, z]

    def two(x: nf.Int[0, 5], y: nf.Int[0, 5]):
        return [x, y]

    with pytest.raises(TypeError):
        nf.merge([h_a, other])
    with pytest.raises(ValueError):
        nf.merge([h_a, two], combine="priority")  # 3 vs 2 outputs
    with pytest.raises(ValueError):
        nf.merge([h_a, h_b], combine=[1.0])
    with pytest.raises(ValueError):
        nf.merge([h_a, h_b], combine="nope")
    assert nf.merge([h_a, two], combine="stack").report["groups"] == [3, 2]


def bonus_scores(a: nf.Vec(nf.Int[0, 1], 7), b: nf.Vec(nf.Int[0, 1], 7), c: nf.Vec(nf.Int[0, 1], 7)):
    # tiers 1000 / 100 / 50 plus a per-move constant bonus 0..3, and a sentinel
    return [1000 * (b[i] > 0) + 100 * (c[i] > 0) - 50 * (b[i] < c[i]) + (3 - abs(i - 3)) if a[i] == 0 else ILLEGAL
            for i in range(7)]


def test_lex_constant_bonus_tier():
    # the per-move constant must not break the tier proof (it used to fall
    # back to direct comparisons, inexact in float32)
    net = nf.compile(bonus_scores, output="rank")
    assert net.report["margin_sites"] == 0
    assert nf.verify(net, n=3000, dtype=np.float32).ok
