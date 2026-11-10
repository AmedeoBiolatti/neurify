"""Output encodings (rank / onehot) and lexicographic score comparisons."""

import numpy as np
import pytest

import neurify as nf

ILLEGAL = -1e9


def scores(a: nf.Vec(nf.Int[0, 3], 4), b: nf.Vec(nf.Int[0, 5], 4), legal: nf.Vec(nf.Bool(), 4)):
    # lexicographic weights with a sentinel, like game heuristics
    return [1e6 * a[i] - 1e2 * b[i] + (a[i] * b[i]) % 7 if legal[i] else ILLEGAL for i in range(4)]


def ranks(s):
    s = np.asarray(s)
    return [int((s[i] > np.delete(s, i)).sum()) for i in range(len(s))]


def small(x: nf.Int[0, 5], y: nf.Int[0, 5], z: nf.Real[-1, 1]):
    return [x * y, x - y, 3, z]


@pytest.mark.parametrize("output", ["rank", "onehot"])
@pytest.mark.parametrize("arch", ["mlp", "resmlp"])
def test_encodings_exact(output, arch):
    net = nf.compile(scores, arch=arch, output=output)
    assert net.exact
    assert nf.verify(net, n=3000).ok
    assert nf.verify(net, n=3000, dtype=np.float32).ok


def test_rank_semantics_and_ties():
    net = nf.compile(scores, output="rank")
    a, b, legal = [1, 1, 0, 3], [2, 2, 0, 5], [True, True, True, False]
    want = ranks(scores(a, b, legal))
    assert net(a, b, legal, cast=True) == want == [2, 2, 1, 0]
    oh = nf.compile(scores, output="onehot")
    assert oh(a, b, legal, cast=True) == [True, True, False, False]


def _max_abs(d):
    if isinstance(d, dict):
        return max([_max_abs(v) for v in d.values()] + [0.0])
    if isinstance(d, (list, tuple)):
        return max([_max_abs(v) for v in d] + [0.0])
    if isinstance(d, (int, float)) and not isinstance(d, bool):
        return abs(float(d))
    return 0.0


def test_lex_keeps_internal_ranges_small():
    from neurify import _core

    lex = nf.compile(scores, output="rank")
    direct, _ = _core.compile(lex.source_graph.encode_outputs("rank", lex=False))
    w_lex, w_direct = _max_abs(lex._core.to_dict()), _max_abs(direct.to_dict())
    assert w_lex < 1e3 < 1e6 < w_direct, (w_lex, w_direct)


def test_real_scores_use_margins():
    net = nf.compile(small, output="rank")
    assert net.report["margin_sites"] > 0
    assert nf.verify(net).ok


def one_score(x: nf.Int[0, 3]):
    return x + 1


def test_output_needs_several_scores():
    with pytest.raises(nf.CompileError):
        nf.compile(one_score, output="rank")
