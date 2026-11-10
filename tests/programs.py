"""Programs used by the test-suite (module-level so their source is inspectable)."""

import math

import neurify as nf
from neurify import Bool, Int, Real

OFFSET = 3


def affine_real(x: Real[-4, 4], y: Real[-2, 2]):
    return 3 * x - y / 2 + OFFSET


def piecewise_real(x: Real[-3, 3], y: Real[-3, 3]):
    return max(x, y) - abs(x - 1) + min(2 * y, 0.5) + nf.relu(x + y)


def int_branches(a: Int[-6, 9], b: Int[0, 7]):
    if a > b:
        return (a * b) % 7
    elif a == b:
        return 0
    m = max(a, b) // 2
    return m * 3 + (1 if a % 2 == 0 else -1)


def helper_scale(v, k=2):
    return v * k


def loops(n: Int[0, 8], x: Int[0, 10]):
    s = 0
    for i in range(n):
        if i * x > 20:
            break
        if i == 2:
            continue
        s += helper_scale(i)
    while x > 0:
        x -= 3
        s += 1
    return s, x


def gcd(a: Int[1, 24], b: Int[1, 24]):
    while b:
        a, b = b, a % b
    return a


def dict_io(cfg: {"w": Int[1, 3], "flip": Bool()}, x: Int[0, 9]):
    out = {"raw": x * cfg["w"]}
    if cfg["flip"]:
        out["raw"] = -out["raw"]
    out["pos"] = out["raw"] > 0
    out.update(scaled=out["raw"] // 2)
    return out


def swap(a, i, j):
    a[i], a[j] = a[j], a[i]


def bubble(xs: nf.Vec(Int[0, 3], 4), k: Int[0, 3]):
    for n in range(len(xs) - 1, 0, -1):
        for i in range(n):
            if xs[i] > xs[i + 1]:
                swap(xs, i, i + 1)
    xs[k] = xs[k] + 10
    return xs, xs[k], sorted(xs, reverse=True)[0]


def closures(x: Int[0, 9]):
    def sq(v):
        return v * v

    f = lambda v: v + 1  # noqa: E731
    ys = [sq(f(i)) for i in range(3) if i != 1]
    return sum(ys) + sq(x) % 5


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def opaque(n: Int[0, 15]):
    return fib(n) % 10 + math.comb(n, 2) + math.isqrt(n)


def assumption(x: Int[-5, 5], y: Int[-5, 5]):
    assert y != 0
    if x < -3:
        raise ValueError("outside domain")
    return x // y


def real_cmp(x: Real[-3, 3]):
    return 2 * x if x > 0 else math.floor(x)


def poly(x: Real[-2, 2], y: Real[-1, 1]):
    return x * y - x ** 2 + 3 * y ** 3


def bools(p: Bool(), q: Bool(), r: Bool()):
    return (p and not q) or (q ^ r), any([p, q, r]), all([p, q])


EXACT_ALL_ARCHS = [affine_real, piecewise_real, int_branches, loops, gcd, dict_io, bubble, closures, opaque, assumption, bools]


def seq_prog(xs: nf.Seq(Int[0, 9], 6), k: Int[0, 5], t: Int[0, 9]):
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    count = 0
    for x in xs:
        if x >= t:
            count += 1
    return {"argmax": best, "max": xs[best], "pick": xs[k] * 2 - xs[0], "count": count}


def seq_real(xs: nf.Seq(Real[-1, 1], 5), w: nf.Seq(Real[0, 2], 5), i: Int[0, 4]):
    # two features on the same tokens; real-valued weighted read
    return xs[i] * 3 + w[i] - xs[(i + 1) % 5]
