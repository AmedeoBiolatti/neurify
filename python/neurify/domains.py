"""Input domains.

Every parameter of a compiled function needs a domain. Exactness is
guaranteed on the declared domain, and bounds drive gadget selection, so
tight bounds give smaller networks.

    Real(lo, hi)        real number in [lo, hi]  (unbounded by default)
    Int(lo, hi)         integer in {lo, ..., hi} (inclusive)
    Bool()              True / False
    Vec(dom, n)         list of n values of domain dom
    Seq(dom, n)         like Vec, but with arch="transformer" its elements live
                        on tokens 1..n and are read through attention
    [d0, d1], (d0, d1)  fixed-structure list / tuple
    {"k": dom, ...}     dict with known keys (pytree-like, as in JAX)
    Static(value)       compile-time constant argument (like jit static args)

Annotation syntax also works: ``def f(x: Int[0, 9], y: Real[-1, 1]): ...``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class Domain:
    """Base class of input domains."""

    def spec(self) -> tuple:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class Real(Domain):
    lo: float = -math.inf
    hi: float = math.inf

    def __class_getitem__(cls, item):
        lo, hi = item
        return cls(lo, hi)

    def spec(self):
        return ("real", float(self.lo), float(self.hi))


@dataclass(frozen=True)
class Int(Domain):
    lo: float = -math.inf
    hi: float = math.inf

    def __class_getitem__(cls, item):
        lo, hi = item
        return cls(lo, hi)

    def spec(self):
        return ("int", float(self.lo), float(self.hi))


@dataclass(frozen=True)
class Bool(Domain):
    def spec(self):
        return ("bool",)


@dataclass(frozen=True)
class Vec(Domain):
    elem: Any
    n: int

    def __class_getitem__(cls, item):
        elem, n = item
        return cls(elem, n)

    def spec(self):
        return ("list", [as_domain(self.elem).spec()] * self.n)


@dataclass(frozen=True)
class Seq(Domain):
    """A list of n scalars that a transformer lays out as a token sequence."""

    elem: Any
    n: int

    def __class_getitem__(cls, item):
        elem, n = item
        return cls(elem, n)

    def spec(self):
        e = as_domain(self.elem)
        if not isinstance(e, (Real, Int, Bool)):
            raise TypeError("Seq elements must be Real, Int or Bool")
        return ("seq", [e.spec()] * self.n)


@dataclass(frozen=True)
class Static(Domain):
    value: Any

    def spec(self):
        return ("static", self.value)


def as_domain(d: Any) -> Domain | list | tuple | dict:
    if isinstance(d, Domain):
        return d
    if d is bool:
        return Bool()
    if d in (Real, Int, Bool):
        return d()
    if isinstance(d, range):
        if d.step != 1 or len(d) == 0:
            raise ValueError(f"only non-empty step-1 ranges can be used as domains, got {d}")
        return Int(d.start, d.stop - 1)
    if isinstance(d, (list, tuple, dict)):
        return d
    raise TypeError(f"not a domain: {d!r}")


def to_spec(d: Any) -> tuple:
    d = as_domain(d)
    if isinstance(d, Domain):
        return d.spec()
    if isinstance(d, list):
        return ("list", [to_spec(x) for x in d])
    if isinstance(d, tuple):
        return ("tuple", [to_spec(x) for x in d])
    if isinstance(d, dict):
        return ("dict", [(k, to_spec(v)) for k, v in d.items()])
    raise TypeError(f"not a domain: {d!r}")
