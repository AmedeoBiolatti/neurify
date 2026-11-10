"""Bridge between the Rust interpreter and the Python process.

The Rust frontend parses and interprets source; whenever it meets a name the
source does not define, it asks this host, which looks in the function's
closure cells, globals and builtins and answers with one of:

    ("value", data)        plain data (numbers, str, lists/tuples/dicts of them)
    ("function", src, ...) a Python function to inline (its source)
    ("builtin", name)      an operation the interpreter implements natively
    ("object", handle)     anything else; attributes and calls come back here
    ("missing",)
"""

from __future__ import annotations

import builtins
import inspect
import math
import textwrap
import types

_BUILTIN_NAMES = {
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "list", "tuple", "dict", "int",
    "float", "bool", "round", "divmod", "pow", "sorted", "reversed", "any", "all", "print", "isinstance",
}


def _known_builtins() -> dict[int, str]:
    from . import ops

    known = {id(getattr(builtins, n)): n for n in _BUILTIN_NAMES}
    known[id(math.floor)] = "math.floor"
    known[id(math.ceil)] = "math.ceil"
    known[id(math.trunc)] = "math.trunc"
    known[id(math.fabs)] = "math.fabs"
    known[id(ops.where)] = "neurify.where"
    known[id(ops.relu)] = "neurify.relu"
    return known


def _plain(v, depth=0) -> bool:
    if depth > 50:
        return False
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in v.items())
    return False


def function_source(fn: types.FunctionType) -> tuple[str, str, int]:
    src = getattr(fn, "__neurify_source__", None)  # generated functions (neurify.merge)
    if src is not None:
        return src, fn.__code__.co_filename, fn.__code__.co_firstlineno
    src = textwrap.dedent(inspect.getsource(fn))
    return src, fn.__code__.co_filename, fn.__code__.co_firstlineno


class Host:
    def __init__(self):
        self.objs: dict[int, object] = {}
        self.known = _known_builtins()

    def handle(self, obj) -> int:
        h = id(obj)
        self.objs[h] = obj
        return h

    def classify(self, v):
        from .regions import Pipeline, Region
        from .scopes import ScopedNetwork

        if isinstance(v, (Region, Pipeline, ScopedNetwork)):
            raise TypeError("region and pipeline objects cannot be inlined; compose them with neurify.pipeline")
        if id(v) in self.known and not isinstance(v, (bool, int, float, str)):
            return ("builtin", self.known[id(v)])
        try:
            import numpy as np

            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.generic):
                v = v.item()
        except ImportError:  # pragma: no cover
            pass
        if _plain(v):
            return ("value", v)
        if isinstance(v, types.FunctionType) and v.__name__ != "<lambda>":
            try:
                src, file, line = function_source(v)
            except (OSError, TypeError):
                return ("object", self.handle(v))
            h = self.handle(v)
            return ("function", src, v.__name__, file, line, h, h)
        return ("object", self.handle(v))

    # --- called from Rust -------------------------------------------------

    def resolve(self, ctx: int, name: str):
        fn = self.objs.get(ctx)
        if fn is None:
            return ("missing",)
        code = getattr(fn, "__code__", None)
        if code is not None and fn.__closure__:
            for n, cell in zip(code.co_freevars, fn.__closure__):
                if n == name:
                    try:
                        return self.classify(cell.cell_contents)
                    except ValueError:  # empty cell
                        return ("missing",)
        g = getattr(fn, "__globals__", {})
        if name in g:
            return self.classify(g[name])
        if name in _BUILTIN_NAMES:
            return ("builtin", name)
        if hasattr(builtins, name):
            return self.classify(getattr(builtins, name))
        return ("missing",)

    def getattr(self, h: int, attr: str):
        return self.classify(getattr(self.objs[h], attr))

    def call(self, h: int, args: list):
        out = self.objs[h](*args)
        try:
            import numpy as np

            if isinstance(out, np.generic):
                out = out.item()
            elif isinstance(out, np.ndarray):
                out = out.tolist()
        except ImportError:  # pragma: no cover
            pass
        return out
