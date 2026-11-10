"""Helpers usable inside compiled functions (they also run as plain Python)."""


def where(cond, a, b):
    """Branch-free select: ``a if cond else b`` (works on nested structures)."""
    return a if cond else b


def relu(x):
    """max(x, 0) as a single ReLU unit."""
    return x if x > 0 else 0 * x
