"""Call-local activation collectors that preserve normal nn.Module hooks."""

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Capture:
    detach: bool
    structured: bool
    values: dict = field(default_factory=dict)


_ACTIVE = ContextVar("neurify_activation_capture", default=())


def current_capture(module):
    return next((request for ident, request in reversed(_ACTIVE.get()) if ident == id(module)), None)


def capture_call(module, x, *, detach, structured):
    request = Capture(detach, structured)
    token = _ACTIVE.set(_ACTIVE.get() + ((id(module), request),))
    try:
        output = module(x)
    finally:
        _ACTIVE.reset(token)
    return output, request.values
