"""
Convenience wrappers around the global autograd enable/disable switches.

These proxy to the C-side toggles via `greyml.tensor`.
"""
from .tensor import no_grad, enable_grad, _set_grad_enabled, _is_grad_enabled

__all__ = ["no_grad", "enable_grad", "set_grad_enabled", "is_grad_enabled"]


def set_grad_enabled(enabled: bool):
    _set_grad_enabled(bool(enabled))


def is_grad_enabled() -> bool:
    return bool(_is_grad_enabled())
