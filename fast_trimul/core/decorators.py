# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The front door: explicit, scoped, side-effect-free acceleration.

`accelerated(...)` is a context manager that sets the preferred backend for the
enclosed block only (via a ContextVar, so it never leaks into other code or other
threads). `@accelerate` runs a function inside that scope. Neither touches global
state or monkeypatches anything, which keeps them safe under strict runtime
policies. Both are O(1) around the wrapped call.
"""

import contextlib
import functools
from contextvars import ContextVar

_ACTIVE_BACKEND: ContextVar = ContextVar("fast_trimul_backend", default="auto")


def active_backend() -> str:
    """The backend preference in force for the current scope ("auto" by default)."""
    return _ACTIVE_BACKEND.get()


@contextlib.contextmanager
def accelerated(backend: str = "auto"):
    """Scoped acceleration: `with accelerated("cuda"): out = model(z)`.

    Reverts automatically on exit; affects only code run inside the block.
    """
    token = _ACTIVE_BACKEND.set(backend)
    try:
        yield
    finally:
        _ACTIVE_BACKEND.reset(token)


def accelerate(func=None, *, backend: str = "auto"):
    """Run a function under `accelerated(backend)`. Usable as `@accelerate` or
    `@accelerate(backend="cuda")`. Explicit opt-in, no global side effects."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with accelerated(backend):
                return fn(*args, **kwargs)
        return inner
    return wrap(func) if callable(func) else wrap
