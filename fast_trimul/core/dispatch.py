# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The switchboard: pick a backend, guard the input, run it, fall back on failure.

Multi-step fallback chain: the preferred backend is tried first, then "torch"
(the universal, always-correct backend). If a backend can't take the input (guard
returns None) or raises, the dispatcher moves to the next candidate. `torch` is
last and always runnable, so a call never dies from an unsupported shape/dtype --
it degrades to the slow-but-correct path instead of an illegal-memory crash.

Cost: candidate selection is O(1) (fixed short list); the op itself dominates.
"""

from .registry import get_backend
from .context import normalize
from .decorators import active_backend


def _order(device, prefer: str) -> list:
    """The fallback chain for this call, most-preferred first."""
    if prefer == "auto":
        prefer = active_backend()
    if prefer != "auto":
        return [prefer] + (["torch"] if prefer != "torch" else [])
    if getattr(device, "type", None) == "cuda":
        return ["cuda", "torch"]        # cuda -> torch
    return ["torch"]


def run(z, mask, params, prefer: str = "auto"):
    """Run the triangle-multiply op for `params` on `z`, walking the fallback chain."""
    last_error = None
    for name in _order(z.device, prefer):
        be = get_backend(name)
        if be is None:
            continue
        inp = normalize(z, mask, be.caps)
        if inp is None:                 # backend can't handle this input -> next
            continue
        try:
            return be.execute(inp, params)
        except Exception as error:      # kernel/runtime failure -> next (usually torch)
            last_error = error
    raise RuntimeError(f"fast_trimul: no backend could run the op ({last_error})")
