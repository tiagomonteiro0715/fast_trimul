# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The plug-in board.

Three lookup tables and the decorators that fill them. Adding a new hardware
backend or a new library weight-map is "write a function/class + one decorator",
never an edit to the core. All lookups and registrations are O(1) dict ops.
"""

from .context import BackendCapabilities

# name -> backend instance (e.g. "cuda", "torch")   |  source -> weight-remap fn
_BACKENDS: dict = {}
_LOADERS: dict = {}


def backend(name: str, **capabilities):
    """Class decorator: register a HardwareBackend under `name`.

    Example:
        @backend("cuda", dtypes={torch.float16}, min_align=8, supports_graph=True)
        class CudaCuteBackend: ...
    """
    def register(cls):
        _BACKENDS[name] = cls(BackendCapabilities(**capabilities))
        return cls
    return register


def get_backend(name: str):
    return _BACKENDS.get(name)


def list_backends() -> list:
    """Names of every backend that registered on import (what's available here)."""
    return list(_BACKENDS)


def weights_for(source: str):
    """Function decorator: register a state_dict remap for a library `source`.

    The function takes the source state_dict and returns a dict keyed by *our*
    parameter names. Example: @weights_for("openfold").
    """
    def register(fn):
        _LOADERS[source] = fn
        return fn
    return register


def get_loader(source: str):
    if source not in _LOADERS:
        raise KeyError(
            f"no weight loader registered for '{source}' "
            f"(available: {sorted(_LOADERS)})"
        )
    return _LOADERS[source]
