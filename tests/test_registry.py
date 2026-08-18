# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The plug-in registry: @backend / @weights_for + lookups (CPU-only)."""

import pytest
import torch

from fast_trimul.core import registry
from fast_trimul.core.context import BackendCapabilities


def test_register_and_get_backend():
    @registry.backend("__test_be", dtypes={torch.float16}, min_align=8)
    class _BE:
        def __init__(self, caps):
            self.caps = caps

    try:
        assert "__test_be" in registry.list_backends()
        be = registry.get_backend("__test_be")
        assert be is not None
        assert isinstance(be.caps, BackendCapabilities)
    finally:
        registry._BACKENDS.pop("__test_be", None)


def test_get_backend_unknown_is_none():
    assert registry.get_backend("__nope__") is None


def test_register_and_get_loader():
    @registry.weights_for("__test_src")
    def _loader(sd):
        return sd

    try:
        assert registry.get_loader("__test_src") is _loader
    finally:
        registry._LOADERS.pop("__test_src", None)


def test_get_loader_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_loader("__does_not_exist__")


def test_builtin_loaders_present():
    for src in ("openfold", "openfold3", "protenix", "boltz", "chai"):
        assert callable(registry.get_loader(src))


def test_torch_backend_registered():
    assert "torch" in registry.list_backends()
