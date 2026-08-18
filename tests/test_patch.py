# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The library patchers: _adapter / _patch swap classes (CPU-only)."""

import sys
import types

import fast_trimul
from fast_trimul.integrations import patch
from fast_trimul.ops.triangle import FastTriangleMultiplication


def test_adapter_is_fast_trimul_subclass():
    assert issubclass(patch._adapter("outgoing"), FastTriangleMultiplication)
    assert issubclass(patch._adapter("incoming"), FastTriangleMultiplication)


def test_patch_swaps_module_classes():
    mod = types.ModuleType("__fake_stack")

    class TriangleMultiplicationOutgoing:
        pass

    class TriangleMultiplicationIncoming:
        pass

    mod.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing
    mod.TriangleMultiplicationIncoming = TriangleMultiplicationIncoming
    sys.modules["__fake_stack"] = mod
    try:
        patch._patch("__fake_stack",
                     ["TriangleMultiplicationOutgoing"],
                     ["TriangleMultiplicationIncoming"])
        assert issubclass(mod.TriangleMultiplicationOutgoing, FastTriangleMultiplication)
        assert issubclass(mod.TriangleMultiplicationIncoming, FastTriangleMultiplication)
    finally:
        sys.modules.pop("__fake_stack", None)


def test_patch_ignores_missing_names():
    mod = types.ModuleType("__fake_stack2")
    sys.modules["__fake_stack2"] = mod
    try:
        patch._patch("__fake_stack2", ["NotPresent"], [])
        assert not hasattr(mod, "NotPresent")
    finally:
        sys.modules.pop("__fake_stack2", None)


def test_patch_functions_are_callable():
    for name in ("patch_openfold", "patch_openfold3", "patch_boltz", "patch_protenix"):
        assert callable(getattr(fast_trimul, name))
