# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Backward-compatible re-exports.

The module implementation moved to `ops/` (the module) and `core/` (registry,
guard, dispatch, decorators). This shim keeps `from fast_trimul.nn import
FastTriangleMultiplication` and the OpenFold name map working.
"""

from .ops.triangle import FastTriangleMultiplication
from .integrations.loaders import _OPENFOLD_TO_FAST   # noqa: F401  (kept for compatibility)

__all__ = ["FastTriangleMultiplication"]
