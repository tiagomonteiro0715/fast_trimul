# Copyright (c) 2026 Tiago Monteiro. MIT License.
"""fast_trimul: fused Triangle Multiplicative Update on CUTLASS CuTe DSL kernels."""

from . import functional, integrations, nn
from .nn import FastTriangleMultiplication

__all__ = ["FastTriangleMultiplication", "functional", "nn", "integrations"]
__version__ = "0.0.1"
