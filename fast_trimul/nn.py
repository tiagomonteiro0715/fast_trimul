# Copyright (c) 2026 Tiago Monteiro. MIT License.
"""High-level nn.Module API matching target-library TriMul signatures."""

import torch
import torch.nn as nn

from ._kernels import TriangleMultiplicativeUpdateKernelFused
from .functional import triangle_multiplication


class FastTriangleMultiplication(nn.Module):
    """Drop-in Triangle Multiplicative Update on fused CuTe DSL kernels.

    Args:
        d_z:  pair-representation channels (a.k.a. c_z).
        d_c:  intermediate channels (a.k.a. c_hidden); defaults to d_z.
        mode: 'outgoing' or 'incoming'.

    Kernels are fp16-only, so weights are stored in fp16 and bf16/fp32 inputs are
    cast in and cast back. Keep the module in fp16 (do not call .float()/.bfloat16()
    on it). Requires a CUDA GPU and `nvidia-cutlass-dsl`.
    """

    def __init__(self, d_z: int = 128, d_c: int = None, mode: str = "outgoing"):
        super().__init__()
        d_c = d_z if d_c is None else d_c
        self.mode = mode
        self._impl = TriangleMultiplicativeUpdateKernelFused(d_z, d_c, mode).half()

    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return triangle_multiplication(z, self._impl, mask=mask)
