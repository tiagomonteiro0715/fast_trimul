# Copyright (c) 2026 Tiago Monteiro. MIT License.
"""Low-level functional entry point (FlashAttention-style)."""

import torch


def triangle_multiplication(z: torch.Tensor, params, mask: torch.Tensor = None) -> torch.Tensor:
    """Fused Triangle Multiplicative Update.

    z:      (B, N, N, d_z) pair representation, any float dtype.
    params: a module holding the weights and mode -- a
            `TriangleMultiplicativeUpdateKernelFused` (or `FastTriangleMultiplication._impl`),
            kept in fp16 (the custom kernels are fp16-only).
    mask:   optional (B, N, N) mask; applied to the pair representation in and out.

    Returns a tensor in z's original dtype. Autograd flows (backward is a correct
    torch recompute inside the kernel module -- correct, not yet fast).
    """
    orig = z.dtype
    if mask is not None:
        z = z * mask.unsqueeze(-1).to(z.dtype)
    out = params(z.half())                         # fp16 kernels
    out = out.to(orig)
    if mask is not None:
        out = out * mask.unsqueeze(-1).to(out.dtype)
    return out
