# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The CUDA backend: a thin wrapper over the existing CUTLASS CuTe kernels.

This is the ONLY new code that touches the kernel path, and it only forwards to
the existing `functional.triangle_multiplication`. The kernels themselves
(`_kernels.py`) are unchanged. The tensor-core kernels need the sequence dim N to
be a multiple of 8, so instead of falling back to torch on N % 8 != 0 we pad N up
to the next multiple of 8, run the kernel, and slice the result back -- keeping
the fast path for every N. `params._pad_n` tells the kernel to zero the padded
contraction rows of a/b so the padding never leaks into the valid output. Same
asymptotics as the reference, with fp16 tensor-core constants.
"""

import torch
import torch.nn.functional as F

from ..core.registry import backend
from ..functional import triangle_multiplication

_FLOATS = {torch.float16, torch.bfloat16, torch.float32}   # cast to fp16 inside
_ALIGN = 8                                                 # kernel tile on the N dim


@backend("cuda", dtypes=_FLOATS, min_align=1, supports_graph=True)
class CudaCuteBackend:
    def __init__(self, caps):
        self.caps = caps

    def execute(self, inp, params):
        z = inp.tensor
        n = z.shape[1]
        pad = (-n) % _ALIGN
        if pad == 0:                                       # already aligned: fast path
            return triangle_multiplication(z, params, mask=inp.mask)
        # pad both N dims (z is B,N,N,d -> pad dims -2 and -3) and the mask, then
        # tell the kernel the valid length so it zeros the padded k-slab of a/b.
        zp = F.pad(z, (0, 0, 0, pad, 0, pad))
        mp = None if inp.mask is None else F.pad(inp.mask, (0, pad, 0, pad))
        params._pad_n = n
        try:
            out = triangle_multiplication(zp, params, mask=mp)
        finally:
            params._pad_n = None
        return out[:, :n, :n, :].contiguous()
