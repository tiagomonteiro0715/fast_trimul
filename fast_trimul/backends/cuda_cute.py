# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The CUDA backend: a thin wrapper over the existing CUTLASS CuTe kernels.

This is the ONLY new code that touches the kernel path, and it only forwards to
the existing `functional.triangle_multiplication`. The kernels themselves
(`_kernels.py`) are unchanged. Needs N % 8 == 0 (declared as min_align) and a
CUDA device; the dispatcher routes non-conforming inputs to the torch backend.
Same asymptotics as the reference, with fp16 tensor-core constants.
"""

import torch

from ..core.registry import backend
from ..functional import triangle_multiplication

_FLOATS = {torch.float16, torch.bfloat16, torch.float32}   # cast to fp16 inside


@backend("cuda", dtypes=_FLOATS, min_align=8, supports_graph=True)
class CudaCuteBackend:
    def __init__(self, caps):
        self.caps = caps

    def execute(self, inp, params):
        return triangle_multiplication(inp.tensor, params, mask=inp.mask)
