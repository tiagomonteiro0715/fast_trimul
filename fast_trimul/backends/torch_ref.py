# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The universal backend: pure-torch triangle multiply.

Correct on any device torch supports (CUDA, CPU, TPU-via-XLA, Intel-XPU), which is
what makes the library vendor-agnostic. It is the last link in every fallback
chain, so an unsupported shape/dtype degrades here instead of crashing. Slower
than the fused kernels (larger constants), same asymptotics:
O(B*N^3*d + B*N^2*d^2) time, O(B*N^2*d) memory.
"""

import torch

from ..core.registry import backend

_ALL_FLOATS = {torch.float16, torch.bfloat16, torch.float32}


@backend("torch", dtypes=_ALL_FLOATS, min_align=1, supports_graph=False)
class TorchReferenceBackend:
    def __init__(self, caps):
        self.caps = caps

    def execute(self, inp, params):
        """`params` is the fp16 weight module; reuse its exact pure-torch path so
        the result matches the fused kernel to fp16 tolerance."""
        z = inp.tensor
        out_dtype = z.dtype
        weight_dtype = params.proj_a.weight.dtype
        z = z.to(weight_dtype)
        if inp.mask is not None:
            z = z * inp.mask.unsqueeze(-1).to(weight_dtype)
        out = params._torch_forward(z, params._params())
        if inp.mask is not None:
            out = out * inp.mask.unsqueeze(-1).to(out.dtype)
        return out.to(out_dtype)
