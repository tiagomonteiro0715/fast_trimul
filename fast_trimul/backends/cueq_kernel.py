# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""A faster fallback than pure torch: NVIDIA cuEquivariance's fused triangle
multiply.

This backend sits between `cuda` (the CUTLASS CuTe path) and `torch` (the always
-correct floor) in the fallback chain. When the CuTe kernel can't run -- a non
-Ampere GPU, or a runtime failure -- the dispatcher tries this cuEquivariance
kernel before dropping all the way to pure torch, which is typically much faster
than the eager reference (the reviewer's suggestion #1).

It is guarded end to end: if `cuequivariance_torch` isn't installed, the backend
doesn't register at all (see backends/__init__), and any per-call failure raises,
so the dispatcher simply falls through to the correct torch backend. It never
returns a silently-wrong result.

Weights come from our fused module (`params`), already in fp16 in our layout; we
hand cuEquivariance the same LayerNorm / projection / gate tensors it expects.
"""

import torch

from ..core.registry import backend

# Import here so a missing package makes registration fail loudly *at import*,
# which backends/__init__ catches -> the backend is simply absent.
import cuequivariance_torch as _cuet   # noqa: F401  (presence gate)

_FLOATS = {torch.float16, torch.bfloat16, torch.float32}

# cuEquivariance exposes the fused op under one of a couple of names across
# versions; pick whichever is present, else this backend never registers.
_TRIMUL = (getattr(_cuet, "triangle_multiplicative_update", None)
           or getattr(_cuet, "triangle_multiply", None))
if _TRIMUL is None:
    raise ImportError("cuequivariance_torch has no triangle_multiplicative_update")


@backend("cueq", dtypes=_FLOATS, min_align=1, supports_graph=False)
class CuEquivarianceBackend:
    def __init__(self, caps):
        self.caps = caps

    def execute(self, inp, params):
        z = inp.tensor.half()
        if inp.mask is not None:
            z = z * inp.mask.unsqueeze(-1).half()
        p = params
        out = _TRIMUL(
            z,
            direction=p.mode,                         # 'outgoing' | 'incoming'
            mask=inp.mask,
            norm_in_weight=p.norm_in.weight, norm_in_bias=p.norm_in.bias,
            norm_out_weight=p.norm_out.weight, norm_out_bias=p.norm_out.bias,
            p_in_weight=torch.cat([p.proj_a.weight, p.proj_b.weight], dim=0),
            g_in_weight=torch.cat([p.gate_a.weight, p.gate_b.weight], dim=0),
            p_out_weight=p.proj_out.weight,
            g_out_weight=p.proj_g.weight,
        )
        if inp.mask is not None:
            out = out * inp.mask.unsqueeze(-1).to(out.dtype)
        return out.to(inp.tensor.dtype)
