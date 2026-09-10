# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""A faster fallback than pure torch: NVIDIA cuEquivariance's fused triangle
multiply.

This backend sits between `cuda` (the CUTLASS CuTe path) and `torch` (the always
-correct floor) in the fallback chain. When the CuTe kernel can't run -- a non
-Ampere GPU, or a runtime failure -- the dispatcher tries this cuEquivariance
kernel before dropping all the way to pure torch (the reviewer's suggestion #1).

The call mirrors OpenFold-3's own `_cueq_triangle_mult` exactly (same
`cuequivariance_torch.triangle_multiplicative_update`, same weight layout), so it
matches a known-good path rather than a guess:
  * pass the RAW pair rep (the kernel does its own layer_norm_in),
  * fuse the a/b projections into single weights (cat over the output dim),
  * the kernel returns the delta; we add the residual only if the module wants it.

It is guarded: if `cuequivariance_torch` isn't installed the backend never
registers (see backends/__init__), and the cueq kernel requires the channel dim
to be a multiple of 32 -- otherwise we raise and the dispatcher falls through to
the correct torch backend. It never returns a silently-wrong result.
"""

import torch

from ..core.registry import backend

# Exact function OF-3 uses; a missing package makes this import (and registration)
# fail, which backends/__init__ catches -> the backend is simply absent.
from cuequivariance_torch import triangle_multiplicative_update as _TRIMUL

_FLOATS = {torch.float16, torch.bfloat16, torch.float32}


@backend("cueq", dtypes=_FLOATS, min_align=1, supports_graph=False)
class CuEquivarianceBackend:
    def __init__(self, caps):
        self.caps = caps

    def execute(self, inp, params):
        z = inp.tensor
        if z.shape[-1] % 32 != 0:                 # cueq kernel constraint -> torch fallback
            raise ValueError("cueq triangle-multiply needs channel dim % 32 == 0")
        p = params
        out = _TRIMUL(
            z,                                    # raw pair rep; kernel does layer_norm_in
            direction="outgoing" if p.mode == "outgoing" else "incoming",
            mask=inp.mask,
            norm_in_weight=p.norm_in.weight, norm_in_bias=p.norm_in.bias,
            g_in_weight=torch.cat([p.gate_a.weight, p.gate_b.weight]),
            p_in_weight=torch.cat([p.proj_a.weight, p.proj_b.weight]),
            norm_out_weight=p.norm_out.weight, norm_out_bias=p.norm_out.bias,
            p_out_weight=p.proj_out.weight,
            g_out_weight=p.proj_g.weight,
            eps=p.norm_in.eps,
        )
        if getattr(p, "residual", False):         # kernel returns the delta only
            out = z + out
        return out.to(inp.tensor.dtype)
