# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The input guard ("unified adapter standard").

Turns any raw tensor into a `NormalizedInput` a backend can trust: contiguous
layout, a dtype the backend supports, 64-bit strides, and an alignment check so
the kernel never reads unallocated edges. If the tensor cannot be made to satisfy
a backend's capabilities, `normalize` returns None and the dispatcher falls back
to the next backend in the chain.

Cost: O(1) when the tensor already conforms (metadata only); O(B*N*N*d) when it
must copy/cast to become contiguous or change dtype.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can handle. Declared once at registration time."""
    dtypes: frozenset          # e.g. {torch.float16, torch.bfloat16, torch.float32}
    min_align: int = 1         # required multiple of the sequence dim N (kernel tile)
    supports_graph: bool = False

    def __post_init__(self):
        # allow passing a plain set at registration; store immutable
        object.__setattr__(self, "dtypes", frozenset(self.dtypes))


@dataclass
class NormalizedInput:
    """A cleaned-up tensor bundle handed to a backend."""
    tensor: torch.Tensor
    mask: Optional[torch.Tensor]
    shape: Tuple[int, ...]     # (B, N, N, d)
    strides: Tuple[int, ...]   # int64, so N=3072 addressing never overflows 32-bit


def normalize(z: torch.Tensor, mask, caps: BackendCapabilities) -> Optional[NormalizedInput]:
    """Adapt `z`/`mask` to `caps`, or return None if this backend can't take it."""
    if z.dim() != 4:
        return None
    n = z.shape[1]
    if caps.min_align > 1 and n % caps.min_align != 0:
        return None                                   # e.g. cuda needs N % 8 == 0
    if not z.is_contiguous():                         # layout guard
        z = z.contiguous()
    if z.dtype not in caps.dtypes:                    # dtype guard
        target = torch.float16 if torch.float16 in caps.dtypes else torch.float32
        z = z.to(target)
    if mask is not None and not mask.is_contiguous():
        mask = mask.contiguous()
    return NormalizedInput(z, mask, tuple(z.shape), tuple(z.stride()))
