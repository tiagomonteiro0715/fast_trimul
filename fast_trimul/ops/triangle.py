# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""FastTriangleMultiplication: the drop-in module.

Holds the fp16 weights (`_impl`) and routes forward() through the dispatcher, so
it automatically uses the fastest available backend and falls back to pure torch
when the fast path can't run. Weights load from any supported library via one
`load_weights(sd, source=...)` call. CUDA-graph capture (`graphed`) and its replay
path are unchanged. The kernel module is imported lazily so `import fast_trimul`
does not require CUTLASS until you actually build a module.
"""

import torch
import torch.nn as nn

from ..core import dispatch
from ..core.registry import get_loader


class FastTriangleMultiplication(nn.Module):
    """Drop-in Triangle Multiplicative Update on fused kernels, with a portable
    fallback.

    Args:
        d_z:      pair-representation channels (a.k.a. c_z).
        d_c:      intermediate channels (a.k.a. c_hidden); defaults to d_z.
        mode:     'outgoing' or 'incoming'.
        residual: True returns z + delta; False returns delta only (use False to
                  match target libraries, which add their residual outside).
        backend:  'auto' (pick best, fall back to torch), or force 'cuda'/'torch'.
    """

    def __init__(self, d_z: int = 128, d_c: int = None, mode: str = "outgoing",
                 residual: bool = True, backend: str = "auto"):
        super().__init__()
        from .._kernels import TriangleMultiplicativeUpdateKernelFused   # lazy: needs CUTLASS
        d_c = d_z if d_c is None else d_c
        self.mode = mode
        self.backend = backend
        self._impl = TriangleMultiplicativeUpdateKernelFused(d_z, d_c, mode, residual).half()
        self._graph = None
        self._static_in = None
        self._static_mask = None
        self._static_out = None

    # ------------------------------------------------------------------ weights
    def _load_remapped(self, remapped):
        """Load an already-remapped state dict; zero any bias the source omitted
        (bias-free libraries) and raise on any unexpected/missing non-bias key."""
        result = self._impl.load_state_dict(remapped, strict=False)
        if result.unexpected_keys:
            raise KeyError(f"unexpected keys after remap: {result.unexpected_keys}")
        missing_non_bias = [k for k in result.missing_keys if not k.endswith(".bias")]
        if missing_non_bias:
            raise KeyError(f"missing keys after remap: {missing_non_bias}")
        own = dict(self._impl.named_parameters())
        for k in result.missing_keys:
            own[k].data.zero_()
        return result

    def load_weights(self, state_dict, source: str):
        """Load pretrained weights from a library, remapping names by `source`
        ('openfold' | 'openfold3' | 'protenix' | 'boltz' | 'chai')."""
        remapped = get_loader(source)(dict(state_dict))
        return self._load_remapped(remapped)

    # backward-compatible named loaders (thin aliases over load_weights)
    def load_openfold_state_dict(self, sd):  return self.load_weights(sd, "openfold")
    def load_openfold3_state_dict(self, sd): return self.load_weights(sd, "openfold3")
    def load_protenix_state_dict(self, sd):  return self.load_weights(sd, "protenix")
    def load_boltz_state_dict(self, sd):     return self.load_weights(sd, "boltz")

    # ------------------------------------------------------------------- cuda graph
    def graphed(self, z: torch.Tensor, mask: torch.Tensor = None):
        """Capture a CUDA graph of forward at `z`'s shape, then replay it on every
        call (inference only; fixed shape). Removes per-launch overhead."""
        from ..functional import triangle_multiplication
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):                       # warmup: JIT-compile every kernel
                triangle_multiplication(z, self._impl, mask=mask)
        torch.cuda.current_stream().wait_stream(stream)

        self._static_in = z.clone()
        self._static_mask = None if mask is None else mask.clone()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self._static_out = triangle_multiplication(
                self._static_in, self._impl, mask=self._static_mask
            )
        return self

    # ---------------------------------------------------------------------- forward
    def forward(self, z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self._graph is not None:                  # replay the captured graph
            self._static_in.copy_(z)
            if mask is not None and self._static_mask is not None:
                self._static_mask.copy_(mask)
            self._graph.replay()
            return self._static_out
        return dispatch.run(z, mask, self._impl, prefer=self.backend)
