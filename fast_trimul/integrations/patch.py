# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""One-line integration: replace a library's Triangle Multiplicative Update with
fast_trimul, in place, before you build the model.

    import fast_trimul
    fast_trimul.patch_openfold3()     # OpenFold-3 now runs on the fast kernel
    # ... build and run OpenFold-3 exactly as usual ...

Each `patch_*` swaps the library's TriMul classes (outgoing/incoming, plus the
fused variants where present) for a fast_trimul adapter that matches the library's
constructor, ignores its extra forward kwargs, and uses `residual=False` (the block
adds its own residual). Call BEFORE constructing the model.
"""

import importlib

from ..ops.triangle import FastTriangleMultiplication


def _adapter(mode):
    """A fast_trimul module that drops straight into a library's TriMul slot."""
    class _FastTriMul(FastTriangleMultiplication):
        def __init__(self, c_z, c_hidden=None, *args, **kwargs):
            super().__init__(d_z=c_z, d_c=c_hidden or c_z, mode=mode, residual=False)
        def forward(self, z, mask=None, **kwargs):
            return super().forward(z, mask=mask)
    return _FastTriMul


def _patch(module_path, outgoing, incoming):
    module = importlib.import_module(module_path)      # where the block references TriMul
    for name in outgoing:
        if hasattr(module, name):
            setattr(module, name, _adapter("outgoing"))
    for name in incoming:
        if hasattr(module, name):
            setattr(module, name, _adapter("incoming"))
    return module


def patch_openfold3():
    """Make OpenFold-3's Pairformer use fast_trimul. Call before building the model."""
    return _patch("openfold3.core.model.latent.base_blocks",
                  ["TriangleMultiplicationOutgoing", "FusedTriangleMultiplicationOutgoing"],
                  ["TriangleMultiplicationIncoming", "FusedTriangleMultiplicationIncoming"])


def patch_openfold():
    """Make OpenFold (AF2)'s Evoformer use fast_trimul. Call before building the model."""
    return _patch("openfold.model.evoformer",
                  ["TriangleMultiplicationOutgoing", "FusedTriangleMultiplicationOutgoing"],
                  ["TriangleMultiplicationIncoming", "FusedTriangleMultiplicationIncoming"])


def patch_boltz():
    """Make Boltz-1's Pairformer use fast_trimul. Call before building the model."""
    return _patch("boltz.model.layers.pairformer",
                  ["TriangleMultiplicationOutgoing"], ["TriangleMultiplicationIncoming"])


def patch_protenix():
    """Make Protenix's Pairformer use fast_trimul. Call before building the model."""
    return _patch("protenix.model.modules.pairformer",
                  ["TriangleMultiplicationOutgoing"], ["TriangleMultiplicationIncoming"])
