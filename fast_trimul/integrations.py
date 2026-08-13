# Copyright (c) 2026 Tiago Monteiro. MIT License.
"""Monkeypatch helpers: swap a target library's TriMul for the fast one.

Each `patch_*` replaces the target class with an adapter that matches its
CONSTRUCTOR signature, so the model *builds* with FastTriangleMultiplication.

IMPORTANT — read before trusting these:
  * Loading PRETRAINED weights needs parameter-name REMAPPING: each library names
    its projections/norms differently from this package, so a strict checkpoint
    load will NOT line up out of the box. For inference on pretrained models you
    must supply a remap (or patch, then load with strict=False and a name map).
    Patching BEFORE building + training/fine-tuning avoids this.
  * Module import paths and signatures CHANGE across library versions. Validate
    against your installed version; if an import path is wrong, set the attribute
    yourself: `<their_module>.<TheirClass> = adapter('outgoing')`.
  * The kernels are fp16 and slower than torch.compile(fp16) above small N (see
    README). These patches are for correctness/compat first, not guaranteed speed.
"""

from .nn import FastTriangleMultiplication


def adapter(mode: str):
    """Subclass whose __init__ accepts the common (c_z, c_hidden) target signature."""
    class _Adapter(FastTriangleMultiplication):
        def __init__(self, c_z, c_hidden=None, *args, **kwargs):
            super().__init__(d_z=c_z, d_c=c_hidden if c_hidden is not None else c_z, mode=mode)
    _Adapter.__name__ = f"FastTriangleMultiplication_{mode}"
    return _Adapter


def patch_openfold():
    import openfold.model.triangular_multiplicative_update as m
    m.TriangleMultiplicationOutgoing = adapter("outgoing")
    m.TriangleMultiplicationIncoming = adapter("incoming")


def patch_boltz():
    import boltz.model.layers.triangular_mult as m
    m.TriangleMultiplicationOutgoing = adapter("outgoing")
    if hasattr(m, "TriangleMultiplicationIncoming"):
        m.TriangleMultiplicationIncoming = adapter("incoming")


def patch_protenix():
    import protenix.model.modules.pairformer as m
    m.TriangleMultiplication = adapter("outgoing")


def patch_chai():
    # Chai-1's TriMul module path varies by version — set it explicitly instead:
    #   import <chai module> as m; m.<Class> = fast_trimul.integrations.adapter("outgoing")
    raise NotImplementedError(
        "Chai-1's TriMul path is version-dependent; patch the attribute manually "
        "with fast_trimul.integrations.adapter('outgoing' | 'incoming')."
    )
