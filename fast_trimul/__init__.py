# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""fast_trimul: fused Triangle Multiplicative Update (AlphaFold family) on a
modular, vendor-agnostic backend architecture.

- `FastTriangleMultiplication` -- the drop-in module (auto-selects a backend,
  falls back to pure torch).
- `@accelerate` / `accelerated()` -- explicit, scoped opt-in (no global patching).
- `verify(source, reference)` -- one-line correctness check for a target library.
- `list_backends()` -- what's registered on this machine.
"""

from . import functional
from . import backends          # importing registers the cuda + torch backends
from . import integrations      # importing registers the weight loaders
from .ops.triangle import FastTriangleMultiplication
from .core.decorators import accelerate, accelerated
from .core.registry import list_backends
from .integrations.checks import verify
from .integrations.patch import patch_openfold, patch_openfold3, patch_boltz, patch_protenix
from . import nn                # backward-compatible shim

__all__ = [
    "FastTriangleMultiplication",
    "accelerate",
    "accelerated",
    "verify",
    "list_backends",
    "patch_openfold",
    "patch_openfold3",
    "patch_boltz",
    "patch_protenix",
    "functional",
    "backends",
    "integrations",
    "nn",
]
__version__ = "2.1.0"
