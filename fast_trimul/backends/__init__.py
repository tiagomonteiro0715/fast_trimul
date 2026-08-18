# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Importing this package registers the available backends.

`torch` (universal, runs anywhere torch runs) always registers. `cuda` registers
too -- its module only imports torch-level code, so registration is cheap; the
actual CUTLASS kernels load lazily when a module is constructed.
"""

from . import torch_ref   # noqa: F401  registers "torch"

try:
    from . import cuda_cute   # noqa: F401  registers "cuda"
except Exception:             # pragma: no cover - keep torch fallback usable
    pass
