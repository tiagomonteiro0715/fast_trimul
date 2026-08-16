# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Library integrations: importing this registers the weight loaders and exposes
the one-line `verify` correctness check."""

from . import loaders          # noqa: F401  registers @weights_for for the 5 stacks
from .checks import verify
from .patch import patch_openfold, patch_openfold3, patch_boltz, patch_protenix

__all__ = ["verify", "patch_openfold", "patch_openfold3", "patch_boltz", "patch_protenix"]
