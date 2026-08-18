# Protenix on fast_trimul -- Lightning AI quickstart (GPU Studio; A100).
# One line -- patch_protenix() -- swaps Protenix's Triangle Multiplicative Update for
# the fused fast_trimul kernel, then you build and run Protenix exactly as usual.
#
# Install once in the Lightning terminal:
#   uv pip install --system protenix cuequivariance-torch cuequivariance-ops-torch-cu12 fast_trimul "cuda-python<13"
# Then run:
#   python quickstart_lightning_protenix.py

import torch
import fast_trimul

fast_trimul.patch_protenix()       # <-- that's it. Call BEFORE building the model.

from protenix.model.modules.pairformer import PairformerStack

model = PairformerStack(n_blocks=8, n_heads=16, c_z=128, c_s=384).cuda().eval()

N = 256
s = torch.randn(1, N, 384, device="cuda")
z = torch.randn(1, N, N, 128, device="cuda")
pair_mask = torch.ones(1, N, N, device="cuda")

with torch.no_grad():
    _, out_z = model(s, z, pair_mask, triangle_multiplicative="torch")

print("Protenix running on fast_trimul ->", tuple(out_z.shape))    # (1, 256, 256, 128)
