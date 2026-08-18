# Chai / AF3 on fast_trimul -- Lightning AI quickstart (GPU Studio; A100).
#
# Chai-1 ships no importable Triangle Multiplicative Update layer, so there is no
# patch_chai(). Instead you use fast_trimul's module directly as the AF3-style TriMul,
# and (optionally) load Chai / AF3 fused weights with load_weights(..., source="chai").
#
# Install once in the Lightning terminal:
#   uv pip install --system fast_trimul "cuda-python<13"
# Then run:
#   python quickstart_lightning_chai.py

import torch
from fast_trimul import FastTriangleMultiplication

# AF3 / Chai fuse the a/b projections and add their residual OUTSIDE the block,
# so build with residual=False to match their output exactly.
trimul = FastTriangleMultiplication(d_z=128, d_c=128, mode="outgoing", residual=False).cuda().eval()

# To use real Chai / AF3 weights (the fused p_in/g_in are split for you):
#   trimul.load_weights(chai_state_dict, source="chai")

N = 256
z = torch.randn(1, N, N, 128, device="cuda")     # pair representation (B, N, N, d_z)
mask = torch.ones(1, N, N, device="cuda")

with torch.no_grad():
    out = trimul(z, mask=mask)

print("Chai / AF3 TriMul running on fast_trimul ->", tuple(out.shape))    # (1, 256, 256, 128)
