# OpenFold-3 on fast_trimul -- Lightning AI example (GPU Studio; verified on A100).
# One-line integration with patch_openfold3().
#
# Install once in the Lightning terminal:
#   uv pip install --system openfold3 "fast_trimul>=2.1.2" "cuda-python<13"
# Then run:
#   python example_trimul_openfold3.py

import torch
import fast_trimul

fast_trimul.patch_openfold3()      # <-- OpenFold-3 now runs on the fast kernel; call BEFORE building the model

from openfold3.core.model.latent.pairformer import PairFormerStack

model = PairFormerStack(c_s=384, c_z=128, no_blocks=8, c_hidden_pair_bias=32, no_heads_pair_bias=4,
                        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
                        transition_type="swiglu", transition_n=4, pair_dropout=0.25,
                        fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9).cuda().eval()
N = 256
s = torch.randn(1, N, 384, device="cuda")
z = torch.randn(1, N, N, 128, device="cuda")
with torch.no_grad():
    _, out_z = model(s, z, torch.ones(1, N, device="cuda"), torch.ones(1, N, N, device="cuda"))
print("OpenFold-3 running on fast_trimul ->", tuple(out_z.shape))
