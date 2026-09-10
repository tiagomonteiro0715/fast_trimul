# Prediction-diff check: fast_trimul vs OpenFold-3 baseline, end to end -- Lightning AI (A100).
#
# Answers "if you run the inference integration, how much do the predictions
# differ from the baseline?" It runs the SAME OpenFold-3 PairFormerStack twice on
# the same input and the same weights -- once with OpenFold-3's own kernels, once
# with fast_trimul patched in -- and reports the difference in the output pair
# representation (max / mean abs, and relative). Every weight is held identical
# except the Triangle-Multiplication kernel, so the diff you see is purely the
# kernel swap propagated through all 8 blocks.
#
# Install once in the Lightning terminal, then run:
#   uv pip install --system openfold3 "fast_trimul>=2.1.2" "cuda-python<13"
#   python integration_diff_openfold3.py

import torch

assert torch.cuda.is_available(), "needs a CUDA GPU"
DEV = "cuda"


def make_stack():
    from openfold3.core.model.latent.pairformer import PairFormerStack
    return PairFormerStack(
        c_s=384, c_z=128, no_blocks=8, c_hidden_pair_bias=32, no_heads_pair_bias=4,
        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        transition_type="swiglu", transition_n=4, pair_dropout=0.25,
        fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9).to(DEV).eval()


def main(n=256):
    torch.manual_seed(0)

    # 1) baseline stack (OpenFold-3's own kernels), with randomized trimul weights
    #    so the check is non-trivial (untrained output projections are ~zero).
    baseline = make_stack()
    for name, p in baseline.named_parameters():
        if "tri" in name.lower() and "mul" in name.lower():
            p.data.normal_(0, 0.02)

    # 2) patched stack, then force every weight identical to the baseline: all
    #    non-trimul weights by name, and each trimul kernel via load_weights.
    import fast_trimul
    fast_trimul.patch_openfold3()
    from fast_trimul import FastTriangleMultiplication
    patched = make_stack()
    patched.load_state_dict(baseline.state_dict(), strict=False)   # non-trimul weights
    base_mods = dict(baseline.named_modules())
    n_swapped = 0
    for mod_name, mod in patched.named_modules():
        if isinstance(mod, FastTriangleMultiplication):
            mod.load_weights(base_mods[mod_name].state_dict(), "openfold3")
            n_swapped += 1
    print(f"swapped {n_swapped} Triangle-Multiplication modules to fast_trimul")

    # 3) identical input, compare the two output pair representations.
    s = torch.randn(1, n, 384, device=DEV)
    z = torch.randn(1, n, n, 128, device=DEV)
    smask = torch.ones(1, n, device=DEV)
    zmask = torch.ones(1, n, n, device=DEV)
    with torch.no_grad():
        _, z0 = baseline(s, z, smask, zmask)
        _, z1 = patched(s, z, smask, zmask)
    z0, z1 = z0.float(), z1.float()

    diff = (z0 - z1).abs()
    rel = diff.max() / z0.abs().max().clamp_min(1e-6)
    print(f"\nN={n}, pair repr {tuple(z0.shape)} after 8 Pairformer blocks:")
    print(f"  max abs diff : {diff.max().item():.3e}")
    print(f"  mean abs diff: {diff.mean().item():.3e}")
    print(f"  max rel diff : {rel.item():.3e}")
    print(f"  baseline range [{z0.min():.3g}, {z0.max():.3g}]")
    ok = torch.allclose(z0, z1, atol=1e-2, rtol=1e-2)
    print(f"  within fp16 tol (atol=rtol=1e-2): {'PASS' if ok else 'see numbers above'}")


if __name__ == "__main__":
    main()
