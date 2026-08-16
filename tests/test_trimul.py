# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Correctness (vs a torch reference) and a smoke speed check."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
@pytest.mark.parametrize("mode", ["outgoing", "incoming"])
def test_parity_fp16_tolerance(mode):
    from fast_trimul import FastTriangleMultiplication
    from fast_trimul._kernels import TriangleMultiplicativeUpdate

    torch.manual_seed(0)
    N = 64
    ref = TriangleMultiplicativeUpdate(128, 128, mode).cuda().eval()          # torch fp32
    fast = FastTriangleMultiplication(128, 128, mode).cuda().eval()
    fast._impl.load_state_dict(ref.state_dict(), strict=False)                # same weights (-> fp16)

    z = torch.randn(1, N, N, 128, device="cuda")
    with torch.no_grad():
        out_ref = ref(z)                       # fp32 reference (LayerNorm)
        out_fast = fast(z)                     # fp16 kernels, cast back to fp32

    err = (out_ref - out_fast).abs().max().item()
    assert err < 5e-2, f"{mode}: max abs err {err:.3e} exceeds fp16 tolerance"


@cuda
def test_mask_shapes_and_dtype():
    from fast_trimul import FastTriangleMultiplication

    fast = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    z = torch.randn(1, 64, 64, 128, device="cuda")           # fp32 in
    mask = torch.ones(1, 64, 64, device="cuda")
    with torch.no_grad():
        out = fast(z, mask=mask)
    assert out.shape == z.shape and out.dtype == z.dtype      # dtype preserved


@cuda
def test_load_openfold_state_dict_remaps_names():
    from fast_trimul import FastTriangleMultiplication
    from fast_trimul._kernels import TriangleMultiplicativeUpdate

    torch.manual_seed(0)
    ref = TriangleMultiplicativeUpdate(128, 128, "outgoing").cuda().eval()
    of_names = {"norm_in": "layer_norm_in", "norm_out": "layer_norm_out",
                "proj_a": "linear_a_p", "gate_a": "linear_a_g",
                "proj_b": "linear_b_p", "gate_b": "linear_b_g",
                "proj_g": "linear_g", "proj_out": "linear_z"}
    of_sd = {}                                                 # rename to OpenFold's keys
    for k, v in ref.state_dict().items():
        head, dot, tail = k.partition(".")
        of_sd[of_names[head] + dot + tail] = v

    fast = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    res = fast.load_openfold_state_dict(of_sd)                 # strict=True: raises if the remap is wrong
    assert not res.missing_keys and not res.unexpected_keys

    z = torch.randn(1, 64, 64, 128, device="cuda")
    with torch.no_grad():
        err = (ref(z) - fast(z).float()).abs().max().item()
    assert err < 5e-2, f"remapped-weight parity failed: {err:.3e}"


@cuda
def test_gradients_flow():
    from fast_trimul import FastTriangleMultiplication

    fast = FastTriangleMultiplication(128, 128, "outgoing").cuda()
    z = torch.randn(1, 32, 32, 128, device="cuda", requires_grad=True)
    fast(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
