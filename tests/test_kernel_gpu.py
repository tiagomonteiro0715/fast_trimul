# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""GPU kernel tests -- auto-skip on CPU; run these on a CUDA machine (e.g. Colab)."""

import pytest
import torch

pytestmark = pytest.mark.gpu   # so `pytest -m gpu` / `pytest -m "not gpu"` selects these

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU")


@requires_cuda
def test_forward_shape_and_dtype():
    from fast_trimul import FastTriangleMultiplication
    m = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    z = torch.randn(1, 64, 64, 128, device="cuda")
    with torch.no_grad():
        out = m(z)
    assert out.shape == z.shape
    assert out.dtype == z.dtype


@requires_cuda
def test_incoming_mode_runs():
    from fast_trimul import FastTriangleMultiplication
    m = FastTriangleMultiplication(128, 128, "incoming").cuda().eval()
    z = torch.randn(1, 64, 64, 128, device="cuda")
    with torch.no_grad():
        out = m(z)
    assert out.shape == z.shape


@requires_cuda
def test_functional_matches_module():
    from fast_trimul import FastTriangleMultiplication, functional
    m = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    z = torch.randn(1, 64, 64, 128, device="cuda")
    mask = torch.ones(1, 64, 64, device="cuda")
    with torch.no_grad():
        a = m(z, mask=mask).float()
        b = functional.triangle_multiplication(z, m._impl, mask=mask).float()
    assert torch.allclose(a, b, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_graphed_replay_matches_eager():
    from fast_trimul import FastTriangleMultiplication
    m = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    z = torch.randn(1, 64, 64, 128, device="cuda")
    mask = torch.ones(1, 64, 64, device="cuda")
    with torch.no_grad():
        eager = m(z, mask=mask).float().clone()
        m.graphed(z, mask)
        replay = m(z, mask=mask).float()
    assert torch.allclose(eager, replay, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_residual_false_shape():
    from fast_trimul import FastTriangleMultiplication
    m = FastTriangleMultiplication(128, 128, "outgoing", residual=False).cuda().eval()
    z = torch.randn(1, 32, 32, 128, device="cuda")
    with torch.no_grad():
        out = m(z)
    assert out.shape == z.shape


@requires_cuda
def test_non_multiple_of_8_uses_torch_fallback():
    from fast_trimul import FastTriangleMultiplication
    m = FastTriangleMultiplication(128, 128, "outgoing").cuda().eval()
    z = torch.randn(1, 30, 30, 128, device="cuda")  # N=30 -> cuda backend declines
    with torch.no_grad():
        out = m(z)
    assert out.shape == z.shape
