# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The input guard: normalize() and NormalizedInput (CPU-only)."""

import torch

from fast_trimul.core.context import BackendCapabilities, NormalizedInput, normalize


def _caps(dtypes, align=1):
    return BackendCapabilities(dtypes=dtypes, min_align=align)


def test_rejects_non_4d():
    assert normalize(torch.randn(4, 4), None, _caps({torch.float16})) is None


def test_rejects_bad_alignment():
    caps = _caps({torch.float32}, align=8)
    assert normalize(torch.randn(1, 12, 12, 4), None, caps) is None


def test_accepts_aligned():
    out = normalize(torch.randn(1, 8, 8, 4), None, _caps({torch.float32}, 8))
    assert isinstance(out, NormalizedInput)
    assert out.shape == (1, 8, 8, 4)


def test_min_align_one_accepts_any_n():
    assert normalize(torch.randn(1, 3, 3, 4), None, _caps({torch.float32})) is not None


def test_casts_fp32_to_fp16():
    out = normalize(torch.randn(1, 8, 8, 4), None, _caps({torch.float16}, 8))
    assert out.tensor.dtype == torch.float16


def test_keeps_supported_dtype():
    out = normalize(torch.randn(1, 8, 8, 4), None, _caps({torch.float32}, 8))
    assert out.tensor.dtype == torch.float32


def test_casts_bf16_to_fp16_when_only_fp16():
    z = torch.randn(1, 8, 8, 4).to(torch.bfloat16)
    out = normalize(z, None, _caps({torch.float16}, 8))
    assert out.tensor.dtype == torch.float16


def test_casts_to_fp32_when_fp16_unavailable():
    z = torch.randn(1, 8, 8, 4).to(torch.float16)
    out = normalize(z, None, _caps({torch.float32}, 8))
    assert out.tensor.dtype == torch.float32


def test_makes_input_contiguous():
    z = torch.randn(1, 8, 8, 4).transpose(1, 2)
    assert not z.is_contiguous()
    out = normalize(z, None, _caps({torch.float32}, 8))
    assert out.tensor.is_contiguous()


def test_makes_mask_contiguous():
    z = torch.randn(1, 8, 8, 4)
    mask = torch.ones(1, 8, 8).transpose(1, 2)
    out = normalize(z, mask, _caps({torch.float32}, 8))
    assert out.mask is not None and out.mask.is_contiguous()


def test_strides_are_int_tuple():
    out = normalize(torch.randn(1, 8, 8, 4), None, _caps({torch.float32}, 8))
    assert len(out.strides) == 4
    assert all(isinstance(s, int) for s in out.strides)
