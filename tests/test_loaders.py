# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Per-library weight remaps (@weights_for) -- pure dict/tensor ops (CPU-only)."""

import torch

from fast_trimul.core.registry import get_loader
from fast_trimul.integrations.loaders import _OPENFOLD_TO_FAST


def test_openfold_rename():
    sd = {"layer_norm_in.weight": torch.zeros(2),
          "linear_a_p.weight": torch.zeros(2, 2),
          "linear_z.weight": torch.zeros(2, 2)}
    out = get_loader("openfold")(sd)
    assert "norm_in.weight" in out
    assert "proj_a.weight" in out
    assert "proj_out.weight" in out


def test_protenix_uses_openfold_map():
    out = get_loader("protenix")({"linear_a_g.weight": torch.zeros(2, 2)})
    assert "gate_a.weight" in out


def test_openfold_map_keys():
    assert _OPENFOLD_TO_FAST["linear_a_p"] == "proj_a"
    assert _OPENFOLD_TO_FAST["layer_norm_out"] == "norm_out"


def test_boltz_splits_fused_projections():
    d = 4
    p_in = torch.arange(2 * d * d).reshape(2 * d, d).float()
    sd = {
        "norm_in.weight": torch.zeros(d), "norm_in.bias": torch.zeros(d),
        "norm_out.weight": torch.zeros(d), "norm_out.bias": torch.zeros(d),
        "p_in.weight": p_in, "g_in.weight": torch.zeros(2 * d, d),
        "p_out.weight": torch.zeros(d, d), "g_out.weight": torch.zeros(d, d),
    }
    out = get_loader("boltz")(sd)
    assert out["proj_a.weight"].shape == (d, d)
    assert out["proj_b.weight"].shape == (d, d)
    assert torch.equal(out["proj_a.weight"], p_in[:d])
    assert torch.equal(out["proj_b.weight"], p_in[d:])
    assert "proj_out.weight" in out and "proj_g.weight" in out


def test_chai_is_boltz_loader():
    assert get_loader("chai") is get_loader("boltz")


def test_openfold3_nonfused_falls_back_to_rename():
    out = get_loader("openfold3")({"linear_a_p.weight": torch.zeros(2, 2)})
    assert "proj_a.weight" in out


def test_openfold3_fused_splits():
    d = 4
    ab = torch.arange(2 * d * d).reshape(2 * d, d).float()
    sd = {
        "layer_norm_in.weight": torch.zeros(d), "layer_norm_in.bias": torch.zeros(d),
        "layer_norm_out.weight": torch.zeros(d), "layer_norm_out.bias": torch.zeros(d),
        "linear_ab_p.weight": ab, "linear_ab_g.weight": torch.zeros(2 * d, d),
        "linear_z.weight": torch.zeros(d, d), "linear_g.weight": torch.zeros(d, d),
    }
    out = get_loader("openfold3")(sd)
    assert out["proj_a.weight"].shape == (d, d)
    assert torch.equal(out["proj_a.weight"], ab[:d])
    assert torch.equal(out["proj_b.weight"], ab[d:])


def test_openfold3_fused_splits_bias():
    d = 4
    sd = {
        "layer_norm_in.weight": torch.zeros(d), "layer_norm_in.bias": torch.zeros(d),
        "layer_norm_out.weight": torch.zeros(d), "layer_norm_out.bias": torch.zeros(d),
        "linear_ab_p.weight": torch.zeros(2 * d, d),
        "linear_ab_g.weight": torch.zeros(2 * d, d),
        "linear_ab_p.bias": torch.arange(2 * d).float(),
        "linear_z.weight": torch.zeros(d, d), "linear_g.weight": torch.zeros(d, d),
    }
    out = get_loader("openfold3")(sd)
    assert out["proj_a.bias"].shape == (d,)
    assert out["proj_b.bias"].shape == (d,)
