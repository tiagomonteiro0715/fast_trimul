# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Per-library weight maps, each registered with one decorator.

Two shapes of mapping:
  * rename only (OpenFold / Protenix / OpenFold-3 non-fused) -- separate a/b
    projections, 1:1 key rename.
  * rename + split (Boltz / Chai / AF3 / OpenFold-3 fused) -- a+b live in one
    wide linear (d -> 2d); split it into proj_a/proj_b and gate_a/gate_b.

Each function returns a dict keyed by OUR parameter names. Missing biases (from
bias-free libraries) are zeroed by the module's `_load_remapped`. Cost: O(P) over
the parameters, one-time at load.
"""

from ..core.registry import weights_for

# OpenFold / AF2 names -> ours (Protenix shares these names)
_OPENFOLD_TO_FAST = {
    "layer_norm_in": "norm_in", "layer_norm_out": "norm_out",
    "linear_a_p": "proj_a", "linear_a_g": "gate_a",
    "linear_b_p": "proj_b", "linear_b_g": "gate_b",
    "linear_g": "proj_g", "linear_z": "proj_out",
}


@weights_for("openfold")
def _openfold(sd: dict) -> dict:
    out = {}
    for key, val in sd.items():
        head, dot, tail = key.partition(".")
        out[_OPENFOLD_TO_FAST.get(head, head) + dot + tail] = val
    return out


# Protenix is OpenFold-derived (same names, bias-free linears): reuse the rename.
weights_for("protenix")(_openfold)


def _split(sd: dict, name: str):
    """Split a fused (2d, ...) tensor into its a|b halves."""
    tensor = sd[name]
    half = tensor.shape[0] // 2
    return tensor[:half], tensor[half:]


@weights_for("boltz")
def _boltz(sd: dict) -> dict:
    """Boltz-1 / Chai / AF3-style: fused p_in/g_in, bias-free, no residual."""
    proj_a, proj_b = _split(sd, "p_in.weight")
    gate_a, gate_b = _split(sd, "g_in.weight")
    return {
        "norm_in.weight": sd["norm_in.weight"], "norm_in.bias": sd["norm_in.bias"],
        "norm_out.weight": sd["norm_out.weight"], "norm_out.bias": sd["norm_out.bias"],
        "proj_a.weight": proj_a, "proj_b.weight": proj_b,
        "gate_a.weight": gate_a, "gate_b.weight": gate_b,
        "proj_out.weight": sd["p_out.weight"],   # p_out -> output projection
        "proj_g.weight": sd["g_out.weight"],     # g_out -> output gate
    }


# Chai ships no importable layer; its operator is the AF3 reference = Boltz shape.
weights_for("chai")(_boltz)


@weights_for("openfold3")
def _openfold3(sd: dict) -> dict:
    """OpenFold-3: TriangleMultiplicationOutgoing is OpenFold-style (rename);
    FusedTriangleMultiplicationOutgoing fuses into linear_ab_p/linear_ab_g (split)."""
    if "linear_ab_p.weight" not in sd:                 # separate-projection variant
        return _openfold(sd)
    proj_a, proj_b = _split(sd, "linear_ab_p.weight")
    gate_a, gate_b = _split(sd, "linear_ab_g.weight")
    out = {
        "norm_in.weight": sd["layer_norm_in.weight"], "norm_in.bias": sd["layer_norm_in.bias"],
        "norm_out.weight": sd["layer_norm_out.weight"], "norm_out.bias": sd["layer_norm_out.bias"],
        "proj_a.weight": proj_a, "proj_b.weight": proj_b,
        "gate_a.weight": gate_a, "gate_b.weight": gate_b,
        "proj_out.weight": sd["linear_z.weight"],   # linear_z -> output projection
        "proj_g.weight": sd["linear_g.weight"],     # linear_g -> output gate
    }
    for src, (dst_a, dst_b) in (("linear_ab_p.bias", ("proj_a.bias", "proj_b.bias")),
                                ("linear_ab_g.bias", ("gate_a.bias", "gate_b.bias"))):
        if src in sd:
            out[dst_a], out[dst_b] = _split(sd, src)
    for src, dst in (("linear_z.bias", "proj_out.bias"), ("linear_g.bias", "proj_g.bias")):
        if src in sd:
            out[dst] = sd[src]
    return out
