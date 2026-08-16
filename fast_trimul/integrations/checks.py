# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""One-line correctness: `verify(source, reference)`.

Builds a fast module, loads the reference module's weights through the `source`
remap, runs both on a random probe, and returns whether they match within fp16
tolerance. It survives library API changes because YOU pass the reference module
-- only the weight-name remap (selected by `source`) is library-specific, and
that lives in one registered function.

Note: the reference must have non-trivial weights (many stacks zero-init their
output projection, so a fresh untrained module outputs 0 and would match
trivially). Randomize the reference's weights first if you want a real test.
"""

import torch

from ..ops.triangle import FastTriangleMultiplication


def verify(source: str, reference, n: int = 128, d_z: int = 128, d_c: int = 128,
           mode: str = "outgoing", atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """True if fast_trimul (loaded with `reference`'s weights) matches `reference`.

    Example:  fast_trimul.verify("openfold", my_openfold_trimul)
    """
    device = next(reference.parameters()).device
    fast = FastTriangleMultiplication(
        d_z=d_z, d_c=d_c, mode=mode, residual=False).to(device).eval()
    fast.load_weights(reference.state_dict(), source)

    z = torch.randn(1, n, n, d_z, device=device)
    mask = torch.ones(1, n, n, device=device)
    with torch.no_grad():
        want = reference(z, mask=mask).float()
        got = fast(z, mask=mask).float()
    return bool(torch.allclose(want, got, atol=atol, rtol=rtol))
