# Prove the fast_trimul cuEquivariance fallback actually runs and is correct -- Lightning A100.
#
# The dispatcher would silently fall through to torch if the cueq backend were
# broken, which could hide a bug behind a false PASS. So this script invokes the
# cueq backend DIRECTLY (no torch fallback to mask a failure) and compares it to
# fast_trimul's own torch backend on identical weights. If cueq errors, you see the
# real error; if it runs, you see the numerical difference.
#
# Only claim "the cueq fallback works" once this prints PASS.
#
# Install on Lightning, then run:
#   uv pip install --system -e . cuequivariance-torch cuequivariance-ops-torch-cu12 "cuda-python<13"
#   python examples/verify_cueq_backend.py

import torch

import fast_trimul
from fast_trimul import FastTriangleMultiplication
from fast_trimul.core.registry import get_backend
from fast_trimul.core.context import normalize

assert torch.cuda.is_available(), "needs a CUDA GPU"
DEV = "cuda"

print("registered backends:", fast_trimul.list_backends())
be_cueq = get_backend("cueq")
if be_cueq is None:
    raise SystemExit(
        "cueq backend NOT registered -> cuequivariance_torch isn't importable.\n"
        "Install it (uv pip install --system cuequivariance-torch) and re-run.")
be_torch = get_backend("torch")


def check(mode, residual, n=128, d=128):
    # non-trivial weights (a fresh module's output projection is ~0 and would match
    # trivially); channel d=128 is a multiple of 32, which the cueq kernel requires.
    m = FastTriangleMultiplication(d_z=d, d_c=d, mode=mode, residual=residual).to(DEV).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.02)
    z = torch.randn(1, n, n, d, device=DEV)
    mask = torch.ones(1, n, n, device=DEV)

    with torch.no_grad():
        # DIRECT cueq call -- raises loudly if the kernel/layout is wrong.
        got = be_cueq.execute(normalize(z, mask, be_cueq.caps), m._impl).float()
        # fast_trimul's own torch path, same weights = the reference.
        ref = be_torch.execute(normalize(z, mask, be_torch.caps), m._impl).float()

    diff = (ref - got).abs()
    ok = torch.allclose(ref, got, atol=1e-2, rtol=1e-2)
    print(f"  {mode:<8} residual={str(residual):<5}: "
          f"max {diff.max():.3e}  mean {diff.mean():.3e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


print("\ncueq backend vs torch backend (identical weights):")
results = []
for mode in ("outgoing", "incoming"):
    for residual in (False, True):
        try:
            results.append(check(mode, residual))
        except Exception as err:
            print(f"  {mode:<8} residual={str(residual):<5}: cueq RAISED -> "
                  f"{type(err).__name__}: {err}")
            results.append(False)

print()
if all(results):
    print("ALL PASS -- the cuEquivariance fallback runs and matches the torch reference.")
    print("You can now say the cueq backend is validated (not just wired).")
else:
    print("Some cases failed -- do NOT claim the cueq fallback works yet; see errors above.")
