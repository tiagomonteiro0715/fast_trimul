# Benchmark OpenFold-3's OWN triangle-multiply kernels (torch / triton / cuequivariance).
#
# Answers the reviewer's Q1 ("benchmark speeds/memory for the triton or cuequivariance
# kernel") using OF-3's native kernels, selected through the real forward kwargs found
# in openfold3/core/model/layers/triangular_multiplicative_update.py:
#   torch  : m(z, mask=mask)
#   triton : m(z, mask=mask, inplace_safe=True, use_triton_triangle_kernels=True)   # in-place
#   cueq   : m(z, mask=mask, use_cueq_triangle_kernels=True)                         # channel %32==0
# fast_trimul's fast+graph is shown alongside for context if installed. Also sweeps to
# 4096 for Q3 (long queries), reporting median ms/call and peak VRAM.
#
# Install on Lightning, then run:
#   uv pip install --system "numpy<2" openfold3 cuequivariance-ops-torch-cu12 cuequivariance-torch "cuda-python<13"
#   # optional context column:  uv pip install --system fast_trimul
#   python benchmark_openfold3_native_kernels.py

import gc
import statistics

import torch

assert torch.cuda.is_available(), "needs a CUDA GPU"
torch.set_float32_matmul_precision("high")
DEV = "cuda"
B, C_Z, C_HIDDEN = 1, 128, 128      # channel 128 -> % 32 == 0, so cueq is eligible

SWEEP = [8, 32, 128, 256, 512, 1024, 2048, 3072, 4096]

from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing, TRITON_AVAILABLE)
try:
    from openfold3.core.kernels.cueq_utils import is_cuequivariance_available
    CUEQ_AVAILABLE = is_cuequivariance_available()
except Exception:
    CUEQ_AVAILABLE = False


def bench(fn, iters=30, warmup=5):
    """Median ms/call + peak VRAM (GB)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEV)
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return statistics.median(ts), torch.cuda.max_memory_allocated(DEV) / 1024 ** 3


def build():
    return TriangleMultiplicationOutgoing(c_z=C_Z, c_hidden=C_HIDDEN).to(DEV).eval()


def fast_available():
    try:
        import fast_trimul  # noqa: F401
        return True
    except Exception:
        return False


# (label, forward-kwargs, available, in_place) -- in_place kernels overwrite z.
CONFIGS = [
    ("OF3 torch",  {}, True, False),
    ("OF3 triton", {"inplace_safe": True, "use_triton_triangle_kernels": True},
     TRITON_AVAILABLE, True),
    ("OF3 cueq",   {"use_cueq_triangle_kernels": True},
     CUEQ_AVAILABLE and C_Z % 32 == 0, False),
]


def main():
    print("== availability ==")
    print(f"  triton kernels        : {TRITON_AVAILABLE}")
    print(f"  cuequivariance kernels: {CUEQ_AVAILABLE} (channel {C_Z} % 32 == {C_Z % 32})")
    has_fast = fast_available()
    print(f"  fast_trimul (context) : {has_fast}\n")

    cols = [label for label, _, ok, _ in CONFIGS if ok]
    if has_fast:
        cols.append("fast+graph")

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"{'N':>5} | {'metric':<9}" + "".join(f"{c:>13}" for c in cols))
    print("-" * (17 + 13 * len(cols)))

    for n in SWEEP:
        z = torch.randn(B, n, n, C_Z, device=DEV)
        mask = torch.ones(B, n, n, device=DEV)
        row = {}

        for label, kwargs, ok, in_place in CONFIGS:
            if not ok:
                continue
            try:
                m = build()
                # in-place kernels overwrite z; reuse one throwaway buffer so the
                # per-call clone cost isn't charged to the kernel (work is identical).
                buf = z.clone() if in_place else z
                with torch.no_grad():
                    row[label] = bench(lambda: m(buf, mask=mask, **kwargs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); row[label] = (float("inf"), float("inf"))
            except Exception as err:
                if n == SWEEP[0]:
                    print(f"  ({label} failed: {type(err).__name__}: {err})")
                row[label] = (float("nan"), float("nan"))
            finally:
                m = buf = None; gc.collect(); torch.cuda.empty_cache()

        if has_fast:
            try:
                from fast_trimul import FastTriangleMultiplication
                m = FastTriangleMultiplication(
                    d_z=C_Z, d_c=C_HIDDEN, mode="outgoing", residual=False,
                    backend="cuda").to(DEV).eval()
                m.graphed(z, mask)
                with torch.no_grad():
                    row["fast+graph"] = bench(lambda: m(z, mask))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); row["fast+graph"] = (float("inf"), float("inf"))
            except Exception:
                row["fast+graph"] = (float("nan"), float("nan"))
            finally:
                m = None; gc.collect(); torch.cuda.empty_cache()

        print(f"{n:>5} | {'ms/call':<9}" + "".join(f"{row.get(c, (float('nan'),))[0]:>13.2f}" for c in cols))
        print(f"{'':>5} | {'peak GB':<9}" + "".join(f"{row.get(c, (0, float('nan')))[1]:>13.2f}" for c in cols))
        del z, mask, row; gc.collect(); torch.cuda.empty_cache()

    print("\nnan = kernel unavailable/failed, inf = OOM. 'OF3 triton' is in-place "
          "(inplace_safe); its latency is the fair inference number for that path.")


if __name__ == "__main__":
    main()
