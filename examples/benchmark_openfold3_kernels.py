# Kernel + long-sequence benchmark for fast_trimul vs OpenFold-3 -- Lightning AI (A100).
#
# Answers the reviewer's benchmark questions in one run:
#   * speed / VRAM of each kernel path -- fast_trimul's CuTe kernel, its
#     cuEquivariance fallback, and its pure-torch fallback -- next to the
#     OpenFold-3 baseline (which itself uses triton/cuequivariance) and compile;
#   * long queries (N up to 4096, i.e. > 2000 residues), reporting BOTH median
#     latency and peak VRAM, with OOM handled per-size;
#   * a correctness probe at a non-multiple-of-8 N (2001) that exercises the new
#     pad-to-8 path so you can see the fast kernel still matches the reference.
#
# Install once in the Lightning terminal, then run:
#   uv pip install --system openfold3 "fast_trimul>=2.1.2" "cuda-python<13"
#   # optional (adds the cueq row): uv pip install --system cuequivariance-torch
#   python benchmark_openfold3_kernels.py

import gc
import statistics

import torch

assert torch.cuda.is_available(), "needs a CUDA GPU"
torch.set_float32_matmul_precision("high")
DEV = "cuda"
B, D_Z, D_C = 1, 128, 128

# sizes: a short ramp, then the long queries the reviewer asked about (> 2000).
SWEEP = [256, 512, 1024, 2048, 3072, 4096]
ODD_N = 2001                      # not a multiple of 8 -> exercises the pad path


def bench(fn, iters=30, warmup=5):
    """Median ms/call + peak VRAM (GB) for one callable, CUDA-event timed."""
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


def build_openfold3():
    """OpenFold-3's own outgoing TriMul (its triton/cuequivariance kernel), or None."""
    try:
        from openfold3.core.model.layers.triangular_multiplicative_update import (
            TriangleMultiplicationOutgoing)
        return TriangleMultiplicationOutgoing(c_z=D_Z, c_hidden=D_C).to(DEV).eval()
    except Exception as err:                      # keep the fast_trimul rows even if OF3 differs
        print(f"  (OpenFold-3 baseline unavailable: {type(err).__name__}: {err})")
        return None


def fast_module(backend):
    from fast_trimul import FastTriangleMultiplication
    return FastTriangleMultiplication(
        d_z=D_Z, d_c=D_C, mode="outgoing", residual=False, backend=backend).to(DEV).eval()


def correctness_check():
    """Copy OF-3 weights into each fast backend and report max/mean abs diff,
    including the non-multiple-of-8 size that goes through the pad path."""
    of3 = build_openfold3()
    if of3 is None:
        print("skip correctness check (no OpenFold-3 reference)\n"); return
    for p in of3.parameters():                    # non-trivial weights (see checks.py note)
        p.data.normal_(0, 0.02)
    print("Correctness vs OpenFold-3 (weights copied across):")
    for n in (128, ODD_N):
        z = torch.randn(B, n, n, D_Z, device=DEV)
        mask = torch.ones(B, n, n, device=DEV)
        with torch.no_grad():
            want = of3(z, mask=mask).float()
        for be in ("cuda", "cueq", "torch"):
            fast = fast_module(be)
            try:
                fast.load_weights(of3.state_dict(), "openfold3")
                with torch.no_grad():
                    got = fast(z, mask=mask).float()
                d = (want - got).abs()
                print(f"  N={n:>4} {be:<5}: max {d.max():.3e}  mean {d.mean():.3e}"
                      f"  {'(pad path)' if n % 8 else ''}")
            except Exception as err:
                print(f"  N={n:>4} {be:<5}: skipped ({type(err).__name__}: {err})")
            del fast
        del z, mask; gc.collect(); torch.cuda.empty_cache()
    print()


def sweep():
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"{'N':>5} | {'metric':<10}"
          f"{'OF3 base':>12}{'OF3+comp':>12}{'fast cuda':>12}{'fast cueq':>12}{'fast torch':>12}")
    print("-" * 77)
    for n in SWEEP:
        z = torch.randn(B, n, n, D_Z, device=DEV)
        mask = torch.ones(B, n, n, device=DEV)
        of3 = build_openfold3()
        cols = {}
        cands = {
            "OF3 base":   (lambda m=of3: m(z, mask=mask)) if of3 is not None else None,
            "OF3+comp":   None,
            "fast cuda":  lambda m=fast_module("cuda"):  m(z, mask),
            "fast cueq":  lambda m=fast_module("cueq"):  m(z, mask),
            "fast torch": lambda m=fast_module("torch"): m(z, mask),
        }
        if of3 is not None:
            comp = torch.compile(of3, mode="reduce-overhead")
            cands["OF3+comp"] = lambda: comp(z, mask=mask)
        for name, fn in cands.items():
            if fn is None:
                cols[name] = (float("nan"), float("nan")); continue
            try:
                with torch.no_grad():
                    cols[name] = bench(fn)
            except torch.cuda.OutOfMemoryError:
                cols[name] = (float("inf"), float("inf"))
                torch.cuda.empty_cache()
            except Exception:
                cols[name] = (float("nan"), float("nan"))
        order = ["OF3 base", "OF3+comp", "fast cuda", "fast cueq", "fast torch"]
        print(f"{n:>5} | {'ms/call':<10}" + "".join(f"{cols[k][0]:>12.2f}" for k in order))
        print(f"{'':>5} | {'peak GB':<10}" + "".join(f"{cols[k][1]:>12.2f}" for k in order))
        del z, mask, of3, cands, cols; gc.collect()
        torch.compiler.reset(); torch.cuda.empty_cache()
    print("\nnan = unavailable/mismatch, inf = OOM. 'fast cueq' appears only when "
          "cuequivariance-torch is installed; otherwise the dispatcher skips it.")


if __name__ == "__main__":
    correctness_check()
    sweep()
