# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Rigorous benchmark: fast_trimul vs torch vs torch.compile.

Ships with the package so you can compare correctly after install:

    from fast_trimul.benchmark import run_benchmark
    run_benchmark()

or from a shell:  python -m fast_trimul.benchmark

Ports the evaluation style used for the individual kernels:
  * measured machine ceilings (memory bandwidth, fp16 tensor-core peak, launch floor),
  * a per-iteration CUDA-event MEDIAN timer (median / min / p95 / CV, not a mean),
  * roofline placement (% of measured peak), effective GB/s, achieved TFLOP/s,
  * a head-to-head table + a size sweep.
"""

import copy
import statistics

import torch

from .nn import FastTriangleMultiplication
from ._kernels import TriangleMultiplicativeUpdate   # torch reference (same op)


# --------------------------------------------------------------------------- #
# per-iteration CUDA-event timer -> median/min/p95/CV (ms)
# --------------------------------------------------------------------------- #
def bench_ev(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()                       # completed timestamp, not queued
        ts.append(s.elapsed_time(e))
    ts.sort()
    mean = statistics.mean(ts)
    return {
        "median": ts[len(ts) // 2],
        "min": ts[0],
        "p95": ts[min(len(ts) - 1, int(0.95 * len(ts)))],
        "cv": statistics.pstdev(ts) / mean * 100.0,    # coefficient of variation, %
    }


# --------------------------------------------------------------------------- #
# machine ceilings (measured, not from a datasheet)
# --------------------------------------------------------------------------- #
def bandwidth_peak_gbps(mb=256):
    n = mb * 1024 * 1024 // 4
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    ms = bench_ev(lambda: y.copy_(x))["median"]
    return 2 * n * 4 / (ms / 1e3) / 1e9                # read + write


def fp16_matmul_peak_tflops(m=8192):
    a = torch.randn(m, m, device="cuda", dtype=torch.float16)
    b = torch.randn(m, m, device="cuda", dtype=torch.float16)
    ms = bench_ev(lambda: torch.mm(a, b))["median"]
    return 2 * m ** 3 / (ms / 1e3) / 1e12


def launch_floor_us():
    x = torch.zeros(1, device="cuda")
    return bench_ev(lambda: x.add_(1.0))["median"] * 1e3


# --------------------------------------------------------------------------- #
# TriMul work counters (matmuls dominate; elementwise/norm are negligible)
# --------------------------------------------------------------------------- #
def trimul_flops(B, N, d_z, d_c):
    M = B * N * N
    gemm = (4 * 2 * M * d_z * d_c        # proj_a, gate_a, proj_b, gate_b
            + 2 * M * d_z * d_z          # proj_g
            + 2 * M * d_c * d_z)         # proj_out
    contraction = 2 * B * d_c * N ** 3   # L = B*d_c batched N x N x N
    return gemm + contraction


def trimul_min_bytes(B, N, d_z, dtype_bytes):
    return 2 * B * N * N * d_z * dtype_bytes            # read z + write out (ideal)


def _build(N, d_z=128, d_c=128, mode="outgoing"):
    torch.manual_seed(0)
    ref = TriangleMultiplicativeUpdate(d_z, d_c, mode).cuda().eval()      # torch fp32
    fast = FastTriangleMultiplication(d_z, d_c, mode).cuda()
    fast._impl.load_state_dict(ref.state_dict(), strict=False)           # same weights
    z = torch.randn(1, N, N, d_z, device="cuda")
    return ref, fast, z


def run_benchmark(head_size=256, sweep=(8, 16, 32, 64, 128, 192, 256, 512, 1024),
                  d_z=128, d_c=128, mode="outgoing"):
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    torch.set_float32_matmul_precision("high")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    bw = bandwidth_peak_gbps()
    peak = fp16_matmul_peak_tflops()
    floor = launch_floor_us()
    print(f"  measured mem bandwidth peak : {bw:8.0f} GB/s")
    print(f"  measured fp16 matmul peak   : {peak:8.0f} TFLOP/s")
    print(f"  launch-overhead floor       : {floor:8.1f} us\n")

    # ---- head-to-head at one size ----
    # Shows fast_trimul BOTH un-graphed and with a captured CUDA graph, next to
    # the fair fp16 reduce-overhead compile baseline and naive fp32 eager.
    N = head_size
    ref, fast, z = _build(N, d_z, d_c, mode)
    flop = trimul_flops(1, N, d_z, d_c)
    ref_h = copy.deepcopy(ref).half().eval()
    z_h = z.half()
    ref_c16 = torch.compile(ref_h, mode="reduce-overhead")
    ref_c32 = torch.compile(ref)                      # fp32 default (no reduce-overhead)
    compute_floor_ms = flop / (peak * 1e12) * 1e3     # ideal time at measured fp16 peak

    with torch.no_grad():
        err = (fast(z).float() - ref(z)).abs().max().item()   # first call also autotunes
        s_nog = bench_ev(lambda: fast(z))            # fast_trimul, un-graphed
        fast.graphed(z)                              # capture CUDA graph
        s_graph = bench_ev(lambda: fast(z))          # fast_trimul, fp16 + CUDA graph
        s_eager = bench_ev(lambda: ref(z))           # torch eager, fp32 (naive)
        s_c32 = bench_ev(lambda: ref_c32(z))         # torch.compile fp32 (default)
        s_c16 = bench_ev(lambda: ref_c16(z_h))       # torch.compile fp16 reduce-overhead
    stats = [s_nog, s_graph, s_eager, s_c32, s_c16]
    names = ["fast no-graph", "fast +graph", "torch eager", "compile32", "compile16"]
    dbytes = [2, 2, 4, 4, 2]
    eager_med, c16_med = s_eager["median"], s_c16["median"]

    print(f"Head-to-head  N={N}, d_z={d_z}, d_c={d_c}   "
          f"({flop/1e9:.1f} GFLOP/call, fp16 err vs torch = {err:.1e})")
    metrics = [
        ("median (us)",          lambda s, b: s["median"] * 1e3),
        ("min (us)",             lambda s, b: s["min"] * 1e3),
        ("p95 (us)",             lambda s, b: s["p95"] * 1e3),
        ("run-to-run CV (%)",    lambda s, b: s["cv"]),
        ("TFLOP/s",              lambda s, b: flop / (s["median"] / 1e3) / 1e12),
        ("% of fp16 peak",       lambda s, b: flop / (s["median"] / 1e3) / 1e12 / peak * 100),
        ("effective GB/s",       lambda s, b: trimul_min_bytes(1, N, d_z, b) / (s["median"] / 1e3) / 1e9),
        ("x above roofline",     lambda s, b: s["median"] / compute_floor_ms),
        ("x launch floor",       lambda s, b: s["median"] * 1e3 / floor),
        ("speedup vs eager",     lambda s, b: eager_med / s["median"]),
        ("speedup vs compile16", lambda s, b: c16_med / s["median"]),
    ]
    print("  " + f"{'metric':<20}" + "".join(f"{n:>14}" for n in names))
    print("  " + "-" * (20 + 14 * len(names)))
    for label, fn in metrics:
        print("  " + f"{label:<20}" + "".join(f"{fn(s, b):>14.2f}" for s, b in zip(stats, dbytes)))
    del ref, fast, ref_h, ref_c16, ref_c32, z, z_h
    torch.compiler.reset()
    torch.cuda.empty_cache()

    # ---- size sweep (median us/call) ----
    print("\nSize sweep (median us/call). fast_ng = un-graphed, fast_g = fp16+CUDA graph, "
          "compile32 = fp32 default, compile16 = fp16 reduce-overhead:")
    print(f"  {'N':>5}{'fast_ng':>10}{'fast_g':>10}{'eager':>10}{'compile32':>11}"
          f"{'compile16':>11}{'fast TFLOP/s':>14}")
    for N in sweep:
        ref = fast = ref_h = ref_c16 = ref_c32 = z = z_h = None
        try:
            ref, fast, z = _build(N, d_z, d_c, mode)
            ref_h = copy.deepcopy(ref).half().eval()
            z_h = z.half()
            ref_c16 = torch.compile(ref_h, mode="reduce-overhead")
            ref_c32 = torch.compile(ref)
            with torch.no_grad():
                tng = bench_ev(lambda: fast(z))["median"]     # un-graphed
                fast.graphed(z)                               # capture graph
                tg = bench_ev(lambda: fast(z))["median"]      # graphed
                te = bench_ev(lambda: ref(z))["median"]
                tc32 = bench_ev(lambda: ref_c32(z))["median"]
                tc16 = bench_ev(lambda: ref_c16(z_h))["median"]
            tfs = trimul_flops(1, N, d_z, d_c) / (tg / 1e3) / 1e12
            print(f"  {N:>5}{tng*1e3:9.1f}u{tg*1e3:9.1f}u{te*1e3:9.1f}u"
                  f"{tc32*1e3:10.1f}u{tc16*1e3:10.1f}u{tfs:13.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {N:>5}  OOM (fp32 baselines don't fit; try a smaller sweep)")
        finally:
            del ref, fast, ref_h, ref_c16, ref_c32, z, z_h
            torch.compiler.reset()
            torch.cuda.empty_cache()

    print("\nNote: fast_ng shows the cost of NOT graphing (host/launch bound); fast_g adds")
    print("the CUDA graph. compile16 (fp16 reduce-overhead) is the fair fight; compile32")
    print("(fp32 default) and eager (fp32) are references.")


if __name__ == "__main__":
    run_benchmark()
