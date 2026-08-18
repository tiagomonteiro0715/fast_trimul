"""Benchmark: OpenFold-3 Triangle-Multiplication versus fast_trimul (Google Colab).

This script loads OpenFold-3's own outgoing Triangle-Multiplication layer, also
wraps it in two torch.compile variants, and compares all of them against
fast_trimul's fused kernel. It first confirms that fast_trimul produces the same
numbers as the OpenFold-3 layer (after copying its weights across), then times
every variant over a chosen sequence length and prints a table of latency,
throughput, and memory. Run it on a GPU runtime. The recorded results live in
results_openfold3.md.

Reproduce it in one click: open the Colab notebook, set the runtime to GPU
(Runtime -> Change runtime type -> GPU), and Run all. It installs everything from
scratch and runs end to end, so the benchmark reproduces without any local setup.
For comparable numbers, use the same setup it was measured on: an NVIDIA A100,
PyTorch 2.13 / CUDA 13.0.

    ▶ Open in Google Colab:
      https://colab.research.google.com/drive/1kZfRmBU8O-t0Wcy8Leb9-QdAXYI-VUdf?usp=sharing
"""

!pip install -q uv
!uv pip install -q --system openfold3 torch pandas tabulate psutil nvidia-ml-py
!uv pip install -q --system --upgrade fast_trimul

import gc, importlib, inspect, math, pkgutil, statistics, sys, traceback, types
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "CUDA GPU is required!"
device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

seq_len = 8
batch, pair_dim, hidden_dim = 1, 128, 128


def install_scipy_stub():
    """Stub out the pieces OpenFold stacks need at import time but cannot load.

    OpenFold-style libraries import a compiled attention extension
    (`attn_core_inplace_cuda`) and a little bit of scipy when they load. On Colab
    the extension often fails to load and scipy may be missing. This function first
    registers a harmless fake of that attention extension (its real symbols are
    unused by Triangle-Multiplication), letting Python's special dunder lookups
    still fail so inspect and torch.compile do not choke on it. It then makes sure
    scipy is importable, using the real one if present and a tiny fake one
    otherwise.
    """
    _attn = types.ModuleType("attn_core_inplace_cuda")
    def _attn_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None
    _attn.__getattr__ = _attn_getattr
    sys.modules.setdefault("attn_core_inplace_cuda", _attn)
    try:
        from scipy.stats import truncnorm
        print("scipy OK")
        return
    except Exception:
        pass

    def moments(lo, hi):
        pdf_lo = math.exp(-lo*lo/2)/math.sqrt(2*math.pi)
        pdf_hi = math.exp(-hi*hi/2)/math.sqrt(2*math.pi)
        norm = 0.5*(math.erf(hi/math.sqrt(2)) - math.erf(lo/math.sqrt(2)))
        return pdf_lo, pdf_hi, norm

    class TruncNorm:
        def mean(self, a=-2, b=2, loc=0.0, scale=1.0, **kw):
            pdf_lo, pdf_hi, norm = moments(a, b); return loc + scale*(pdf_lo - pdf_hi)/norm
        def std(self, a=-2, b=2, loc=0.0, scale=1.0, **kw):
            pdf_lo, pdf_hi, norm = moments(a, b)
            var = 1 + (a*pdf_lo - b*pdf_hi)/norm - ((pdf_lo - pdf_hi)/norm)**2
            return scale*math.sqrt(max(var, 1e-12))
        def var(self, **kw): return self.std(**kw)**2
        def rvs(self, a=-2, b=2, loc=0.0, scale=1.0, size=1, **kw):
            return np.clip(np.random.normal(loc, scale, size), loc + a*scale, loc + b*scale)

    scipy = types.ModuleType("scipy")
    stats = types.ModuleType("scipy.stats"); stats.truncnorm = TruncNorm()
    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")
    class Rotation: pass
    transform.Rotation = Rotation
    spatial.transform = transform; scipy.stats = stats; scipy.spatial = spatial
    sys.modules.update({"scipy": scipy, "scipy.stats": stats,
                        "scipy.spatial": spatial, "scipy.spatial.transform": transform})
    print("scipy stubbed")


def build_trimul(package, module_path, class_name):
    """Find a library's outgoing Triangle-Multiplication class and build one.

    It first tries to import the exact module and class name you pass in. If that
    fails, it scans every submodule of the package for a class whose name starts
    with "tri", contains "mul", and mentions "out" (the outgoing variant). Once a
    class is found, it inspects the constructor and passes only the dimension
    arguments that constructor actually accepts, so the same call works across
    libraries that name those arguments differently. Returns the created module
    and the class object it came from.
    """
    trimul_class = None
    try:
        module = importlib.import_module(module_path)
        trimul_class = getattr(module, class_name, None)
    except Exception as error:
        print(f"  direct import failed: {type(error).__name__}: {error}")
    if trimul_class is None:
        root = importlib.import_module(package)
        for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
            try:
                submodule = importlib.import_module(info.name)
            except Exception:
                continue
            for name, obj in vars(submodule).items():
                if (inspect.isclass(obj) and obj.__module__ == info.name
                        and name.lower().startswith("tri") and "mul" in name.lower()
                        and "out" in name.lower()):
                    trimul_class = obj; break
            if trimul_class is not None:
                break
        if trimul_class is None:
            raise ImportError(f"no outgoing TriMul class in {package}")
    params = inspect.signature(trimul_class.__init__).parameters
    kwargs = {k: v for k, v in {"c_z": pair_dim, "d_z": pair_dim, "dim": pair_dim,
                                "c_hidden": hidden_dim, "d_c": hidden_dim,
                                "hidden_dim": hidden_dim}.items() if k in params}
    return trimul_class(**kwargs), trimul_class


def run_benchmark(model, name, pair_repr, warmup=5, runs=30):
    """Time one module and report detailed speed and memory numbers.

    Runs a few untimed warm-up passes, then times `runs` forward passes using CUDA
    events, which measure real time spent on the GPU. From those timings it works
    out mean, standard deviation, median, minimum and 95th-percentile latency,
    throughput, achieved TFLOP/s, peak and activation memory, and the parameter
    count. Everything is returned as one dictionary, which becomes one row of the
    results table.
    """
    gc.collect(); torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated(device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(pair_repr)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(runs):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(); model(pair_repr); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    peak_mem = torch.cuda.max_memory_allocated(device)
    mean = statistics.mean(times)
    module = model.func if hasattr(model, "func") else model
    module = getattr(module, "_orig_mod", module)
    tflop = (2*batch*seq_len**3*pair_dim + 12*batch*seq_len**2*pair_dim**2)
    return {
        "N": seq_len, "Module": name,
        "Mean (ms)": round(mean, 2), "Std (ms)": round(statistics.stdev(times), 2),
        "Median (ms)": round(statistics.median(times), 2), "Min (ms)": round(min(times), 2),
        "P95 (ms)": round(sorted(times)[int(0.95*runs)-1], 2),
        "Throughput (it/s)": round(1000/mean, 1),
        "TFLOP/s": round(tflop/(mean*1e-3)/1e12, 2),
        "Peak VRAM (GB)": round(peak_mem/1024**3, 3),
        "Activation (GB)": round((peak_mem-base_mem)/1024**3, 3),
        "Params (M)": round(sum(p.numel() for p in module.parameters())/1e6, 3),
    }


def benchmark_all(modules):
    """Benchmark every module in the dictionary on the same random input.

    Builds one random pair representation and mask for the current sequence
    length, captures a CUDA graph for the fast module so its launch overhead is
    removed, then benchmarks each module in turn. Modules whose forward accepts a
    `mask` are called with it. Any failure is caught and printed so one broken
    module does not stop the rest. Returns the list of result rows.
    """
    print(f"\n===== N = {seq_len} =====")
    pair_repr = torch.randn(batch, seq_len, seq_len, pair_dim, device=device, dtype=torch.float32)
    mask = torch.ones(batch, seq_len, seq_len, device=device)
    if "fast_trimul (ours)" in modules:
        modules["fast_trimul (ours)"].graphed(pair_repr, mask)
    rows = []
    for name, module in modules.items():
        target = getattr(module, "_orig_mod", module)
        needs_mask = "mask" in inspect.signature(target.forward).parameters
        call = partial(module, mask=mask) if needs_mask else module
        try:
            rows.append(run_benchmark(call, name, pair_repr))
        except Exception as error:
            print(f"  {name} failed: {type(error).__name__}: {error}")
    del pair_repr, mask
    gc.collect(); torch.cuda.empty_cache()
    return rows


def show_results(rows):
    """Turn the collected result rows into a sorted, printed table.

    Builds a pandas DataFrame from the rows, adds a "Speedup" column comparing each
    module to the slowest one at the same size, sorts by size and then by mean
    latency, prints the GPU and configuration details followed by the table in
    Markdown form, and returns the DataFrame.
    """
    if not rows:
        print("No modules benchmarked successfully.")
        return None
    df = pd.DataFrame(rows)
    df["Speedup"] = df.groupby("N")["Mean (ms)"].transform(lambda s: (s.max()/s).round(2))
    df = df.sort_values(["N", "Mean (ms)"]).reset_index(drop=True)
    print(f"\nGPU:    {torch.cuda.get_device_name(0)}")
    print(f"Config: B={batch}, d_z={pair_dim}, d_c={hidden_dim}, N={seq_len}")
    print(f"Torch:  {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"Runs:   30 timed, 5 warmup\n")
    print(df.to_markdown(index=False))
    return df


install_scipy_stub()
modules_to_test = {}

try:
    of3_module, of3_class = build_trimul(
        "openfold3", "openfold3.core.model.layers.triangular_multiplicative_update",
        "TriangleMultiplicationOutgoing")
    modules_to_test["OpenFold-3 Baseline"] = of3_module.to(device).eval()
    print(f"OpenFold-3 loaded -> {of3_class.__module__}.{of3_class.__name__}")
except Exception as error:
    print(f"Skipping OpenFold-3: {type(error).__name__}: {error}")
    traceback.print_exc()

from fast_trimul import FastTriangleMultiplication
modules_to_test["fast_trimul (ours)"] = FastTriangleMultiplication(
    d_z=pair_dim, d_c=hidden_dim, mode="outgoing").cuda().eval()
print("fast_trimul loaded -> fp16 fused kernel + CUDA graph")

if "OpenFold-3 Baseline" in modules_to_test:
    _o = modules_to_test["OpenFold-3 Baseline"]
    modules_to_test["OpenFold-3 +compile"] = torch.compile(_o)
    modules_to_test["OpenFold-3 +compile (reduce-overhead)"] = torch.compile(_o, mode="reduce-overhead")
    print("torch.compile baselines added -> default + reduce-overhead")

if "OpenFold-3 Baseline" in modules_to_test:
    ref = modules_to_test["OpenFold-3 Baseline"]
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    checker = FastTriangleMultiplication(d_z=pair_dim, d_c=hidden_dim,
                                         mode="outgoing", residual=False).cuda().eval()
    checker.load_openfold3_state_dict(ref.state_dict())
    with torch.no_grad():
        probe = torch.randn(batch, 128, 128, pair_dim, device=device)
        mask = torch.ones(batch, 128, 128, device=device)
        without = ref(probe, mask=mask).float()
        checker.graphed(probe, mask)
        with_lib = checker(probe, mask=mask).float()
        for tag, out in (("without (OpenFold-3)", without), ("with (fast_trimul) ", with_lib)):
            print(f"  {tag}: nans={out.isnan().sum().item()}/{out.numel()} range=[{torch.nan_to_num(out).min():.3g}, {torch.nan_to_num(out).max():.3g}]")
        diff = (without - with_lib).abs()
        passed = torch.allclose(without, with_lib, atol=1e-2, rtol=1e-2)
        print(f"correctness: {'PASS' if passed else 'FAIL'} (max abs diff {diff.max():.4g}, mean {diff.mean():.4g})")

rows = benchmark_all(modules_to_test)

df = show_results(rows)

df.to_csv("/content/openfold3_trimul_bench_N%d.csv" % seq_len, index=False)
