"""Benchmark: Chai / AF3 Triangle-Multiplication versus fast_trimul (Google Colab).

Chai-1 only ships compiled artifacts, so this script uses a plain-PyTorch
reference layer (`AF3ReferenceTriMulOutgoing`) that performs the exact same
operation. It wraps that reference in two torch.compile variants and compares all
of them against fast_trimul's fused kernel. It first confirms fast_trimul matches
the reference (after copying weights across), then times every variant over a
chosen sequence length and prints a table of latency, throughput, and memory. Run
it on a GPU runtime. The recorded results live in results_chai.md.

Reproduce it in one click: open the Colab notebook, set the runtime to GPU
(Runtime -> Change runtime type -> GPU), and Run all. It installs everything from
scratch and runs end to end, so the benchmark reproduces without any local setup.
For comparable numbers, use the same setup it was measured on: an NVIDIA A100,
PyTorch 2.13 / CUDA 13.0.

    ▶ Open in Google Colab:
      https://colab.research.google.com/drive/1pXfcURA_xJE5rutXps1Ne-Am-2PTHyvD?usp=sharing
"""

!pip install -q uv
!uv pip install -q --system torch pandas tabulate psutil nvidia-ml-py
!uv pip install -q --system --upgrade fast_trimul

import gc, inspect, statistics
from functools import partial
import pandas as pd
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "CUDA GPU is required!"
device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

seq_len = 8
batch, pair_dim, hidden_dim = 1, 128, 128


class AF3ReferenceTriMulOutgoing(nn.Module):
    """A plain-PyTorch stand-in for the AF3 / Chai outgoing Triangle-Multiplication.

    Chai-1 does not ship an importable layer, so this class reproduces the same
    math in ordinary PyTorch to serve as the "baseline" to benchmark against. It
    matches the fused AF3 design: the input and gate projections are fused into a
    single wide linear (width 2*dim, then split into two halves a and b), the
    linear layers have no bias, and there is no residual connection. The forward
    normalises the input, forms the two halves, contracts them with an einsum over
    the outgoing index, and applies the output projection gated by a sigmoid.
    """
    def __init__(self, dim=128):
        super().__init__()
        self.norm_in  = nn.LayerNorm(dim, eps=1e-5)
        self.p_in     = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in     = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out    = nn.Linear(dim, dim, bias=False)
        self.g_out    = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        """Run the outgoing triangle update on the pair tensor `x`.

        Normalises `x`, computes the gated input projection, optionally applies the
        mask, splits the result into its a and b halves, contracts them over the
        shared index, and returns the gated, normalised output projection.
        """
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x.float(), 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        return self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()


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


modules_to_test = {}

modules_to_test["Chai (AF3 ref) Baseline"] = AF3ReferenceTriMulOutgoing(pair_dim).to(device).eval()
print("Chai (AF3 ref) loaded -> unfused PyTorch baseline")

from fast_trimul import FastTriangleMultiplication
modules_to_test["fast_trimul (ours)"] = FastTriangleMultiplication(
    d_z=pair_dim, d_c=hidden_dim, mode="outgoing").cuda().eval()
print("fast_trimul loaded -> fp16 fused kernel + CUDA graph")

_ref = modules_to_test["Chai (AF3 ref) Baseline"]
modules_to_test["Chai +compile"] = torch.compile(_ref)
modules_to_test["Chai +compile (reduce-overhead)"] = torch.compile(_ref, mode="reduce-overhead")
print("torch.compile baselines added -> default + reduce-overhead")

ref = modules_to_test["Chai (AF3 ref) Baseline"]
for p in ref.parameters():
    p.data.normal_(0, 0.02)
checker = FastTriangleMultiplication(d_z=pair_dim, d_c=hidden_dim,
                                     mode="outgoing", residual=False).cuda().eval()
checker.load_weights(ref.state_dict(), source="chai")
with torch.no_grad():
    probe = torch.randn(batch, 128, 128, pair_dim, device=device)
    mask = torch.ones(batch, 128, 128, device=device)
    without = ref(probe, mask=mask).float()
    checker.graphed(probe, mask)
    with_lib = checker(probe, mask=mask).float()
    for tag, out in (("without (Chai/AF3)", without), ("with (fast_trimul)", with_lib)):
        print(f"  {tag}: nans={out.isnan().sum().item()}/{out.numel()} range=[{torch.nan_to_num(out).min():.3g}, {torch.nan_to_num(out).max():.3g}]")
    diff = (without - with_lib).abs()
    passed = torch.allclose(without, with_lib, atol=1e-2, rtol=1e-2)
    print(f"correctness: {'PASS' if passed else 'FAIL'} (max abs diff {diff.max():.4g}, mean {diff.mean():.4g})")

rows = benchmark_all(modules_to_test)

df = show_results(rows)

df.to_csv("/content/chai_trimul_bench_N%d.csv" % seq_len, index=False)
