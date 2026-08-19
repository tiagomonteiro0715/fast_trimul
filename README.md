<h1 align="center">fast_trimul</h1>

<p align="center">
  <strong>A drop-in, hardware-agnostic library for Fused Triangle Multiplicative Updates across AlphaFold3 family models, powered by CuTe DSL</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/fast_trimul/"><img src="https://img.shields.io/pypi/v/fast_trimul" alt="PyPI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://pypi.org/project/fast_trimul/"><img src="https://img.shields.io/pypi/pyversions/fast_trimul" alt="Python"/></a>
  <img src="https://img.shields.io/github/stars/tiagomonteiro0715/fast_trimul" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/tiagomonteiro0715/fast_trimul" alt="Forks"/>
  <img src="https://img.shields.io/github/last-commit/tiagomonteiro0715/fast_trimul" alt="Last Commit"/>
</p>

<p align="center">
  Works with <strong>OpenFold-3, Chai, and Protenix (Others coming!!)</strong>.
</p>

## At a glance

This runs the real **OpenFold-3** code with `fast_trimul`! 

Just 1 line swap from the [Quickstart](#quickstart-accelerate-openfold-3-in-one-line),
with a ready-to-run `quickstart_lightning` code example included. 

The Openfold version with fast_trimul is faster than the normal Openfold version:

<p align="center">
  <img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_trunk_speed_vs_N.png" width="70%" alt="OpenFold-3 Pairformer trunk latency vs N"/>
</p>

| N (Sequence lenght) | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| % Faster (Graph vs. Native) | 33% | 30% | 31% | 32% | 24% | 17% | 15% |

**Memory**

<p align="center">
  <img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_trunk_vram_vs_N.png" width="70%" alt="OpenFold-3 Pairformer trunk peak memory vs N"/>
</p>

> **Why graph uses more memory?** Each [CUDA graph](https://modal.com/gpu-glossary/host-software/cuda-graph) reserves its own buffers (not reused across layers), so activation memory reads 0 GB but resurfaces as higher peak VRAM

However, the kernel itself keeps **near-zero activation memory** and uses **~2.2–2.4× less peak VRAM** than the plain layer (10.1 GB vs 22–24 GB at N=2048), letting it fold **~1.4× longer sequences before running out**. The higher VRAM above is only from graphing all 8 trunk layers at once ([memory figures ↓](#benchmark-results)).

Also, the kernel was developed with Python-level CuTe DSL (not thousands of lines of C++/CUDA), making the kernel codebase easily maintanable.

## Quickstart: accelerate OpenFold-3 in one line

One line `patch_openfold3()`swaps OpenFold-3's TriMul for the fast kernel
*before* you build the openfold3 model! **This exact script is verified on an NVIDIA A100 
both Lightning AI and Google Colab:**

```bash
uv pip install --system openfold3 fast_trimul "cuda-python<13"
```
```python
import fast_trimul
fast_trimul.patch_openfold3()          # <-- That is it!! OpenFold-3 now runs on the fast kernel.

# build and run OpenFold-3 exactly as you always would:
import torch
from openfold3.core.model.latent.pairformer import PairFormerStack

model = PairFormerStack(c_s=384, c_z=128, no_blocks=8, c_hidden_pair_bias=32, no_heads_pair_bias=4,
                        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
                        transition_type="swiglu", transition_n=4, pair_dropout=0.25,
                        fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9).cuda().eval()

N = 64
s = torch.randn(1, N, 384, device="cuda")
z = torch.randn(1, N, N, 128, device="cuda")
with torch.no_grad():
    _, out_z = model(s, z, torch.ones(1, N, device="cuda"), torch.ones(1, N, N, device="cuda"))
print("OpenFold-3 running on fast_trimul ->", tuple(out_z.shape))    # (1, 256, 256, 128)
```

Same one-liner for the other stacks: `patch_openfold()`, `patch_boltz()`,
`patch_protenix()`. 

Ready-to-run copies are in [`quickstart/`](quickstart/) , one for
**Lightning AI** (the snippet above) and one for **Google Colab** (adds a tiny
fake-`scipy` shim so OpenFold-3 imports without Colab's numpy quirk)

#### What if I do not want a global patch? 

No problem!

Use the module directly (`FastTriangleMultiplication`, see
*Quick start*), and use `@accelerate` / `with accelerated("cuda"):` to pin which
**backend** runs your own code.

Run the shipped benchmark on your own GPU for numbers.

See *Benchmark* below, and
read *Limitations* for where `torch.compile` is the better choice.

## Why fast_trimul?

Verified by loading trained weights and comparing outputs across OpenFold, OpenFold-3, Boltz-1, Protenix, and an AF3/Chai-style reference. These are all in reports/results/ and figures below. 

This library's trimulupdate output is identical to the standard version, with only an invisible difference of about 0.0006%.

Works at any sequence length as a direct drop-in replacement with no whole-model compilation step needed. 

Uses roughly half the GPU memory with almost zero activation data at any sequence length $N$. This way it cuts memory usage by 2.2–2.4× at $N=2048$ and fits ~1.4× longer sequences before running out of memory while reaching up to 39 TFLOP/s. 

Fastest on short sequences, running 4.5–6.8× quicker than the plain layer at $N=8$ and remaining the fastest up to $N=128$. At small sizes, most execution time goes to launching tiny GPU steps. This is done thanks to CUDA graphs that eliminate this overhead.  Thanks to this, it helps in computing workloads like large-scale screening, peptides, and repeated refinement passes. 

**Also, the library runs anywhere without crashing by using a fast CUTLASS kernel when supported and falling back to a plain PyTorch version for unusual shapes or data types.**

Supporting new hardware like TPUs or Intel GPUs requires only a  add-on rather than a complete library rewrite.

## Benchmark results

Speed (latency) and peak-VRAM scaling for a single Triangle Multiplicative Update across five stacks, plus the OpenFold-3 Pairformer End-to-End Passage. 

The below results were run on a **single NVIDIA A100-SXM4-80GB** and the whole trunk on a **single NVIDIA A100 40GB** in 100 passes.

Full tables live in [`reports/results/`](reports/results/), one-click reproduction
notebooks in [`reports/colab_reproduce/`](reports/colab_reproduce/), and the figures
are regenerated by [`reports/plot_results.py`](reports/plot_results.py).

### Boltz-1
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/boltz_speed_vs_N.png" width="100%" alt="Boltz-1 latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/boltz_vram_vs_N.png" width="100%" alt="Boltz-1 peak memory vs N"/></td>
</tr></table>

### Chai / AF3
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/chai_speed_vs_N.png" width="100%" alt="Chai / AF3 latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/chai_vram_vs_N.png" width="100%" alt="Chai / AF3 peak memory vs N"/></td>
</tr></table>

### OpenFold (AF2)
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold_speed_vs_N.png" width="100%" alt="OpenFold latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold_vram_vs_N.png" width="100%" alt="OpenFold peak memory vs N"/></td>
</tr></table>

### OpenFold-3
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_speed_vs_N.png" width="100%" alt="OpenFold-3 latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_vram_vs_N.png" width="100%" alt="OpenFold-3 peak memory vs N"/></td>
</tr></table>

### Protenix
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/protenix_speed_vs_N.png" width="100%" alt="Protenix latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/protenix_vram_vs_N.png" width="100%" alt="Protenix peak memory vs N"/></td>
</tr></table>

### Whole-trunk - OpenFold-3 Pairformer (8 blocks)
`fast_trimul (ungraphed)` = the fused kernel un-graphed; `fast_trimul (graphed)` = with a captured CUDA graph; `OpenFold-3 (stock)` = the unmodified trunk.
<table><tr>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_trunk_speed_vs_N.png" width="100%" alt="OpenFold-3 Pairformer trunk latency vs N"/></td>
<td><img src="https://raw.githubusercontent.com/tiagomonteiro0715/fast_trimul/main/reports/figures/openfold3_trunk_vram_vs_N.png" width="100%" alt="OpenFold-3 Pairformer trunk peak memory vs N"/></td>
</tr></table>

> **Why the baselines are Out of Memory (OOM):** their temporary activations grow quadratically-to-cubically with N. By N=3072 they no longer fit the 80 GB NVIDIA A100 GPU. `torch.compile` shrinks this but doesn't remove those buffers.


## Table of Contents
- [Quickstart: accelerate OpenFold-3 in one line](#quickstart-accelerate-openfold-3-in-one-line)
- [Why fast_trimul?](#why-fast_trimul)
- [Benchmark results](#benchmark-results)
- [Install](#install)
- [Quick start](#quick-start)
- [Swap into a real model (whole-trunk example)](#swap-into-a-real-model-whole-trunk-example)
- [Gradients (training)](#gradients-training)
- [Benchmark](#benchmark)
- [Architecture](#architecture)
- [API](#api)
- [Limitations](#limitations-read-before-relying-on-it)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Related Resources](#related-resources)
- [Built With](#built-with)
- [Contact](#contact)
- [Citation](#citation)
- [License](#license)

## Install

```bash
pip install fast_trimul          # or: uv pip install fast_trimul
```

Or install the latest straight from GitHub.

```bash
pip install git+https://github.com/tiagomonteiro0715/fast_trimul
```

| | Hardware / precision / shape |
|---|---|
| **Fast path** (CUTLASS kernel) | NVIDIA **Ampere** (A100, RTX 3090/4090), fp16, N divisible by 8 |
| **Pure-torch fallback** (always correct) | Everything else: Hopper/Blackwell, non-Ampere GPUs, CPU, TPU, bf16/fp32, or N not divisible by 8 |

**Driver note:** `cuda-python` must match your CUDA **driver** (it can be newer than the runtime, never older). If `nvidia-smi`
shows CUDA 12.x :

```bash
pip install "cuda-python<13"
```
`fast_trimul` pins `cuda-python<13` by default (most drivers are still CUDA 12.x); override it if your driver is CUDA 13+.

**Troubleshooting the first run** (kernels JIT-compile then):
- **JIT/build errors** (missing `nvcc` or CUDA headers): set `CUDA_HOME` to your CUDA toolkit so the compiler and headers are found.
- **Slow or failing first-shape autotune:** set `FAST_TRIMUL_AUTOTUNE=0` to skip GEMM autotuning.

## Quick start

On **Google Colab** (Runtime → Change runtime type → **GPU**), install first:

```python
!pip install -q uv
!uv pip install fast_trimul
```

Then use it:

```python
import torch
from fast_trimul import FastTriangleMultiplication

module = FastTriangleMultiplication(d_z=128, d_c=128, mode="outgoing").cuda()
z = torch.randn(1, 256, 256, 128, device="cuda")          # (B, N, N, d_z)
mask = torch.ones(1, 256, 256, device="cuda")             # optional (B, N, N)
out = module(z, mask=mask)                                 # same dtype as z
```

For **fastest inference at a fixed shape**, capture a CUDA graph once. 

This removes the per-launch overhead of the internal kernels, which dominates the runtime at small Ns:

```python
module.graphed(z, mask)      # capture once at this shape (inference only)
out = module(z, mask=mask)   # subsequent calls replay the graph
```

Low-level functional API:

```python
from fast_trimul import functional
out = functional.triangle_multiplication(z, module._impl, mask=mask)
```

Load pretrained weights from a target library.

With one call, pick the `source` (names are remapped, and fused a/b projections are split, for you):

```python
module.load_weights(ref.state_dict(), source="openfold")   # or: openfold3 / protenix / boltz
```

The five named helpers (`load_openfold_state_dict`, …) still work as thin aliases.

These target modules apply their residual (`+ z`) **outside** the triangle block. For this reason it was build with `residual=False` tp match their output exactly:

```python
module = FastTriangleMultiplication(d_z=128, d_c=128, mode="outgoing", residual=False).cuda()
```

**Pick or force a backend**

 (default is `"auto"` - fastest available, then torch):

```python
from fast_trimul import list_backends
list_backends()                                            # e.g. ['torch', 'cuda']
FastTriangleMultiplication(d_z=128, backend="cuda")        # force the fast path (still falls back)
FastTriangleMultiplication(d_z=128, backend="torch")       # force the portable reference
```

**Opt in explicitly, without any global monkeypatching** (safe under strict runtime policies) - `@accelerate` on a function, or the scoped `accelerated()`:

```python
from fast_trimul import accelerate, accelerated

@accelerate                    # runs this function with the accelerated backend
def infer(z): ...

with accelerated("cuda"):      # scoped: reverts on exit, never leaks
    out = model(z)
```

**One-line correctness check**. Build your library's TriMul, then:

```python
import fast_trimul
fast_trimul.verify("openfold", my_openfold_trimul)         # -> True if outputs match (fp16 tol)
```

`verify` stays a one-liner even when a library changes its API, because *you* pass the reference module and only the name remap is library-specific.

## Swap into a real model (whole-trunk example)

Drop `fast_trimul` into a library's trunk by replacing its TriMul class with a adapter, then capturing a CUDA graph per layer. 

**Verified end-to-end on the OpenFold-3 Pairformer trunk (A100).**

```python
import torch
import openfold3.core.model.latent.base_blocks as blocks
from openfold3.core.model.latent.pairformer import PairFormerStack
from fast_trimul import FastTriangleMultiplication

# 1) a thin adapter: match the library's constructor, swallow its extra forward kwargs,
#    and use residual=False (the block adds the residual itself).
class FastTriMul(FastTriangleMultiplication):
    def __init__(self, c_z, c_hidden=None, *args, **kw):
        super().__init__(d_z=c_z, d_c=c_hidden or c_z, mode="outgoing", residual=False)
    def forward(self, z, mask=None, **kw):
        return super().forward(z, mask=mask)

# 2) patch the library's TriMul classes BEFORE building the model
blocks.TriangleMultiplicationOutgoing = FastTriMul
blocks.TriangleMultiplicationIncoming = FastTriMul
blocks.FusedTriangleMultiplicationOutgoing = FastTriMul
blocks.FusedTriangleMultiplicationIncoming = FastTriMul

model = PairFormerStack(c_s=384, c_z=128, no_blocks=8, ...).cuda().eval()

# 3) capture a CUDA graph for each swapped-in layer (fixed shape, inference only)
dummy_z = torch.randn(1, N, N, 128, device="cuda")
dummy_mask = torch.ones(1, N, N, device="cuda")
for m in model.modules():
    if isinstance(m, FastTriangleMultiplication):
        m.graphed(dummy_z, dummy_mask)

# ... now run model(s, z, single_mask, pair_mask) as usual ...
```

Install for this example (note the CUDA-12 driver pin - see *Install*):
```bash
uv pip install --system openfold3 fast_trimul "cuda-python<13"
```

### Whole-trunk results (OpenFold-3 Pairformer, 8 blocks)

Full forward of the Pairformer trunk, random weights and inputs, 100 timed passes.

Measured on **NVIDIA A100-SXM4-40GB (Lightning AI)**. Latency in ms (lower is better), peak VRAM in GB; **bold** = fastest at that N.

| N | native | eager | graphed | native VRAM | eager VRAM | graphed VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 8   | 29.86  | 53.55  | **22.51**  | 0.086 | 0.084 | 0.085 |
| 16  | 28.87  | 52.49  | **22.15**  | 0.087 | 0.085 | 0.089 |
| 32  | 29.68  | 53.37  | **22.59**  | 0.093 | 0.091 | 0.108 |
| 64  | 29.09  | 52.54  | **22.06**  | 0.116 | 0.118 | 0.183 |
| 128 | 30.15  | 53.60  | **24.26**  | 0.211 | 0.224 | 0.483 |
| 256 | 121.56 | 105.61 | **104.29** | 0.837 | 0.897 | 1.933 |
| 512 | 657.51 | **571.46** | 573.98 | 5.092 | 5.340 | 9.481 |

The graphed approach runs faster than the native setup across batch sizes, providing a 1.15 to 1.35× speedup. However, it allocates one CUDA graph per layer, which increases peak memory to 1.9× the native amount at $N=512$.  This method works best when speed takes priority and extra memory remains available.

The eager approach matches native memory usage while exceeding native speed only at batch sizes of 256 and higher.  At small batch sizes, it runs slower than native because launching ungraphed fused kernels creates overhead across 16 TriMul layers. Overall, selection depends on workload requirements. 

Users should select the graphed option to reduce delay or select the eager option for large batch sizes while keeping memory at native levels. The primary advantage comes from speed using graphs rather than memory reduction.

## Gradients (training)

The forward runs the fused fp16 kernel. 

The backward is a **correct torch recompute** (correct, not yet fast), so gradients flow and you can train and fine-tune with it. 

Do **not** call `.graphed()` for training - graphs are inference only.

```python
import torch
from fast_trimul import FastTriangleMultiplication

trimul = FastTriangleMultiplication(d_z=128, d_c=128, mode="outgoing").cuda()
z = torch.randn(1, 64, 64, 128, device="cuda", requires_grad=True)   # N a multiple of 8
out = trimul(z, mask=torch.ones(1, 64, 64, device="cuda"))
out.sum().backward()                                                 # gradients recomputed through the kernel
assert z.grad is not None and torch.isfinite(z.grad).all()           # input grads flow
assert all(p.grad is not None for p in trimul.parameters())          # parameter grads flow
```

## Benchmark

The package ships a benchmark that measures machine ceilings (memory bandwidth,
fp16 tensor-core peak, launch floor), a per-iteration **median** timer, achieved
TFLOP/s, peak memory, and a size sweep. 

It reports `fast_trimul` **both un-graphed
and graphed**, next to `torch.compile` and an eager reference, so you can compare
on your own hardware:

```python
!pip install -q uv
!uv pip install fast_trimul
```
```python
from fast_trimul.benchmark import run_benchmark
run_benchmark()              # or: run_benchmark(head_size=384, sweep=(128, 256, 512))
```

Or from a shell:
```bash
python -m fast_trimul.benchmark
```

It reports these variants:

* **`fast no-graph`** - the kernel, fp16, un-graphed (shows the launch-overhead cost),
* **`fast +graph`** - the same kernel with a captured CUDA graph (`.graphed()`),
* **`compile`** - `torch.compile(mode="reduce-overhead")` and default mode,
* **`torch eager`** - the eager reference.

**Mode by N (the trade-off):** use `+graph` for small-N latency; at **large N prefer eager (`no-graph`)** - it matches native memory and skips the CUDA-graph VRAM reservation, which only pays off when you're latency-bound with VRAM headroom.

Use CUDA events + `synchronize()` (as the shipped benchmark does) so timing
reflects when the GPU *finishes* the work, not when the launch is queued. Warm up
(or call `.graphed()`) before timing to exclude the one-time JIT/autotune cost.

## Architecture

The library is a small stable front-end over a*registry of interchangeable
backends. 


CUDA is one backend with a pure-torch backend is the universal fallback.

**IMPORTANT: Adding new hardware or a new library is a plug-in (one decorated class/function),
not a core edit**

The kernels are written in **Python-level CuTe DSL**, not thousands of lines of
C++/CUDA

This way, the whole kernel codebase stays **small and readable**, and tuning a
tile size, adding a fused epilogue, or porting to a new GPU is a quick edit instead
of a rewrite.

```
  front-end   @accelerate  accelerated()  verify()          # stable, tiny
      │
  guard        contiguous · dtype · align · int64 strides   # NormalizedInput
      │
  dispatch     pick backend, then FALL BACK: cuda -> torch  # never crashes on a bad shape
      │
  backends     cuda_cute (CUTLASS)   torch_ref (portable)   # + future: xla/tpu, xpu/intel
```

| Module | Role |
|---|---|
| `core/registry.py` | `@backend` / `@weights_for` decorators + O(1) lookup tables |
| `core/context.py` | the input **guard** → `NormalizedInput` (contiguous, dtype, alignment, int64 strides) |
| `core/dispatch.py` | picks a backend and walks the **fallback chain** (`cuda → torch`) |
| `core/decorators.py` | `@accelerate` + `accelerated()` - explicit, scoped, no global patching |
| `backends/torch_ref.py` | universal pure-torch backend (any device torch supports) |
| `backends/cuda_cute.py` | thin wrapper over the existing CUTLASS kernels (unchanged) |
| `ops/triangle.py` | `FastTriangleMultiplication` (dispatch + graph capture + `load_weights`) |
| `integrations/loaders.py` | the five per-library weight maps (`@weights_for`) |
| `integrations/checks.py` | `verify()` |

Everything the architecture adds is **O(1)** overhead

Only the op itself scales with `N`. **The CUDA kernels (`_kernels.py`) are untouched.**

### Adding a new backend

```python
from fast_trimul.core.registry import backend

@backend("mybackend", dtypes={torch.float16}, min_align=8)
class MyBackend:
    def __init__(self, caps): self.caps = caps
    def execute(self, inp, params): ...   # inp.tensor is guarded (contiguous, right dtype)
```

That one file makes `backend="mybackend"` selectable and slots it into the fallback chain.

## API

- `fast_trimul.FastTriangleMultiplication(d_z, d_c=None, mode="outgoing", residual=True, backend="auto")`
  - the module. `forward(z, mask=None)`, `.graphed(z, mask=None)`,
  `.load_weights(state_dict, source=...)` (plus the named aliases
  `.load_openfold_state_dict` / `.load_openfold3_state_dict` /
  `.load_protenix_state_dict` / `.load_boltz_state_dict`).
- `fast_trimul.accelerate` / `fast_trimul.accelerated(backend="auto")` - explicit opt-in decorator / scoped context manager.
- `fast_trimul.verify(source, reference, ...)` - one-line correctness check against a reference module.
- `fast_trimul.list_backends()` - backends registered on this machine.
- `fast_trimul.functional.triangle_multiplication(z, params, mask=None)` - low-level functional call.
- `fast_trimul.core.registry.{backend, weights_for}` - decorators to register a new backend or library weight-map.

## Limitations (read before relying on it)

- **`torch.compile(mode="reduce-overhead")` is competitive and often faster above
  small N.** On an A100 it is often faster per call in the mid-range and, on
  several stacks, uses similar peak memory. These kernels are not yet epilogue-fused
  (future work), so the reasons to prefer this are **drop-in-ness and robustness**,
  not raw latency: `reduce-overhead` needs *static* shapes and **recompiles for
  every new sequence length** - and in protein modeling `N` changes with almost
  every input, so you pay repeated recompiles and latency spikes (and sometimes
  memory blow-ups) throughout a run - and it can break on some models, whereas this
  is a plain `nn.Module` that works on any shape with no compilation step.
  Benchmark both on your workload.

- **Pretrained weights need name remapping.** Each library names its
  projections/norms differently, so a strict checkpoint load will not line up.
  Automated for the common stacks via `load_weights(sd, source=...)`:
  `openfold` (OpenFold/AF2), `openfold3` (separate or fused variant), `protenix`,
  `boltz`, and `chai` (Boltz/Chai/AF3 fuse the a/b projections). Other stacks: supply
  a `@weights_for` remap.

- **Don't wrap a patched model in `torch.compile`.** After `patch_openfold3()` the
  TriMul runs a JIT-compiled CUTLASS kernel that `torch.compile` can't trace through

- **Backward is correct but not fast** (torch recompute), so it helps inference
  more than training throughput.

- **The CUDA kernel is Ampere (sm80) tested** On other hardware, or for
  bf16/fp32, or non-multiple-of-8 `N`, the dispatcher falls back to the pure-torch
  backend (correct, slower). Hopper/Blackwell + fp8 are future work.


## Roadmap

- [ ] Epilogue fusion / megakernel: fuse post-GEMM norms + activations.
- [ ] Multi-arc support: H100/H200 (FP8/FP16 paths), B200 (NVFP4/FP4, Gen 2 Transformer Engine).
- [ ] Fused backward kernel (transposed operands) to remove autograd memory overhead.
- [ ] Per-stack quickstart code examples in the README (e.g. Boltz-1) - once the CI/CD pipeline validates the installs end to end.

## Star History

<a href="https://www.star-history.com/#tiagomonteiro0715/fast_trimul&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=tiagomonteiro0715/fast_trimul&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=tiagomonteiro0715/fast_trimul&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=tiagomonteiro0715/fast_trimul&type=Date" />
 </picture>
</a>

## Contributing

Contributions, bug reports, and benchmark numbers from your own hardware are welcome!
**Start with [CONTRIBUTING.md](CONTRIBUTING.md)** - how to run the tests, the PR rules
(CI must pass), and where to add a backend or weight-map. Good first tasks are in
[`.github/ISSUE_BACKLOG.md`](.github/ISSUE_BACKLOG.md).

- **Found a bug, a shape that falls back unexpectedly, or a stack whose weights
  don't remap?** Open an issue, or reach out at monteiro.t@northeastern.edu.
- **Added a new backend or a `@weights_for` map for another library?** Send a pull
  request - a new backend or weight-map is one decorated class/function and needs no
  changes to the core (see *Architecture → Adding a new backend*).
- **Enjoyed it?** Star the repository - it helps others find the project.

## Related Resources

- [The Math Behind Artificial Intelligence](https://github.com/tiagomonteiro0715/The-Math-Behind-Artificial-Intelligence-A-Guide-to-AI-Foundations) - a guide to AI's mathematical foundations from an engineering perspective.
- [My FreeCodeCamp Articles](https://www.freecodecamp.org/news/author/tiagomonteiro) - tutorials and deep dives on AI and programming.
- [Signal Processing Guide](https://github.com/tiagomonteiro0715/Signal-Processing-and-Systems-in-Programming-Guide-for-Beginners) - companion resource on signal processing.

**New to the GPU terms used here?** Modal's [GPU Glossary](https://modal.com/gpu-glossary) explains them well -
[CUDA graph](https://modal.com/gpu-glossary/host-software/cuda-graph),
[CUTLASS](https://modal.com/gpu-glossary/host-software/cutlass),
[Tensor Core](https://modal.com/gpu-glossary/device-hardware/tensor-core),
[shared memory (SRAM)](https://modal.com/gpu-glossary/device-software/shared-memory),
[kernel](https://modal.com/gpu-glossary/device-software/kernel),
[warp](https://modal.com/gpu-glossary/device-software/warp), and
[Streaming Multiprocessor](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor).

## Built With

- **Python** - the front-end, dispatcher, backends, and integrations.
- [**NVIDIA CUTLASS CuTe DSL**](https://github.com/NVIDIA/cutlass) - the fused GEMM / kernel core, written in Python-level CuTe DSL so the kernels stay clean, editable, and easy to re-tune for new GPU generations.
- [**PyTorch**](https://pytorch.org/) - tensors, autograd, and CUDA-graph capture.
- [**cuda-python**](https://github.com/NVIDIA/cuda-python) - the driver bindings the kernels launch through.
- [**uv**](https://github.com/astral-sh/uv) - fast Python package installer used throughout the docs.

## Contact

**Tiago Monteiro**
- Email: monteiro.t@northeastern.edu
- GitHub: [@tiagomonteiro0715](https://github.com/tiagomonteiro0715)
- FreeCodeCamp: [Author Profile](https://www.freecodecamp.org/news/author/tiagomonteiro)

## Citation

If you use `fast_trimul` in research, please cite it:

```bibtex
@software{monteiro_fast_trimul,
  author = {Monteiro, Tiago},
  title  = {fast_trimul: Fused Triangle Multiplicative Update on CUTLASS CuTe DSL kernels},
  year   = {2026},
  url    = {https://github.com/tiagomonteiro0715/fast_trimul}
}
```

## License

Apache License 2.0 (this project) - see [LICENSE](LICENSE). The GEMM core is
derived from NVIDIA CUTLASS - specifically the CuTe DSL Ampere dense-GEMM example
[`tensorop_gemm.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py)
- and is licensed under BSD 3-Clause - see [NOTICE](NOTICE).

---

<p align="center">
  If you find <code>fast_trimul</code> useful, please consider starring the repository!<br>
  Your support helps others discover this project.
</p>
