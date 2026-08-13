# fast_trimul

Fused **Triangle Multiplicative Update** (AlphaFold2 / OpenFold) built on
hand-written **CUTLASS CuTe DSL** kernels — a drop-in `nn.Module` for the
structural-biology stacks (OpenFold, Boltz, Chai, Protenix).

> **Honest status.** The kernels are **fp16** and **numerically correct** (they
> match PyTorch fp16 to fp16 tolerance). On a *fair* comparison
> (`torch.compile(..., mode="reduce-overhead")` in fp16) they are **slower than
> `torch.compile` above small N** today — the GEMMs are not yet epilogue-fused.
> The wins are: correctness, a drop-in API, and (with full fusion, future work)
> lower memory. Full GEMM epilogue fusion and a FlashAttention-style megakernel
> are **future work** — see *Limitations*.

## Install

```bash
pip install fast_trimul          # or: uv pip install fast_trimul
```
Requires a **CUDA GPU**, `torch`, `nvidia-cutlass-dsl`, and `cuda-python`.
Kernels JIT-compile on first use (one-time cost, then cached in-process).

## Quick start

```python
import torch
from fast_trimul import FastTriangleMultiplication

module = FastTriangleMultiplication(d_z=128, d_c=128, mode="outgoing").cuda()
z = torch.randn(1, 256, 256, 128, device="cuda")          # (B, N, N, d_z)
mask = torch.ones(1, 256, 256, device="cuda")             # optional (B, N, N)
out = module(z, mask=mask)                                 # same dtype as z
```

Low-level functional API (FlashAttention style):

```python
from fast_trimul import functional
out = functional.triangle_multiplication(z, module._impl, mask=mask)
```

## Drop-in monkeypatch for the 4 target libraries

Each helper replaces the library's TriMul class with an adapter matching its
constructor. **Patch _before_ building the model.** See *Limitations* for the
pretrained-weight caveat.

### OpenFold
```python
import fast_trimul.integrations as fti
fti.patch_openfold()          # patches Outgoing + Incoming
# ... now build your OpenFold model as usual ...
```
Equivalent manual form:
```python
import openfold.model.triangular_multiplicative_update as of_tri
from fast_trimul.integrations import adapter
of_tri.TriangleMultiplicationOutgoing = adapter("outgoing")
of_tri.TriangleMultiplicationIncoming = adapter("incoming")
```

### Boltz-1 / BoltzDesign
```python
import fast_trimul.integrations as fti
fti.patch_boltz()
```
Manual form:
```python
import boltz.model.layers.triangular_mult as b_tri
from fast_trimul.integrations import adapter
b_tri.TriangleMultiplicationOutgoing = adapter("outgoing")
b_tri.TriangleMultiplicationIncoming = adapter("incoming")
```

### Protenix
```python
import fast_trimul.integrations as fti
fti.patch_protenix()
```
Manual form:
```python
import protenix.model.modules.pairformer as p_tri
from fast_trimul.integrations import adapter
p_tri.TriangleMultiplication = adapter("outgoing")
```

### Chai-1
Chai's module path is version-dependent, so patch the attribute explicitly
(replace the import path with the one in your installed version):
```python
from fast_trimul.integrations import adapter
import chai_lab.model.<...>.triangle_mult as c_tri   # <- verify path for your version
c_tri.TriangleMultiplicationOutgoing = adapter("outgoing")
c_tri.TriangleMultiplicationIncoming = adapter("incoming")
```

## API

- `fast_trimul.nn.FastTriangleMultiplication(d_z, d_c=None, mode="outgoing")` — high-level module, `forward(z, mask=None)`.
- `fast_trimul.functional.triangle_multiplication(z, params, mask=None)` — low-level functional call.
- `fast_trimul.integrations.{patch_openfold, patch_boltz, patch_protenix, adapter}` — monkeypatch helpers.

## Limitations (read before relying on it)

- **Slower than `torch.compile` (fp16) above small N.** Correctness and drop-in
  compatibility come first; speed parity needs the epilogue fusion / megakernel
  (future work).
- **fp16 only.** bf16/fp32 inputs are cast to fp16 and back; keep the module in
  fp16 (do not call `.float()`/`.bfloat16()` on it).
- **Pretrained weights need name remapping.** Each library names its
  projections/norms differently, so a strict checkpoint load will not line up.
  Patch-then-train, or supply a parameter remap. Loading pretrained checkpoints
  is not yet automated.
- **Mask semantics are approximate.** The mask is applied to the pair tensor in
  and out; validate against each library's exact masking before production use.
- **Backward is correct but not fast** (torch recompute), so it helps inference
  more than training throughput.
- **Ampere (sm80) tested.** Hopper/Blackwell + fp8 are future work.
- **`import fast_trimul` needs a CUDA GPU** (device properties are read at import).

## License

MIT (this project). The GEMM core is derived from NVIDIA CUTLASS and is licensed
under BSD 3-Clause — see [NOTICE](NOTICE).
