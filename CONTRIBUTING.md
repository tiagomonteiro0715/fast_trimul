# Contributing to fast_trimul

Thanks for helping! This guide covers how the code is organized, how to test it,
and the rules a pull request must meet to be merged.

## TL;DR

1. **Pick or open an issue.** Good starting points are labelled `good first issue`
   / `help wanted` — see the ready-to-open list in
   [`.github/ISSUE_BACKLOG.md`](.github/ISSUE_BACKLOG.md).
2. **Fork, branch, code.**
3. **Run locally:** `ruff check .` and `pytest -m "not gpu"` must pass.
4. **Open a PR.** Fill in the template. **CI (lint + CPU tests on Python 3.10–3.12)
   must be green before it can be merged.**
5. **If you touched the CUDA kernel or its output**, include proof it still works on
   a real GPU (a `pytest -m gpu` run — e.g. on Google Colab).

## How the code is organized

fast_trimul is a small **stable front-end** over a **registry of interchangeable
backends** (see [README → Architecture](README.md#architecture)). You almost never
need to touch the kernels to contribute:

- **`core/registry.py`** — the plug-in board. `@backend("name", ...)` registers a
  hardware backend; `@weights_for("source")` registers a library weight-map. Both are
  one decorator, no core edits.
- **`core/context.py`** — the input **guard** (`normalize`): makes a tensor contiguous,
  the right dtype, and checks alignment, or returns `None` so the dispatcher moves on.
- **`core/dispatch.py`** — the **fallback chain** (`cuda → torch`). If a backend can't
  take the input or raises, it falls back to the always-correct pure-torch path.
- **`integrations/loaders.py`** — the per-library weight remaps (`@weights_for`).
- **`integrations/patch.py`** — the one-line `patch_*` swappers.
- **`_kernels.py` / `_tensorop_gemm.py`** — the hand-written CUTLASS CuTe DSL kernels.
  **Please don't reformat these** (they're excluded from lint on purpose).

Common contributions and where they go:

| You want to… | Do this |
|---|---|
| Support a new library's weights | add a `@weights_for("yourlib")` map in `integrations/loaders.py` |
| Add a one-line `patch_yourlib()` | add it in `integrations/patch.py` |
| Add a new hardware backend | add a `@backend(...)` class (see README → *Adding a new backend*) |
| Improve/port the kernel | edit `_kernels.py` / `_tensorop_gemm.py` (needs a GPU to verify) |

## Running the tests

Tests are split by a `gpu` marker:

```bash
pytest -m "not gpu"    # CPU-only tests — no GPU needed (this is what CI runs)
pytest -m gpu          # GPU tests — needs a CUDA device
pytest                 # runs CPU tests; GPU tests show as "skipped"
```

**CPU tests** cover the registry, input guard, weight-map loaders, decorators, the
dispatch fallback chain, and the patch mechanism — no GPU required.

**GPU tests** build the real CUTLASS kernel and check its output. CI has no GPU, so it
can't run these — that's why a green CI means *"CPU checks passed,"* not *"everything
passed."* If your change affects the kernel path, run them on a GPU yourself:

```python
# Google Colab (Runtime -> Change runtime type -> GPU)
!git clone -q https://github.com/tiagomonteiro0715/fast_trimul
%cd fast_trimul
!pip install -q --no-deps .
!pip install -q pytest
!pytest -m gpu -v
```

## Lint

```bash
ruff check .
```

CI pins `ruff` to a fixed version and rule set (`E4/E7/E9/F` — basic correctness), so
local and CI always agree. Don't add opinionated rule sets that flag the intentional
`try/except` around the optional CUDA import.

## Pull request rules

A PR is merged only when **all CI checks are green**. Before you open it, make sure:

- [ ] `ruff check .` passes.
- [ ] `pytest -m "not gpu"` passes.
- [ ] Anything touching the op **matches the pure-torch reference within fp16
  tolerance** (max abs error ≈ **6×10⁻⁶**), and you list the tensor shapes you tested.
- [ ] The **pure-torch fallback still works** — an unsupported shape/dtype must degrade
  gracefully, never crash. **Don't remove the dispatch/fallback safety guards.**
- [ ] **Performance PRs** include a before/after benchmark (`main` vs. your branch, same
  GPU + shape).
- [ ] **Kernel/GPU changes** include a `pytest -m gpu` result.

## Environment notes

- **fp16, Ampere (sm80) tested.** Other hardware / bf16 / fp32 / non-multiple-of-8 `N`
  fall back to pure torch.
- **`cuda-python` must match your CUDA driver** (`<13` for CUDA-12 drivers). See
  [README → Install](README.md#install).

## Questions

Open a [Discussion](https://github.com/tiagomonteiro0715/fast_trimul/discussions) for
usage questions, or reach out at monteiro.t@northeastern.edu.
