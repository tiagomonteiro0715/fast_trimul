## What this PR does

<!-- One or two sentences. Link the issue it closes: "Closes #123". -->

## Type
- [ ] Bug fix
- [ ] New backend / weight-map (`@backend` / `@weights_for`)
- [ ] Kernel / performance
- [ ] Docs / CI / tooling

## Correctness (required for anything touching the op)
- [ ] Output matches the pure-torch reference within fp16 tolerance (**max abs error ≤ ~6×10⁻⁶**)
- [ ] Tensor shapes tested: <!-- e.g. B=1, N ∈ {8, 64, 256}, d_z=d_c=128 -->
- [ ] The pure-torch **fallback path** still works (an unsupported shape/dtype degrades, doesn't crash)

## Benchmarks (required for performance PRs)
<!-- Paste before/after on the same GPU + shape: main vs. this branch. -->

| N | main | this PR |
|--:|--:|--:|
|  |  |  |

## Checklist
- [ ] `pytest` passes locally
- [ ] Ran `ruff` (or noted why not)
- [ ] Did **not** remove or weaken the dispatch / fallback safety guards
