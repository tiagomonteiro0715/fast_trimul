# OpenFold-3 Pairformer (8 blocks) — whole-trunk benchmark results

Full forward of the Pairformer trunk, random weights + inputs, 100 timed passes.

- **GPU:** NVIDIA A100-SXM4-40GB (Lightning AI)
- **Metric:** latency in ms (lower is better); peak VRAM in GB
- **Variants:** `native` (the library's own TriMul), `eager` (fast_trimul un-graphed), `graphed` (fast_trimul with a captured CUDA graph)

| N | native | eager | graphed | native VRAM | eager VRAM | graphed VRAM |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 29.86 | 53.55 | 22.51 | 0.086 | 0.084 | 0.085 |
| 16 | 28.87 | 52.49 | 22.15 | 0.087 | 0.085 | 0.089 |
| 32 | 29.68 | 53.37 | 22.59 | 0.093 | 0.091 | 0.108 |
| 64 | 29.09 | 52.54 | 22.06 | 0.116 | 0.118 | 0.183 |
| 128 | 30.15 | 53.60 | 24.26 | 0.211 | 0.224 | 0.483 |
| 256 | 121.56 | 105.61 | 104.29 | 0.837 | 0.897 | 1.933 |
| 512 | 657.51 | 571.46 | 573.98 | 5.092 | 5.340 | 9.481 |
