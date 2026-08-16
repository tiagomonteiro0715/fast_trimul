# reports

Benchmark evidence for `fast_trimul` — the recorded results, the figures the main
README shows, and the scripts that reproduce them.

```
reports/
├── colab_reproduce/   one-click Google Colab scripts, one per stack (boltz, chai,
│                       openfold, openfold3, protenix) — install, verify parity, benchmark
├── results/           the recorded benchmark tables, results_<stack>.md (+ the
│                       openfold3 whole-trunk table)
├── figures/           speed-vs-N and VRAM-vs-N PNGs (Nature-style, 600 dpi)
├── plot_results.py    regenerates every figure in figures/ from the tables in results/
└── pyproject.toml     the plotting environment (matplotlib, pandas, scienceplots)
```

## Reproduce the measurements

Each script in [`colab_reproduce/`](colab_reproduce/) is a self-contained Colab
notebook export: open it, set the runtime to **GPU**, and Run all. It installs
everything from scratch, checks that `fast_trimul` matches the stack's own layer
within fp16 tolerance, then benchmarks it against the eager baseline and two
`torch.compile` variants. Each script's docstring links the exact Colab notebook.

Measured on an **NVIDIA A100** (single-op: A100-SXM4-80GB, PyTorch 2.13 / CUDA 13.0,
B=1, d_z=d_c=128, medians of 30 timed / 5 warmup runs; whole-trunk: A100-40GB, 100
passes). Runs use random weights and inputs, so absolute numbers vary run to run —
the trends and medians are what reproduce.

## Regenerate the figures

```bash
uv run plot_results.py
```

or, managing the environment yourself:

```bash
pip install matplotlib pandas scienceplots
python plot_results.py
```

It reads every `results/results_*.md` table and writes two PNGs per stack into
`figures/` (`<stack>_speed_vs_N.png`, `<stack>_vram_vs_N.png`).
