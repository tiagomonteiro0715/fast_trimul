"""Generate Nature-style benchmark figures from the results_*.md tables.

For every `results_<stack>.md` file in the `results/` folder this script reads the
consolidated benchmark table and writes two PNG figures into `figures/`:

  * `<stack>_speed_vs_N.png` : latency (ms) versus sequence length N
  * `<stack>_vram_vs_N.png`  : peak GPU memory (GB) versus sequence length N

Each figure draws one line per module (the library baseline, its torch.compile
variants, and fast_trimul), so you can see how speed and memory scale with N.

It handles two table shapes automatically: the long per-operation tables that have
a `Module` column, and the wide whole-trunk table whose columns are the variants
themselves (`native`, `eager`, `graphed`) with matching `... VRAM` columns.

Where a module stops before the largest N tested, it ran out of GPU memory; the
figure marks that point with an "x" and an "out of memory" legend entry.

The figure style follows Nature's figure guidelines and is built on the
`scienceplots` "science + nature" style for a cleaner publication look:
  * a single-column width of 89 mm (about 3.5 inches);
  * a small sans-serif font, matching Nature's figure text;
  * a colourblind-safe palette (Okabe-Ito, with pink removed), no red/green
    confusion, with fast_trimul fixed to a prominent vermilion;
  * log-log axes with fine minor ticks and 0.2 / 0.5 labels, so the low 0.1-to-1
    range is easy to read;
  * top and right spines removed and only faint gridlines;
  * saved as PNG at 600 dpi, above Nature's 300 dpi minimum for line art.

Run it (this installs matplotlib, pandas, and scienceplots automatically):

    uv run main.py

or, if you manage the environment yourself:

    pip install matplotlib pandas scienceplots
    python main.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import pandas as pd

try:
    import scienceplots  # noqa: F401  registers the "science" / "nature" styles
    _HAS_SCIENCE = True
except Exception:
    _HAS_SCIENCE = False

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

# Okabe-Ito, colourblind-safe, with pink (#CC79A7) and low-contrast yellow removed.
# fast_trimul is first, so it always gets the prominent vermilion.
PALETTE = ["#D55E00", "#0072B2", "#009E73", "#000000", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Per-file relabelling of module names (applied by filename stem).
LABEL_OVERRIDES = {
    "results_openfold3_trunk": {
        "native": "OpenFold-3 (stock)",
        "eager": "fast_trimul (ungraphed)",
        "graphed": "fast_trimul (graphed)",
    },
}

# Nicer plot titles (applied by stack name); falls back to the stack name.
TITLE_OVERRIDES = {
    "openfold3_trunk": "OpenFold-3 Pairformer trunk (8 blocks)",
}


def set_nature_style():
    """Apply the scienceplots + Nature style, then our own small overrides.

    Uses the `scienceplots` "science" and "nature" styles when available for a
    clean publication look, then pins the font sizes, thin axis lines, and high
    export resolution so every figure this script produces matches.
    """
    if _HAS_SCIENCE:
        plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 600,
        "figure.dpi": 600,
    })


def _read_raw(path):
    """Read the first Markdown table in a file into a DataFrame of text cells.

    Keeps only the lines that form the table, treats the first such line as the
    header, and skips the dashed separator line.
    """
    header, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            header = cells
            continue
        rows.append(dict(zip(header, cells)))
    return pd.DataFrame(rows)


def _num(value):
    """Turn a table cell into a float, ignoring any Markdown bold asterisks."""
    return float(str(value).replace("*", "").strip())


def load_long(path):
    """Read a results table into a tidy long DataFrame of N, Module, latency, VRAM.

    Two table shapes are supported. If the table has a `Module` column it is
    already long, so the sequence length, latency, and memory columns are pulled
    out directly. Otherwise it is the wide whole-trunk table: every non-N column
    that is not a memory column is treated as one variant, and its memory comes
    from the matching `... VRAM` column. Either way the result has one row per
    (N, module) with numeric `Mean (ms)` and `Peak VRAM (GB)` columns.
    """
    df = _read_raw(path)
    cols = list(df.columns)
    if "Module" in cols:
        out = df[["N", "Module", "Mean (ms)", "Peak VRAM (GB)"]].copy()
    else:
        vram_cols = [c for c in cols if "vram" in c.lower()]
        modules = [c for c in cols if c != "N" and c not in vram_cols]
        records = []
        for _, row in df.iterrows():
            for mod in modules:
                vram_col = next((c for c in vram_cols if c.lower().startswith(mod.lower())), None)
                records.append({
                    "N": row["N"],
                    "Module": mod,
                    "Mean (ms)": row[mod],
                    "Peak VRAM (GB)": row[vram_col] if vram_col else None,
                })
        out = pd.DataFrame(records)
    out["N"] = out["N"].map(lambda x: int(_num(x)))
    out["Mean (ms)"] = out["Mean (ms)"].map(_num)
    out["Peak VRAM (GB)"] = out["Peak VRAM (GB)"].map(_num)
    return out


def ordered_modules(df):
    """Return the module names with fast_trimul first and the rest alphabetical.

    Putting fast_trimul first gives it a fixed, prominent colour across every
    figure, so it is easy to spot in each plot.
    """
    mods = list(dict.fromkeys(df["Module"]))
    fast = [m for m in mods if "fast_trimul" in m.lower()]
    rest = sorted(m for m in mods if m not in fast)
    return fast + rest


def _minor_label(v, _):
    """Label a log-axis minor tick only at the 0.2 and 0.5 points of each decade."""
    if v <= 0:
        return ""
    mantissa = round(v / 10 ** np.floor(np.log10(v)))
    return f"{v:g}" if mantissa in (2, 5) else ""


def plot_metric(df, ycol, ylabel, title, out_path):
    """Draw one metric against N, one line per module, and save it as a PNG.

    Plots the chosen column against sequence length on log-log axes, highlights the
    fast_trimul line, adds fine minor ticks with 0.2 / 0.5 labels so the low range
    is readable, and marks with an "x" any module that ran out of memory before the
    largest N tested. Writes the result to `out_path`.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    all_ns = sorted(df["N"].unique())
    max_n = all_ns[-1]
    oom_seen = False

    for i, mod in enumerate(ordered_modules(df)):
        sub = df[df["Module"] == mod].sort_values("N")
        color = PALETTE[i % len(PALETTE)]
        is_fast = "fast_trimul" in mod.lower()
        ax.plot(sub["N"], sub[ycol],
                marker=MARKERS[i % len(MARKERS)], markersize=3,
                color=color, linewidth=1.6 if is_fast else 1.0, label=mod)
        last_n = int(sub["N"].max())
        if last_n < max_n:                        # the module stopped early -> OOM
            oom_n = min(n for n in all_ns if n > last_n)
            last_y = sub.loc[sub["N"] == last_n, ycol].iloc[0]
            ax.scatter([oom_n], [last_y], marker="x", s=45, linewidths=1.4,
                       color=color, zorder=6)
            oom_seen = True

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length, N")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ticks = [n for n in all_ns if n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)]
    if all_ns[-1] not in ticks:                   # always label the largest N tested
        ticks.append(all_ns[-1])
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plain)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.get_yaxis().set_major_formatter(plain)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.yaxis.set_minor_formatter(FuncFormatter(_minor_label))
    ax.tick_params(axis="y", which="minor", labelsize=4.5)
    ax.tick_params(axis="x", rotation=45)

    ax.grid(True, which="major", lw=0.4, color="0.85")
    ax.grid(True, which="minor", axis="y", lw=0.3, color="0.93")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if oom_seen:
        handles.append(Line2D([0], [0], marker="x", color="0.35", linestyle="none",
                              markersize=6, markeredgewidth=1.4))
        labels.append("out of memory")
    ax.legend(handles, labels, loc="upper left", frameon=False,
              handlelength=1.6, labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_path.relative_to(HERE))


def main():
    """Find every results_*.md file and write its speed and VRAM figures."""
    FIG_DIR.mkdir(exist_ok=True)
    md_files = sorted(RESULTS_DIR.glob("results_*.md"))
    if not md_files:
        print("no results_*.md files found in", RESULTS_DIR)
        return
    for md in md_files:
        stack = md.stem.replace("results_", "")
        df = load_long(md)
        overrides = LABEL_OVERRIDES.get(md.stem)
        if overrides:
            df["Module"] = df["Module"].replace(overrides)
        title = TITLE_OVERRIDES.get(stack, stack)
        plot_metric(df, "Mean (ms)", "Latency (ms)", title,
                    FIG_DIR / f"{stack}_speed_vs_N.png")
        plot_metric(df, "Peak VRAM (GB)", "Peak memory (GB)", title,
                    FIG_DIR / f"{stack}_vram_vs_N.png")


if __name__ == "__main__":
    main()
