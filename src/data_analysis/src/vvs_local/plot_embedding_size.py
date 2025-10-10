from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import scienceplots
from rich.console import Console

from .constants import FIG_DIR, RAW_DIR, COLORS, INDEX_PATH

console = Console()

default_font_size = mpl.rcParamsDefault['font.size']

plt.rcParams.update({
    'patch.linewidth': 3,
    'lines.linewidth': 3,
    'lines.markersize': 12,
    
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': False,

    'axes.spines.top': False,
    'axes.spines.right': False,

    'font.size': 2 * default_font_size,

    'figure.figsize': (8, 8),
    
})

palette_cb_ext = ["#ebac23", "#b80058",  "#008cf9", "#006e00", "#d163e6",
                   "#b24502", "#00bbad",
                    "#ff9287", "#5954d6", "#00c6f8",
                    "#878500", "#00a76c", "#979797", "#1e1e1e"]
palette_cb_ext = palette_cb_ext[2:] + palette_cb_ext[:2]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette_cb_ext)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ───────────────────────── helper ───────────────────────────────────
def _tight_save(ax: plt.Axes, fname: str):
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()
                            ).transformed(fig.dpi_scale_trans.inverted()
                            ).expanded(1.02, 1.04)
    fig.savefig(RAW_DIR / fname, dpi=300, bbox_inches=bbox)

# ─────────────────────── data gathering ─────────────────────────────
def collect_stats(sizes: List[int] | None = None) -> pd.DataFrame:
    """
    Crawl INDEX_PATH/<size>/ for build-stats & latency JSONs.

    Returns tidy DataFrame with columns
        dim, mem_gb, build_min, latency_s
    """
    records: List[Dict] = []
    all_dirs = [p for p in INDEX_PATH.iterdir() if p.is_dir()]
    for d in sorted(all_dirs, key=lambda p: int(p.name)):
        dim = int(d.name)
        if sizes and dim not in sizes:
            continue

        bs = d / "build_stats.json"
        if not bs.exists():
            console.log(f"[yellow]Missing build_stats.json in {d} - skipped")
            continue

        with bs.open() as fh:
            bstats = json.load(fh)
        rec = {
            "dim":        dim,
            "mem_gb":     os.path.getsize(d / "index.usearch") / 1e9,
            "build_min":  bstats["build_seconds"] / 60.0,
            "latency_s":  np.nan,            # filled if available
        }

        lat = d / "query_latency.json"
        if lat.exists():
            with lat.open() as fh:
                lstats = json.load(fh)
            rec["latency_s"] = lstats.get("mean_seconds", np.nan)
        else:
            console.log(f"[yellow]No latency stats for {dim}-d")

        records.append(rec)

    if not records:
        raise RuntimeError("No index stats found - run compression_eval first.")
    return pd.DataFrame(records).sort_values("dim")

# ───────────────────────── plotting ─────────────────────────────────
def make_plots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18*2, 5*2))
    sizes = df["dim"].values

    # 1) memory --------------------------------------------------------
    sns.lineplot(df, x="dim", y="mem_gb", marker="o",
                 ax=axes[0], color=COLORS["blue"])
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Index Size (GB)")
    axes[0].set_title("Index Size on Disk")
    axes[0].set_xticks(sizes)
    _tight_save(axes[0], "embed_mem_vs_dim.png")

    # 2) build time ----------------------------------------------------
    sns.lineplot(df, x="dim", y="build_min", marker="o",
                 ax=axes[1], color=COLORS["green"])
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Build Time (min)")
    axes[1].set_title("Index Build Time")
    axes[1].set_xticks(sizes)
    _tight_save(axes[1], "embed_build_vs_dim.png")

    # 3) latency -------------------------------------------------------
    sns.lineplot(df, x="dim", y="latency_s", marker="o",
                 ax=axes[2], color=COLORS["red"])
    axes[2].set_xlabel("Embedding Dimension")
    axes[2].set_ylabel("Query Latency (s)")
    axes[2].set_title("Mean Query Latency")
    axes[2].set_xticks(sizes)
    _tight_save(axes[2], "embed_latency_vs_dim.png")

    # add subplot labels
    labels = ["a", "b", "c"]
    for axi, label in zip(axes.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "embedding_size_panel.png"
    fig.savefig(out, dpi=300)
    console.log(f"[green]✓ wrote {out}")
    plt.close(fig)

# ──────────────────────────── CLI ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    help="Optional subset of dimensions")
    args = ap.parse_args()

    df = collect_stats(args.sizes)
    make_plots(df)
    console.rule("[bold green]ALL DONE")

if __name__ == "__main__":
    main()
