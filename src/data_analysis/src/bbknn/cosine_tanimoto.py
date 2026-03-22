"""
Cosine vs Tanimoto - correlation analysis & “disagreement” plots
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import scienceplots
from rich.console import Console

from .constants import FIG_DIR, ENAMINE_CLASS, D4_CLASS, EGFR_CLASS
from .utils import _pair_grid, load_data, _tight_save

console = Console()

# plt.style.use(["science", "nature"])
# plt.rcParams.update({
#     "font.size": 10, 
#     "axes.labelsize": 12, 
#     "axes.titlesize": 14,
#     "xtick.labelsize": 10, 
#     "ytick.labelsize": 10, 
#     "legend.fontsize": 10,
# })

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


# ╭────────────────────────── plotting helpers ───────────────────────────────╮
def plot_overview_panel(df: pd.DataFrame,
                        q_cos, 
                        q_tan,
                        p_total,
                        s_total,
                        *, outfile="overview_panel.png") -> None:
    """Create 2x2 metrics panel and save to FIG_DIR."""
    fig, ax = plt.subplots(2, 2, figsize=(12*2, 10*2))

    # (A) scatter
    sub = df[df.tanimoto_similarity < 1.0].reset_index(drop=True)
    subsets = [
        sub[sub.chemical_class == ENAMINE_CLASS],
        sub[sub.chemical_class == D4_CLASS],
        sub[~(sub.chemical_class.isin([ENAMINE_CLASS, D4_CLASS, EGFR_CLASS]))]
    ]

    sub = pd.concat([
        subset.sample(n=min(250_000, subset.shape[0]), random_state=0)
        for subset in subsets 
    ]).reset_index(drop=True)

    ax[0, 0].scatter(sub.cosine_similarity, sub.tanimoto_similarity,
                     s=.8, alpha=.1)
    ax[0, 0].set_xlabel("Cosine Similarity")
    ax[0, 0].set_ylabel("Tanimoto Similarity")
    ax[0, 0].set_title("Cosine-Tanimoto Scatter")

    txt = f"Pearson: {p_total:.5f}\nSpearman: {s_total:.5f}"
    ax[0, 0].text(0.02, 0.95, txt, transform=ax[0, 0].transAxes,
                          ha="left", va="top", fontsize=16, fontweight="bold")

    # (B) cumulative curves
    tmp = df.sort_values("cos_quantile")[["cos_quantile", "cosine_similarity"]]
    ax[0, 1].plot(tmp.cos_quantile, tmp.cosine_similarity, label="Cosine")
    tmp = df.sort_values("tani_quantile")[["tani_quantile", "tanimoto_similarity"]]
    ax[0, 1].plot(tmp.tani_quantile, tmp.tanimoto_similarity, label="Tanimoto")
    ax[0, 1].set_xlabel("Quantile"); ax[0, 1].set_ylabel("Metric Value")
    ax[0, 1].set_title("Metric Quantiles"); ax[0, 1].legend()

    # (C) correlation vs cosine cutoff
    ax[1, 0].plot(*q_cos[:2], label="Pearson")
    ax[1, 0].plot(q_cos[0], q_cos[2], label="Spearman")
    ax[1, 0].set_xlabel("Cosine-quantile cutoff"); ax[1, 0].set_ylabel("Correlation")
    ax[1, 0].set_title("Correlation by Cosine Cutoff"); ax[1, 0].legend()

    # (D) correlation vs Tanimoto cutoff
    ax[1, 1].plot(*q_tan[:2], label="Pearson")
    ax[1, 1].plot(q_tan[0], q_tan[2],  label="Spearman")
    ax[1, 1].set_xlabel("Tanimoto-quantile cutoff"); ax[1, 1].set_ylabel("Correlation")
    ax[1, 1].set_title("Correlation by Tanimoto Cutoff"); ax[1, 1].legend()

    # add subplot labels
    labels = ["a", "b", "c", "d"]
    for axi, label in zip(ax.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "cosine_tanimoto" / outfile
    fig.savefig(out, dpi=300)

    tags = ["scatter", "quantiles", "corr_cos", "corr_tani"]
    for a, tag in zip(ax.flatten(), tags):
        _tight_save(a, FIG_DIR/ "cosine_tanimoto" / f"overview_{tag}.png")

    plt.close(fig)
    console.log(f"[green]✓ wrote {out}")

def plot_disagreement_grids(df: pd.DataFrame,
                            rows: int, cols: int) -> None:
    """
    Draw two molecule-pair grids:

    • high-cosine / low-tanimoto
    • low-cosine  / high-tanimoto
    """
    # purely for plotting aesthetics 
    clean = df[df["query"].map(lambda x: "C(C)(C)" not in x)].reset_index()

    hi_cos_low = clean[
        clean.cosine_similarity.between(0.90, 0.999) &
        clean.tanimoto_similarity.between(0.00, 0.40)
    ]
    low_cos_hi = clean[
        clean.cosine_similarity.between(0.00, 0.75) &
        clean.tanimoto_similarity.between(0.65, 1.01)
    ]
    print(hi_cos_low.shape)
    print(low_cos_hi.shape)

    console.log("[cyan]building disagreement grids")
    _pair_grid(hi_cos_low, rows, cols, "high_cos_low_tani", save_dir="cosine_tanimoto")
    _pair_grid(low_cos_hi, rows, cols, "low_cos_high_tani", save_dir="cosine_tanimoto")

# ╭────────────────────────────── CLI driver ──────────────────────────────────╮
def analyze(latent_size=256, cuts=400, rows=5, cols=4):
    out_path = FIG_DIR / "cosine_tanimoto"
    out_path.mkdir(parents=True, exist_ok=True)
    df, (q_cos, q_tan, p_total, s_total) = load_data(latent_size, cuts)
    plot_overview_panel(df, q_cos, q_tan, p_total, s_total)

    plot_disagreement_grids(df, rows, cols)
    console.log("[bold green]analysis complete")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cuts", type=int, default=400)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    analyze(args.size, args.cuts, args.rows, args.cols)

if __name__ == "__main__":
    main()

