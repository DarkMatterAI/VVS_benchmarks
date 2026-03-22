"""
EGFR docking benchmark - BB-KNN sampling curves + molecule grid
────────────────────────────────────────────────────────────────
* 50 best queries by docking score
* running averages & Top-K curves
* molecule grid for high-improvement examples
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# import scienceplots
from rich.console import Console

from .constants import FIG_DIR, RAW_DIR, ENAMINE_CLASS, EGFR_CLASS
from .utils      import _pair_grid, load_data, _tight_save

console = Console()
# plt.style.use(["science","nature"])
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

# ─────────────────────────────────────────────────────────────────────────────
def _build_curves(df: pd.DataFrame, k_top: int) -> pd.DataFrame:
    """Faithful port of the scratch loop - keeps per-rank running stats."""
    cache = RAW_DIR / f"egfr_curves_k{k_top}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    egfr_data_list = []
    for rank in range(df.max_rank.max() + 1):
        egfr_data = {}
        subset = df[df.max_rank <= rank]

        egfr_data["k"] = rank
        egfr_data["num_results"] = (
            subset.shape[0] / subset["query"].nunique()
        )

        egfr_data["cos_avg"]  = subset["cosine_similarity"].mean()
        egfr_data["tani_avg"] = subset["tanimoto_similarity"].mean()

        egfr_data["pct_chg_avg"]     = subset["pct_change"].mean()
        egfr_data["pct_chg_avg_std"] = subset["pct_change"].std()

        topk_subset = (
            df[df.max_rank <= rank]
            .groupby("query")["pct_change"]
            .nlargest(k_top)
            .reset_index()
        )
        topk_subset = df.loc[topk_subset.level_1]

        egfr_data["cos_topk"]  = topk_subset["cosine_similarity"].mean()
        egfr_data["tani_topk"] = topk_subset["tanimoto_similarity"].mean()
        egfr_data["pct_chg_topk"]     = topk_subset["pct_change"].mean()
        egfr_data["pct_chg_topk_std"] = topk_subset["pct_change"].std()

        egfr_data_list.append(egfr_data)

    res = pd.DataFrame(egfr_data_list)
    res.to_csv(cache, index=False)
    console.log(f"[green]✓ wrote cached curves → {cache.name}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
def build_plots(latent_size: int = 256, 
                cuts: int = 400,
                k_top: int = 10,
                rows: int = 5,
                cols: int = 4) -> None:
    df, _ = load_data(latent_size, cuts)
    df = df[df.chemical_class == EGFR_CLASS].copy()
    df = df[df.result_score > -100].reset_index(drop=True)

    # choose 50 best queries
    best_q = (df[["query", "query_score"]]
              .drop_duplicates()
              .nlargest(50, "query_score")["query"])
    df = df[df["query"].isin(best_q)].reset_index(drop=True)

    df["score_delta"] = df.result_score - df.query_score
    df["pct_change"]  = df.score_delta / df.query_score

    curves = _build_curves(df, k_top)

    # ── panel figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12*2, 5*2))

    # ── (A) Dock-score gain vs sampling depth ─────────────────────────
    ax[0].plot(curves.num_results, curves.pct_chg_avg,
            label="All results")
    ax[0].plot(curves.num_results, curves.pct_chg_topk,
            label=f"Top-{k_top} results")

    ax[0].set_xlabel("Sampling Depth (Results / Query)")
    ax[0].set_ylabel("Percent Dock-Score Improvement")
    ax[0].set_title("Docking-Score Gain vs Sampling Depth")
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax[0].legend()

    # ── (B) Query → result similarity ─────────────────────────────────
    ax[1].plot(curves.num_results, curves.cos_avg,
            label="All results")
    ax[1].plot(curves.num_results, curves.cos_topk,
            label=f"Top-{k_top} results (by dock score)")

    ax[1].set_xlabel("Sampling Depth (Results / Query)")
    ax[1].set_ylabel("Cosine Similarity to Query")
    ax[1].set_title("Query-Result Cosine Similarity vs Depth")
    ax[1].legend()

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(ax.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "egfr" / "egfr_sampling_curves.png", dpi=300)
    _tight_save(ax[0], FIG_DIR / "egfr" / "egfr_dock_gain_vs_depth.png")
    _tight_save(ax[1], FIG_DIR / "egfr" / "egfr_similarity_vs_depth.png")
    plt.close(fig)
    console.log("[green]✓ wrote egfr_sampling_curves.png")

    # ── molecule grid ------------------------------------------------------
    good = (df[df.score_delta > 1]
            .sample(frac=1, random_state=0)
            .groupby("query").head(1))
    _pair_grid(good, rows, cols, "egfr", label_by_score=True, save_dir="egfr")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cuts", type=int, default=400)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    out_path = FIG_DIR / "egfr"
    out_path.mkdir(parents=True, exist_ok=True)
    build_plots(args.size, args.cuts, args.topk, args.rows, args.cols)

if __name__ == "__main__":
    main()
