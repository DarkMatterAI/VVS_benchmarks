"""
BB-KNN sampling-depth / class-similarity plots
──────────────────────────────────────────────
 * cumulative result count & running-mean similarities
 * out-of-distribution class comparison (Top-10 vs mean)
 * example natural-product grid
"""

from __future__ import annotations
import argparse, math, os
from pathlib import Path
import matplotlib as mpl
import numpy as np, pandas as pd, matplotlib.pyplot as plt #, scienceplots
from rich.console import Console
from .constants import (RAW_DIR, 
                        FIG_DIR,
                        ENAMINE_CLASS,
                        D4_CLASS,
                        EGFR_CLASS)
from .utils import _pair_grid, load_data, _cc_labels, _tight_save

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

# ─────────────────────────── helpers ────────────────────────────────

def summarise_bbknn(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised running-mean + % recovered + cumulative counts."""
    df = df.copy().sort_values(['in_size', 'max_rank'])
    g  = df.groupby('in_size', group_keys=False)
    df['rows']      = g.cumcount() + 1
    df['mean_cos']  = g['cosine_similarity'].cumsum()  / df['rows']
    df['mean_tani'] = g['tanimoto_similarity'].cumsum() / df['rows']

    # “recovered” definition = exact Tanimoto 1.0 hit
    df_s          = df.sort_values('max_rank')
    df_s['cum_rec'] = (df_s.groupby(['in_size','query'])['recovered']
                         .cummax())
    pct = (df_s.groupby(['in_size','max_rank','query'], as_index=False)['cum_rec']
                .max()
                .groupby(['in_size','max_rank'])['cum_rec']
                .mean()
                .rename('pct_recovered'))
    out = (df.groupby(['in_size','max_rank'])
             .agg(mean_cos=('mean_cos','last'),
                  mean_tani=('mean_tani','last'))
           ).join(pct).reset_index()

    # cumulative number of results
    n_res = (df.groupby(['in_size','max_rank','query'])
               .size()
               .reset_index(name='n'))
    n_res = (n_res.groupby(['in_size','max_rank'])['n']
                  .mean()
                  .reset_index())
    n_res['cum_results'] = n_res.groupby('in_size')['n'].cumsum()

    return (out.set_index(['in_size','max_rank'])
               .join(n_res.set_index(['in_size','max_rank']))
               .reset_index())

# ─────────────────────────── plotting ───────────────────────────────

def plot_sampling_curves(df: pd.DataFrame, latent_size: int):
    """
    Two-panel figure
      (A) cumulative results vs k_bb  + quadratic fit
      (B) running mean similarities vs sampling depth
    """
    # ────────────────── summarise & prepare data ───────────────────
    summary = (summarise_bbknn(
                  df[df.chemical_class == ENAMINE_CLASS]
                    .assign(recovered = df.tanimoto_similarity == 1.0))
               .sort_values("max_rank"))
    k_vals        = summary.max_rank + 1
    cum_results   = summary.cum_results

    # ────────────────── quadratic fit  y = ax² + bx + c ─────────────
    coeffs        = np.polyfit(k_vals, cum_results, deg=2)   # returns (a,b,c)
    a, b, c       = coeffs
    a = round(a, 1)
    fit_y         = np.polyval((a,0,0), k_vals)
    # fit_y         = np.polyval((a,0,0), k_vals)

    # ────────────────── plotting  ──────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12*2, 5*2))

    # (A) cumulative results  + fitted curve
    ax[0].plot(k_vals, cum_results, marker="o", label="Observed")
    ax[0].plot(k_vals, fit_y,  ls="--", # color="#FF2C00",
               label="Quadratic fit")
    ax[0].set_xticks(k_vals)
    ax[0].set_xlabel("Query $k_{bb}$ Value")
    ax[0].set_ylabel("Total Results")
    ax[0].set_title("Cumulative Results")
    ax[0].legend()

    # annotate equation in top-left corner
    # eqn = (r"$y = {:.2f}x^2 {:+.2f}x {:+.1f}$"
    #        .format(a, b, c))
    eqn = r"$y = {:.2f}x^2$".format(a)
    ax[0].text(0.03, 0.93, eqn, transform=ax[0].transAxes,
               fontsize=16, va="top", ha="left",
               bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # (B) running mean similarities
    ax[1].plot(cum_results, summary.mean_cos,  label="Cosine")
    ax[1].plot(cum_results, summary.mean_tani, label="Tanimoto")
    ax[1].set_xlabel("Sampling Depth (Number of Results)")
    ax[1].set_ylabel("Average Similarity to Query")
    ax[1].set_title("Query → Result Similarity vs Depth")
    ax[1].legend()

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(ax.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    # ────────────────── save  ──────────────────────────────────────
    fig.tight_layout()
    panel_out = FIG_DIR / "sampling" / "sampling_depth_panel.png"
    fig.savefig(panel_out, dpi=300)
    _tight_save(ax[0], FIG_DIR / "sampling" / "sampling_cumulative_results.png")
    _tight_save(ax[1], FIG_DIR / "sampling" / "sampling_similarity_vs_depth.png")

    # raw CSV (unchanged)
    (RAW_DIR / "bbknn_summary.csv").write_text(summary.to_csv(index=False))

    console.log(f"[green]✓ wrote {panel_out}")


def plot_class_similarity(df: pd.DataFrame, max_rank: int=10, k: int = 10):
    """Bar-chart comparing NatProd classes vs Enamine (Top-k & mean)."""

    df = df[(df.max_rank<max_rank) & 
            (df.chemical_class!=EGFR_CLASS)
            ].reset_index(drop=True)
    df["recovered"] = df.tanimoto_similarity==1.0

    sim_means = (df.groupby("chemical_class")
                 [["cosine_similarity"]]
                 .mean())
    
    sim_topk = (df.groupby(["chemical_class", "query"])
                ["cosine_similarity"]
                .nlargest(k)
                .reset_index()
                .groupby("chemical_class")
                [["cosine_similarity"]].mean())

    recovered_query = (df.groupby(["chemical_class", "query"])
                       ["recovered"]
                       .max()
                       .reset_index()
                       .groupby("chemical_class")
                       [["recovered"]]
                       .mean())
    
    plot_vals = (sim_topk
                 .join(sim_means, lsuffix="_topk", rsuffix="_mean")
                 .join(recovered_query)
                 .sort_values("cosine_similarity_topk", ascending=False))

    # tidy labels
    labels = _cc_labels(plot_vals.index, newline=True)

    fig, axes = plt.subplots(2,1, figsize=(16*2,10*2))

    axes[0].tick_params(axis='x', labelsize=21)
    axes[0].bar(labels, plot_vals.cosine_similarity_topk, label="Top-10")
    for i, v in enumerate(plot_vals.cosine_similarity_topk):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=22)

    axes[1].tick_params(axis='x', labelsize=21)
    axes[0].bar(labels, plot_vals.cosine_similarity_mean, label="Mean")
    for i, v in enumerate(plot_vals.cosine_similarity_mean):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=22)

    axes[0].set_ylim(0,1)
    axes[0].set_ylabel("Cosine Similarity", fontsize=30)
    axes[0].tick_params(axis='y', labelsize=28)
    axes[0].set_title("O.O.D. Class Similarity ($k_{bb}$=10)", fontsize=36)
    axes[0].legend()

    axes[1].bar(labels, plot_vals.recovered, color=palette_cb_ext[3])
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel("Percent Queries Recovered", fontsize=30)
    axes[1].tick_params(axis='y', labelsize=28)
    axes[1].set_title("O.O.D. Query Recovery ($k_{bb}$=10)", fontsize=36)

    for i, v in enumerate(plot_vals.recovered):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=22)

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(axes.flatten(), labels):
        axi.text(-0.03, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=32, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "sampling" / "class_similarity_bar.png"
    fig.savefig(out, dpi=300)
    _tight_save(axes[0], FIG_DIR / "sampling" / f"class_similarity_bar_sim.png")
    _tight_save(axes[1], FIG_DIR / "sampling" / f"class_similarity_bar_rec.png")

    plt.close(fig)
    df_path   = RAW_DIR / f"ood_similarity.csv"
    plot_vals.to_csv(df_path)
    console.log(f"[green]✓ wrote {out}")

def plot_natprod_grid(df: pd.DataFrame, rows: int=1, cols: int=1):

    sub = df[~df.chemical_class.isin([ENAMINE_CLASS,D4_CLASS,EGFR_CLASS])].copy()
    sub = sub.loc[sub.groupby("query")["cosine_similarity"].nlargest(3).reset_index().level_1]
    _pair_grid(sub, rows, cols, "natprod_example", label_by_class=True, save_dir="sampling")

    sub = df[df.chemical_class.isin([ENAMINE_CLASS,D4_CLASS])].copy()
    sub = sub.loc[sub.groupby("query")["cosine_similarity"].nlargest(3).reset_index().level_1]
    _pair_grid(sub, rows, cols, "enamine_d4_example", label_by_class=True, save_dir="sampling")

def plot_query_failure_by_k(df: pd.DataFrame, 
                            max_rank: int=10,
                            y_max: float = 0.20,
                            break_lo: float = 0.001,
                            break_hi: float = 0.002):
    """Per-class line plot: fraction of queries returning zero results vs k_bb."""

    classes = sorted(df["chemical_class"].unique())
    classes = [i for i in classes if i!=EGFR_CLASS]
    k_vals  = list(range(max_rank))

    records = []
    for cc in classes:
        df_cc     = df[df.chemical_class == cc]
        n_queries = df_cc["query"].nunique()
        for k in k_vals:
            n_successful = df_cc[df_cc.max_rank <= k]["query"].nunique()
            n_failed     = n_queries - n_successful
            records.append({
                "chemical_class": cc,
                "k_bb":           k + 1,          # display 1-indexed
                "n_queries":      n_queries,
                "n_successful":   n_successful,
                "n_failed":       n_failed,
                "frac_failed":    n_failed / n_queries,
            })

    result_df = pd.DataFrame(records)

    # ── raw data ─────────────────────────────────────────────────────
    raw_out = RAW_DIR / "query_failure_by_k.csv"
    result_df.to_csv(raw_out, index=False)
    console.log(f"[green]✓ wrote {raw_out}")

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    for cc in classes:
        cc_data = result_df[result_df.chemical_class == cc].sort_values("k_bb")
        label   = _cc_labels([cc], newline=False)[0]
        ax.plot(cc_data.k_bb, cc_data.frac_failed, marker="o", label=label)

    ax.set_xticks(np.array(k_vals) + 1)
    ax.set_xlabel("$k_{bb}$ Value")
    ax.set_ylabel("Fraction of Queries Without Results")
    ax.set_title("Query Failure Rate vs $k_{bb}$")
    ax.set_ylim(0, 0.2)
    ax.legend()

    fig.tight_layout()
    _tight_save(ax, FIG_DIR / "sampling" / "query_failure_by_k_tight.png")
    plt.close(fig)
    console.log(f"[green]✓ wrote failure by k figure")


def analyze(latent_size=256, cuts=200, rows=1, cols=1):
    console.rule("[bold]BB-KNN sampling-depth analysis")
    out_path = FIG_DIR / "sampling"
    out_path.mkdir(parents=True, exist_ok=True)
    df, _ = load_data(latent_size, cuts)

    plot_sampling_curves(df, latent_size)
    plot_class_similarity(df)
    plot_natprod_grid(df, rows, cols)
    plot_query_failure_by_k(df)
    console.rule("[bold green]ALL DONE")

# ─────────────────────────── CLI ─────────────────────────────────────-
def main(latent_size=256, cuts=400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cuts", type=int, default=400)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    analyze(args.size, args.cuts, args.rows, args.cols)

if __name__ == "__main__":
    main()

