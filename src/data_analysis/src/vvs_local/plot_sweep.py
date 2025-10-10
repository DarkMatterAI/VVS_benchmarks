"""
VVS learning-rate sweep ⇢ similarity / overlap / score analysis
────────────────────────────────────────────────────────────────

Outputs
-------
* vvs_panel_2up.png      - similarity-curves + LR-overlap heat-map
* vvs_boxplots.png       - update-type & gradient-step score comparison
* vvs_repeat_hist.png    - duplicate-SMILES distribution
* every subplot saved individually to FIG_DIR/raw/ for the paper
"""

from __future__ import annotations
import argparse, json, re, multiprocessing as mp
from ast import literal_eval
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import scienceplots
from rich.console import Console

from .constants import DATA_DIR, BENCH_DIR, ENGINE, FIG_DIR, RAW_DIR, COLORS

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


# ╭───────────────────────────── helpers (generic) ───────────────────────────╮
def _tight_save(ax: plt.Axes, fname: str):
    """Save *one* axis tightly cropped → FIG_DIR/raw/"""
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()
                            ).transformed(fig.dpi_scale_trans.inverted()
                            ).expanded(1.03, 1.05)
    fig.savefig(RAW_DIR / fname, dpi=300, bbox_inches=bbox)


# ╭────────────────────────── Stage-0 • load sweep data ───────────────────────╮
def load_sweep(engine: str = ENGINE
               ) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns
    -------
    sweep_data[score_name]["stats_df" | "prod_df"]  → DataFrame
    """
    sweep_data = {}
    root = DATA_DIR / engine
    for score_dir in root.iterdir():
        if not score_dir.is_dir():
            continue
        try:
            stats = pd.read_parquet(score_dir / "stats_df.parquet")
            prod  = pd.read_parquet(score_dir / "prod_df.parquet")
        except FileNotFoundError:
            continue
        sweep_data[score_dir.name] = {"stats_df": stats, "prod_df": prod}
    console.log(f"[cyan]Loaded {len(sweep_data)} sweep-targets from {root}")
    return sweep_data


# ╭────────────────────────── Stage-1 • similarity curves ─────────────────────╮
def build_similarity_df(sweep: dict, k: int = 10) -> pd.DataFrame:
    """Return tidy DF for lr-vs-cosine curves at a fixed bb-k."""
    dfs = []
    for score, d in sweep.items():
        df = d["stats_df"]
        sel = df[(df.level == "orig_query") & (df.k == k)].copy()
        sel["score_name"] = score
        dfs.append(sel)
    return pd.concat(dfs, ignore_index=True)


# ╭──────────────────── Stage-2 • LR-overlap heat-maps (fast) ─────────────────╮
def _heatmap_for_idx(idx: int,
                     lr_map: Dict[float, set[str]]
                     ) -> Tuple[np.ndarray, List[float]]:
    """Overlap matrix (sym.) for *one* orig_idx."""
    lrs  = np.array(sorted(lr_map))      # deterministic order
    sets = [lr_map[lr] for lr in lrs]
    n    = len(lrs)

    mat = np.ones((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        s1, s2 = sets[i], sets[j]
        mat[i, j] = mat[j, i] = len(s1 & s2) / len(s1) if s1 else 0.0
    return mat, lrs


def collect_heatmaps(sweep: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Returns
    -------
    heatmaps : stacked array  [N_idx, N_lr, N_lr]
    lr_lbls  : ordered label list (shared across all matrices)
    """
    # --- build {orig_idx: {lr: set(results)}} dictionary --------------------
    by_idx: dict[int, Dict[float, set[str]]] = defaultdict(dict)
    for d in sweep.values():
        prod = d["prod_df"]
        grouped = (prod.groupby(["orig_idx", "lr"], sort=False)["result"].agg(set))
        for (idx, lr), res_set in grouped.items():
            by_idx[idx][lr] = res_set

    # --- per-idx matrices (multiprocess → ~10x faster) ----------------------
    with mp.Pool(max(mp.cpu_count() - 2, 2)) as pool:
        mats = pool.starmap(_heatmap_for_idx, by_idx.items())

    # filter for fully-populated matrices (some idx may be missing LR points)
    good = [(m, lr) for m, lr in mats if m.shape[0] == len(m[0])]
    if not good:
        raise RuntimeError("No complete heat-maps found - check input data.")

    heatmaps, lr_lists = zip(*good)
    lr_lbls = [str(int(x)) for x in lr_lists[0]]
    console.log(f"[cyan]Collected {len(heatmaps)} complete heat-maps")
    return np.stack(heatmaps), lr_lbls


# ╭────────────────────────── Stage-3 • score aggregation ─────────────────────╮
def _parse_run_name(name: str) -> Tuple[str, str, str]:
    """<base>_<gridId>-<scoreId>_<replica> → (base, gridId, replica)."""
    m = re.match(r"(.+?)_(\d{2,3}-\d{2})_(\d+)$", name)
    return (m.group(1), m.group(2), m.group(3)) if m else (name, "??", "??")


def load_score_runs(run_root: Path,
                    topk: List[int],
                    trunc: int = 50_000
                    ) -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Parse every run directory under *run_root* and return:
      • summary DF (one row / run replica)
      • dfs[run_name] = full score_log DF
    """
    records, dfs = [], {}
    for rd in sorted(run_root.iterdir()):
        par_p = rd / "params.json"
        log_p = rd / "score_log.csv"
        if not (par_p.exists() and log_p.exists()):
            continue

        prm = json.loads(par_p.read_text())
        df  = (pd.read_csv(log_p)
                 .assign(ts=lambda d: pd.to_datetime(d["ts"], format='mixed'))
                 .sort_values(["ts", "score"], ascending=[True, False])
                 .head(trunc))
        if df.empty:
            continue

        rec = prm.copy()
        for k in topk:
            rec[f"top{k}"] = df["score"].nlargest(k).mean()
        rec["runtime"] = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds()
        base, gid, rep = _parse_run_name(rd.name)
        rec.update(dict(run_base=base, grid_id=gid, replica=rep,
                        n_eval=df.shape[0]))
        records.append(rec)
        dfs[rd.name] = df

    summ = pd.DataFrame(records)
    summ["lrs"] = summ["lrs"].map(lambda x: tuple(x))
    console.log(f"[cyan]Parsed {len(summ)} run replicas from {run_root}")
    return summ, dfs


# ╭──────────────────────────── plotting core ─────────────────────────────────╮
def similarity_panel(sim_df: pd.DataFrame,
                     heatmaps: np.ndarray,
                     lr_lbls: List[str]):
    """2-up figure: similarity curves + median heatmap."""
    fig, ax = plt.subplots(1, 2, figsize=(12*2, 5*2))

    # left - cosine curves ---------------------------------------------------
    sns.lineplot(sim_df, x="lr", y="cos_q_mean", ax=ax[0],
                 label="Result → Original Query")
    sns.lineplot(sim_df, x="lr", y="cos_g_mean", ax=ax[0],
                 label="Result → Gradient Query")
    ax[0].set_xlabel("Learning Rate")
    ax[0].set_ylabel("Cosine Similarity")
    ax[0].set_title("Query vs Result Similarity")
    ax[0].legend()
    _tight_save(ax[0], "sim_lr_curves.png")

    # right - LR overlap heat-map -------------------------------------------
    skip = 5                                     # show every 5th tick label
    med = np.median(heatmaps, axis=0)
    sns.heatmap(med, ax=ax[1], cmap="viridis")
    ticks = np.arange(0, len(lr_lbls), skip)
    ax[1].set_xticks(ticks); ax[1].set_yticks(ticks)
    ax[1].set_xticklabels(lr_lbls[::skip], rotation=45, ha="right")
    ax[1].set_yticklabels(lr_lbls[::skip])
    ax[1].set_xlabel("Learning Rate"); ax[1].set_ylabel("Learning Rate")
    ax[1].set_title("Median Result-Set Overlap Fraction")
    _tight_save(ax[1], "sim_lr_heatmap.png")

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(ax.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "vvs_panel_2up.png", dpi=300)
    console.log(f"[green]✓ wrote {FIG_DIR/'vvs_panel_2up.png'}")
    plt.close(fig)


def boxplot_panel(summ_df: pd.DataFrame, topk: list[int]):
    """Box-plots: (A) update-type   (B) gradient-step count."""
    df = summ_df.copy()

    # clean up LR tuples -----------------------------------------------------
    df["lr_agg"] = df["lrs"].map(lambda t: tuple(i for i in t if i > 0))

    # aggregate replica-means -----------------------------------------------
    agg_cols = ["run_base", "grid_id"]
    kcols = [f"top{k}" for k in topk] + ["runtime"]
    skip_cols = ["db_path", "db_table", "engine_params", 
                 "plugin", "timeout", "score_batch_size",
                 "concurrency", "device", "rng_seed", "replica", "n_eval"]
    remaining = [i for i in df.columns if i not in kcols+skip_cols]
    by_rep = (df[remaining]
              .drop_duplicates(agg_cols)
              .merge(df.groupby(agg_cols)
                     [kcols].mean()
                     .reset_index(),
                     on=agg_cols))
    by_rep["min_lr"] = by_rep["lr_agg"].map(min)
    by_rep["max_lr"] = by_rep["lr_agg"].map(max)
    by_rep["n_lrs"]  = by_rep["lr_agg"].map(len)

    tidy = []
    for k in topk:
        sub = by_rep[[f"top{k}", "update_type", "n_lrs",
                      "min_lr", "max_lr", "exploit_percent"]].copy()
        sub.columns = ["score", "update_type", "n_lrs", "min_lr", 
                       "max_lr", "exploit_percent"]
        sub["k"] = k
        tidy.append(sub)
    tidy = pd.concat(tidy, ignore_index=True)
    tidy["n_lrs_cat"] = tidy["n_lrs"].astype(str)
    tidy["exploit_percent_cat"] = tidy["exploit_percent"].map(lambda x: f"{x:.0%}")
    tidy["max_lr_cat"] = tidy["max_lr"].astype(str)

    # --- plotting -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12*2, 10*2))

    sns.boxplot(data=tidy[(tidy.min_lr >= 1000) & (tidy.max_lr <= 50000)],
                x="k", y="score", hue="update_type", ax=ax[0,0])
    ax[0,0].set_xlabel("Top-K"); ax[0,0].set_ylabel("Average Score")
    ax[0,0].set_title("Standard vs Top-1 Update")
    h, l = ax[0,0].get_legend_handles_labels()
    ax[0,0].legend(handles=h, labels=["Standard", "Top-1"], title="Update Type")
    _tight_save(ax[0,0], "box_update_type.png")

    sns.boxplot(data=tidy[(tidy.min_lr >= 1000) & (tidy.max_lr <= 50000) &
                          (tidy.n_lrs.isin([1, 5]))],
                x="k", y="score", hue="n_lrs_cat",
                hue_order=["1", "5"], ax=ax[0,1])
    ax[0,1].set_xlabel("Top-K"); ax[0,1].set_ylabel("Average Score")
    ax[0,1].set_title("Number of Gradient Steps")
    h, l = ax[0,1].get_legend_handles_labels()
    ax[0,1].legend(handles=h, labels=["1", "5"], title="Gradient Steps", loc="lower left")
    _tight_save(ax[0,1], "box_n_lrs.png")

    sns.boxplot(data=tidy[(tidy.min_lr >= 1000) & (tidy.max_lr <= 50000) &
                          (tidy.n_lrs.isin([1, 5]))],
                x="k", y="score", hue="exploit_percent_cat",
                 ax=ax[1,0])
    ax[1,0].set_xlabel("Top-K"); ax[1,0].set_ylabel("Average Score")
    ax[1,0].set_title("Exploit Percent")
    h, l = ax[1,0].get_legend_handles_labels()
    ax[1,0].legend(handles=h,  title="Exploit Percent", loc="lower left")
    _tight_save(ax[1,0], "box_pct_exploit.png")

    sns.boxplot(data=tidy[(tidy.min_lr >= 1000) & 
                          (tidy.max_lr.isin([1000,3000,5000,10000]))],
                x="k", y="score", hue="max_lr_cat",
                hue_order=["1000", "3000", "5000", "10000"], ax=ax[1,1])
    ax[1,1].set_xlabel("Top-K"); ax[1,1].set_ylabel("Average Score")
    ax[1,1].set_title("Maximum Learning Rate")
    h, l = ax[1,1].get_legend_handles_labels()
    ax[1,1].legend(handles=h, title="Max Learning Rate", loc="lower left")
    _tight_save(ax[1,1], "box_max_lr.png")

    # add subplot labels
    labels = ["a", "b", "c", "d"]
    for axi, label in zip(ax.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "vvs_boxplots.png", dpi=300)
    console.log(f"[green]✓ wrote {FIG_DIR/'vvs_boxplots.png'}")
    plt.close(fig)

def repeat_histogram(dfs: dict[str, pd.DataFrame]):
    """Histogram of repeated SMILES counts + score trend."""
    stacked = pd.concat(dfs.values(), ignore_index=True, copy=False)
    rep = (stacked[["item", "score"]]
           .groupby("item").agg(score=("score", "mean"),
                                size=("score", "size"))
           .reset_index())

    def _bin(n):
        if n <= 3: return str(n)
        if n <= 10: return "4-10"
        return "11+"
    rep["bin"] = rep["size"].map(_bin)

    cnt = (rep.groupby("bin")
           .agg(score_avg=("score", "mean"),
                score_std=("score", "std"),
                count=("size", "size"))
           .reset_index()
           .sort_values("count", ascending=False))
    cnt["count_pct"] = cnt["count"] / cnt["count"].sum()

    # --- plot ---------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6*2, 5*2))
    ax1.bar(cnt["bin"], cnt["count_pct"])
    ax1.set_xlabel("Repeat Count Bin"); ax1.set_ylabel("Frequency")
    ax1.set_title("VVS Search Result Repeat Frequency")

    for i, v in enumerate(cnt["count_pct"]):
        ax1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=16)

    # ax2 = ax1.twinx()
    # ax2.plot(cnt["bin"], cnt["score_avg"],
    #          color=COLORS["red"], marker="o", label="Avg Score")
    # ax2.set_ylabel("Average Score"); ax2.set_ylim(0, 10)

    _tight_save(ax1, "repeat_hist_bar.png")      # only need one axis (twinned)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vvs_repeat_hist.png", dpi=300)
    console.log(f"[green]✓ wrote {FIG_DIR/'vvs_repeat_hist.png'}")
    plt.close(fig)

    # write raw CSVs ---------------------------------------------------------
    cnt.to_csv(RAW_DIR / "repeat_histogram.csv", index=False)


# ╭────────────────────────────── CLI driver ──────────────────────────────────╮
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    # 1) similarity panel ----------------------------------------------------
    sweep  = load_sweep()
    sim_df = build_similarity_df(sweep, k=10)
    heats, lr_lbls = collect_heatmaps(sweep)
    similarity_panel(sim_df, heats, lr_lbls)

    # 2) update-type / gradient-step box-plots -------------------------------
    topk = [1, 5, 10, 100, 1000]
    summ_df, dfs = load_score_runs(BENCH_DIR, topk)
    boxplot_panel(summ_df, topk)

    # 3) repeat histogram ----------------------------------------------------
    repeat_histogram(dfs)

    console.rule("[bold green]ALL DONE")


if __name__ == "__main__":
    main()
