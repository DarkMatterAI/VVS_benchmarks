"""
Analyse *final* benchmark runs (one-shot hyper-params picked previously).

Saves:
  • summary CSV  →  FINAL_DIR/final_results.csv
  • LaTeX table  →  FIG_DIR/raw/final_results_table.tex
  • two 2x2 score curves panels (Inference vs Score, Runtime vs Score)
    with individual subplot PNGs exactly like other modules.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import scienceplots
from rich.console import Console

from .constants import (FINALS, FINAL_DIR, ENUM_FINALS, RANK_KS,
                        BLOB, METHODS, FIG_DIR, RAW_DIR)
from .utils      import parse_run_name, load_one_run, summarise_scores

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
# palette_cb_ext = palette_cb_ext[2:] + palette_cb_ext[:2]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette_cb_ext)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# ╭──────────────────────────────── gather all runs ─────────────────────────╮
def gather_all(final_dirs: dict
               ) -> Tuple[pd.DataFrame, Dict[tuple, dict], Dict[tuple, pd.DataFrame]]:
    """
    Returns
    -------
    summary_df   - one row per (method, run_id, replica)  *and* optional “no-bb”
    params       - keyed by (method, run_id, replica)
    df_store     - same keys, raw score_log DataFrames
    """
    recs, prm_store, df_store = [], {}, {}
    for method, root in final_dirs.items():
        if not root.exists():
            console.log(f"[yellow]⚠  {root} missing - skipped")
            continue
        for run_dir in root.iterdir():
            df, prm = load_one_run(run_dir)
            if df is None:
                continue

            base, gid, rep = parse_run_name(run_dir.name)
            run_id         = f"{base}_{gid}"

            for bb_flag in (False, True):            # collect with & w/o bb rows
                if bb_flag and "is_bb" not in df.columns:
                    continue
                label = method if not bb_flag else method + " (no bb)"
                summ  = summarise_scores(df, strip_bb=bb_flag)
                if "score_cfg" in prm:
                    score_name = prm["score_cfg"]["score_name"]
                else:
                    score_name = prm.get("score_name", prm.get("plugin"))
                summ["score_name"] = score_name

                rec = dict(method=label, run_id=run_id, run_replica=rep)
                rec.update(summ)
                recs.append(rec)

            prm_store[(method, run_id, rep)] = prm
            df_store[(method, run_id, rep)]  = df

    summary = pd.DataFrame(recs)
    console.log(f"[cyan]Loaded {len(summary)} final-run replicas.")
    return summary, prm_store, df_store


# ╭─────────────────────────── make LaTeX table ─────────────────────────────╮
SCORE_LABEL = {
    "docking_2zdt":  r"OpenEye\\Docking 2Zdt",
    "erbb1_mlp":     "EGFR MLP",
    "rocs_2chw":     r"OpenEye\\ROCS 2Chw",
    "synthemol_rf":  r"Antibacterial\\RF",
}
SCORE_PLOT_LABEL = {
    "docking_2zdt": "OpenEye Docking 2Zdt",
    "erbb1_mlp": "EGFR MLP",
    "rocs_2chw": "OpenEye Rocs 2Chw",
    "synthemol_rf": "Antibacterial RF",
}
METHOD_LABEL = {
    "rxnflow":               "RxnFlow",
    "synthemol":             "SyntheMol",
    "synthemol (no bb)":     "SyntheMol (w/o bb)",
    "ts":                    "Thompson Sampling",
    "vvs_local":             "VVS (ours)",
    "rad":                   "RAD"
}

# ╭─────────────────────────── make LaTeX table ─────────────────────────────╮
def make_latex_table(agg: pd.DataFrame,
                     k_vals: list[int],
                     out_file: Path,
                     dataset_name: str,
                     table_name: str,
                     ):
    """
    Write a LaTeX table with bold-faced best Top-K means (higher is
    better) and best runtime (lower is better) for every score group.
    """

    # ------------------------------------------------------------------ prep
    agg = agg.set_index(["score_name", "method"])

    # collect per-score extrema
    extrema = {}
    for score, sub in agg.groupby(level=0):
        extrema[score] = {
            **{f"top{k}": sub[(f"top{k}", "mean")].max() for k in k_vals},
            "runtime":   sub[("runtime", "mean")].min()
        }

    # helper -----------------------------------------------------------------
    def _fmt_cell(mean: float, std: float, bold: bool) -> str:
        if bold:
            cell = f"$ \\textbf{{{mean:.2f}}}\\pm{std:.2f} $"
        else:
            cell = f"$ {mean:.2f}\\pm{std:.2f} $"
        return cell 
        # return f"\\textbf{{{cell}}}" if bold else cell

    # build row strings ------------------------------------------------------
    table_data = defaultdict(list)
    for (score, method), row in agg.iterrows():
        cells = [f"& {METHOD_LABEL[method]}"]

        # Top-K metrics (higher better)
        for k in k_vals:
            mean, std = row[f"top{k}"]["mean"], row[f"top{k}"]["std"]
            best = np.isclose(mean, extrema[score][f"top{k}"])
            cells.append("& " + _fmt_cell(mean, std, best))

        # Runtime  (lower better)
        mean_rt, std_rt = row["runtime"]["mean"], row["runtime"]["std"]
        best_rt = np.isclose(mean_rt, extrema[score]["runtime"])
        cells.append("& " + _fmt_cell(mean_rt, std_rt, best_rt))

        cells.append(r" \\")
        table_data[score].append(" ".join(cells))

    # ------------------------------------------------------------------ latex
    start = r"""\begin{table}[!ht]
\centering
\caption{Results on the \datasetname~ dataset using various scores and search methods. Arrows indicate desired direction of metric improvement.}
\scriptsize
\begin{tabular}{llcccc}
\toprule
Score & Search Method & Top-1 $\uparrow$ &  Top-10 $\uparrow$ & Top-100 $\uparrow$ & Runtime (min) $\downarrow$ \\"""

    end = r"""\bottomrule
\end{tabular}
\label{table:tablename}
\end{table}"""

    start = start.replace("datasetname", dataset_name)
    end = end.replace("tablename", table_name)

    body = ""
    for score, rows in table_data.items():
        # choose \multirowcell vs \multirow exactly like original script
        if score in {"docking_2zdt", "rocs_2chw", "synthemol_rf"}:
            header = ("\midrule\n"
                      rf"\multirowcell{{{len(rows)}}}[0pt][l]{{{SCORE_LABEL[score]}}}")
        else:
            header = ("\midrule\n"
                      rf"\multirow{{{len(rows)}}}{{*}}{{{SCORE_LABEL[score]}}}")
        body += header + "\n\t" + "\n\t".join(rows) + "\n"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(start + "\n" + body + end)
    console.log(f"[green]✓ wrote LaTeX table → {out_file.name}")

# ╭─────────────────────────── plotting helpers ─────────────────────────────╮
def _tight_save(ax: plt.Axes, fname: str):
    """Save *one* axis tightly cropped → FIG_DIR/raw/"""
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()
                            ).transformed(fig.dpi_scale_trans.inverted()
                            ).expanded(1.03, 1.05)
    fig.savefig(RAW_DIR / fname, dpi=300, bbox_inches=bbox)

def build_cut_curves(df_store: Dict[tuple, pd.DataFrame],
                     summary: pd.DataFrame,
                     ) -> pd.DataFrame:
    """
    Returns a long-format DataFrame with columns
    [method, score, cut, k, score_at_k, time_min]
    (k limited to 1 / 10 / 100)
    """
    rows, CUTS = [], [100, 500, 1_000, 5_000, 10_000,
                      15_000, 20_000, 30_000, 40_000, 50_000]
    
    key_to_score = {}
    for i, row in summary.iterrows():
        key = (row["method"], row["run_id"], row["run_replica"])
        key_to_score[key] = row["score_name"]

    for k, df in df_store.items():
        score_name = key_to_score[k]
        meth, run_id, rep = k

        cuts_iter = CUTS
        if df.shape[0] < cuts_iter[-1]:
            cuts_iter = [i for i in cuts_iter if i < df.shape[0]] + [df.shape[0]]

        for cut in cuts_iter:
            
            sub = df.head(cut)
            t_sec = (sub.ts.iloc[-1] - sub.ts.iloc[0]).total_seconds()
            sub   = sub[~np.isinf(sub.score)]

            for k in RANK_KS:
                rows.append({
                    "method": meth,
                    "run_id": run_id,
                    "run_replica": rep,
                    "score":  score_name,
                    "cut":    cut,
                    "k":      k,
                    "score_at_k": sub.score.nlargest(k).mean(),
                    "time":   t_sec / 60.0,
                })

    cut_df = pd.DataFrame(rows)
    cut_time = cut_df.groupby(["method", "score", "cut"])["time"].mean().reset_index()
    cut_time.columns = ["method", "score", "cut", "avg_time"]
    cut_df = cut_df.merge(cut_time, on=["method", "score", "cut"])
    cut_df["Method"] = cut_df["method"].map(lambda x: METHOD_LABEL[x])

    # round runtime cutoff of 719.x to 720 for plotting clarity
    cut_df.loc[cut_df.time>719, "time"] = 720

    return cut_df 


def plot_curves(curves: pd.DataFrame,
                k_val: int,
                prefix: str=""):
    """Two 2x2 panels: (inference → score) and (runtime → score)."""
    for xaxis, fname in [("cut", "inference_vs_score"),
                         ("avg_time", "runtime_vs_score")]:

        fig, axes = plt.subplots(2, 2, figsize=(12*2, 10*2))
        for i, score in enumerate(curves.score.unique()):
            sub = curves[(curves.k == k_val) & (curves.score == score)]
            ax  = axes.flat[i]
            sns.lineplot(sub, x=xaxis, y="score_at_k", hue="Method", ax=ax)
            ax.set_xlabel("Inference" if xaxis == "cut" else "Runtime (min)")
            ax.set_ylabel(f"Top-{k_val} Mean Score")
            ax.set_title(SCORE_PLOT_LABEL[score])
            if xaxis == "avg_time":
                ax.set_xscale("log")

            _tight_save(ax, RAW_DIR / f"{prefix}{fname}_top{k_val}_{score}.png")

        # add subplot labels
        labels = ["a", "b", "c", "d"]
        for axi, label in zip(axes.flatten(), labels):
            axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                    ha="right", va="top", fontsize=28, fontweight="bold")

        fig.tight_layout()

        p = FIG_DIR / f"{prefix}{fname}_top{k_val}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        console.log(f"[green]✓ wrote {p.name}")

def plot_agg_bars(agg: pd.DataFrame, prefix: str = ""):
    """
    2x2 grid of grouped bar charts (one subplot per score_name):
      • Clusters: Top-1, Top-10, Top-100 (left Y) and Runtime (right Y with axis break @100)
      • Bars: methods with mean ± std error bars (runtime errorbars transformed across break)
      • Excludes "SyntheMol (w/o bb)"
      • Adds subplot labels a/b/c/d
    Saves:
      • panel: FIG_DIR/{prefix}bars_grid.png
      • per-axis crops: RAW_DIR/{prefix}bars_<score_name>.png
    """
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'axes.spines.right': True,
    })

    # Filter out SyntheMol (w/o bb)
    agg_f = agg[agg["method"] != "synthemol (no bb)"].copy()
    if agg_f.empty:
        console.log("[yellow]No methods left after filtering; skipping bars panel.")
        return

    # Score panels in first-appearance order (limit to 4 for a 2x2 grid)
    score_names = list(dict.fromkeys(agg_f["score_name"].tolist()))[:4]

    # Global method order for consistent colors across subplots
    methods_order = list(dict.fromkeys(agg_f["method"].tolist()))
    method_label = {m: METHOD_LABEL.get(m, m) for m in methods_order}

    # Stable color map
    colors = plt.rcParams.get("axes.prop_cycle", plt.cycler(color=[])).by_key().get("color", [])
    color_map = {m: colors[i % len(colors)] if colors else None for i, m in enumerate(methods_order)}

    # Metric layout
    metrics = [
        ("top1",   "Top-1 ↑",        "score"),
        ("top10",  "Top-10 ↑",       "score"),
        ("top100", "Top-100 ↑",      "score"),
        ("runtime","Runtime ↓","runtime"),
    ]
    centers = np.arange(len(metrics))

    # Piecewise runtime transform for axis break at 100
    BREAK_AT = 60.0
    COMPRESS = 0.05  # region above break is compressed by this factor

    def _rt_trans(y: float) -> float:
        y = float(y)
        return y if y <= BREAK_AT else BREAK_AT + (y - BREAK_AT) * COMPRESS

    def _rt_inv(yt: float) -> float:
        yt = float(yt)
        return yt if yt <= BREAK_AT else BREAK_AT + (yt - BREAK_AT) / COMPRESS

    def _rt_err_trans(mean: float, std: float) -> tuple[float, float]:
        """Return (lower_err, upper_err) in transformed coordinates."""
        lo = max(mean - std, 0.0)
        hi = mean + std
        ty  = _rt_trans(mean)
        tlo = _rt_trans(lo)
        thi = _rt_trans(hi)
        return (ty - tlo, thi - ty)

    fig, axes = plt.subplots(2, 2, figsize=(28, 20))
    axes = axes.flatten()

    for idx, score in enumerate(score_names):
        ax = axes[idx]
        ax2 = ax.twinx()  # right y for runtime (with break)

        sub = agg_f[agg_f["score_name"] == score].copy()
        methods = [m for m in methods_order if m in set(sub["method"])]
        n_methods = max(len(methods), 1)
        width = 0.8 / n_methods

        # Gather runtime stats to set limits/ticks
        r_means, r_stds = [], []
        for m in methods:
            row = sub[sub["method"] == m]
            r_means.append(float(row[("runtime", "mean")].values[0]) if ("runtime","mean") in row.columns else 0.0)
            r_stds.append(float(row[("runtime", "std") ].values[0]) if ("runtime","std" ) in row.columns else 0.0)

        rts = [rm + rs for rm, rs in zip(r_means, r_stds)]
        if prefix == "":
            rts += [BREAK_AT]
        max_runtime = max(rts)
        
        # max_runtime = max([rm + rs for rm, rs in zip(r_means, r_stds)] + [BREAK_AT])

        # Bars
        for mi, m in enumerate(methods):
            row = sub[sub["method"] == m]
            color = color_map[m]

            for j, (met, xt_label, kind) in enumerate(metrics):
                pos = centers[j] + (mi - (n_methods - 1) / 2.0) * width

                # Pull mean/std safely
                mean = float(row[(met, "mean")].values[0]) if (met, "mean") in row.columns else 0.0
                std  = float(row[(met, "std") ].values[0]) if (met, "std")  in row.columns else 0.0

                if kind == "score":
                    lbl = method_label[m] if j == 0 else "_nolegend_"
                    ax.bar(pos, mean, width=width, yerr=std, capsize=5,
                           linewidth=0, color=color, label=lbl)
                else:
                    # runtime: transform heights and error bars across the break
                    ybar = _rt_trans(mean)
                    err_lo, err_hi = _rt_err_trans(mean, std)
                    lbl = "_nolegend_"  # legend handled via left axis only
                    ax2.bar(pos, ybar, width=width,
                            yerr=np.array([[err_lo], [err_hi]]),
                            capsize=5, linewidth=0, color=color, label=lbl)

        # Left axis
        ax.set_xticks(centers)
        ax.set_xticklabels([lab for _, lab, _ in metrics])
        ax.set_ylabel("Score (mean ± std)")
        ax.set_title(SCORE_PLOT_LABEL.get(score, score))
        if idx == 0:
            ax.legend(loc="lower left")

        # Right axis: limits & ticks with break at 100
        ymax_t = _rt_trans(max_runtime) * 1.10
        ax2.set_ylim(0, ymax_t)

        if prefix=="":

            # Compose ticks: dense below break, sparse above
            low_ticks  = np.linspace(0, BREAK_AT, 6)
            if max_runtime > BREAK_AT:
                # 2–3 ticks above break up to the max observed runtime
                high_ticks = np.linspace(BREAK_AT + (max_runtime - BREAK_AT) / 3,
                                        max_runtime, num=3)
                ticks = list(low_ticks) + list(high_ticks)
            else:
                ticks = list(low_ticks)

            ax2.set_yticks([_rt_trans(t) for t in ticks])
            ax2.set_yticklabels([f"{int(t)}" if abs(t - round(t)) < 1e-6 else f"{t:.0f}" for t in ticks])
            ax2.set_ylabel("Runtime (min) (mean ± std)")

            # Draw a small '//' break marker on the right axis at y=100
            if max_runtime > BREAK_AT:
                yb_t = _rt_trans(BREAK_AT)
                y0, y1 = ax2.get_ylim()
                y_frac = (yb_t - y0) / (y1 - y0)
                d = 0.02  # size of the marker in axes fraction
                kw = dict(transform=ax2.transAxes, color=ax2.spines["right"].get_edgecolor(),
                        clip_on=False, linewidth=2)
                # two short slashes near the right spine
                ax2.plot((1 - d, 1 + d), (y_frac - d, y_frac + d), **kw)
                ax2.plot((1 - d, 1 + d), (y_frac - 2*d, y_frac), **kw)

        # Per-subplot tight save (includes both axes' data region)
        _tight_save(ax, f"{prefix}bars_{score}.png")

    # Hide any unused axes
    for j in range(len(score_names), 4):
        axes[j].axis("off")

    # Subplot labels a/b/c/d
    for axi, label in zip(axes, ["a", "b", "c", "d"]):
        if axi.has_data():
            axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                     ha="right", va="top", fontsize=28, fontweight="bold")

    # Shared legend (methods present anywhere in panel)
    present_methods = [m for m in methods_order if m in set(agg_f["method"])]
    patches = [Patch(facecolor=color_map[m], label=method_label[m]) for m in present_methods]
    if patches:
        fig.subplots_adjust(bottom=0.12)

    fig.tight_layout()
    out = FIG_DIR / f"{prefix}bars_grid.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    console.log(f"[green]✓ wrote {out.name}")


# ╭────────────────────────────────── main ───────────────────────────────────╮
def _main(data_path: Path,
          table_filename: str,
          csv_filename: str,
          dataset_name: str,
          table_name: str,
          prefix: str="",
          ):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    summary, prm_store, df_store = gather_all(data_path)

    # ── aggregate & save csv ────────────────────────────────────────────
    agg_cols = [f"top{i}" for i in RANK_KS] + ["runtime", "n_results"]
    summary["runtime"] = summary["runtime"]/60 # convert to minutes 
    summary.loc[summary["runtime"]>716, "runtime"]=720 # clean up timeouts 
    agg = (summary.groupby(["score_name", "method"])
           .agg({col: ["mean", "std"] for col in agg_cols})
           .reset_index()
           .sort_values(["score_name", "method"], ascending=[True, True]))
    agg = agg.fillna(0.0)

    # ── LaTeX table ─────────────────────────────────────────────────────
    make_latex_table(agg, [1, 10, 100], FIG_DIR / table_filename, dataset_name, table_name)

    # ── grouped bar plots from agg ──────────────────────────────────────
    plot_agg_bars(agg, prefix=prefix)

    # ── save summary ────────────────────────────────────────────────────
    agg.columns = ['_'.join(i).strip("_") for i in  agg.columns.to_flat_index()]
    agg.to_csv(FINAL_DIR / csv_filename, index=False)
    console.log(f"[green]✓ wrote {csv_filename}")

    # ── score-curves panels ────────────────────────────────────────────
    curves = build_cut_curves(df_store, summary)
    plot_curves(curves, 100, prefix)
    curves.to_csv(FINAL_DIR / csv_filename.replace(".csv", "_cuts.csv"), index=False)

def main():
    _main(FINALS, "final_results_table.tex", "final_results.csv", 
          dataset_name="EnamineDataset", table_name="enamine_results")
    _main(ENUM_FINALS, "enum_final_results_table.tex", "enum_final_results.csv", 
          dataset_name="LyuDataset", table_name="enum_results", prefix="enum_")


if __name__ == "__main__":
    main()
