"""
Plots for the Enamine-decomposer model
-----------------------------------------  
"""

from __future__ import annotations
from pathlib import Path
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import scienceplots
from transformers import AutoConfig
import json
import numpy as np
import pandas as pd
import seaborn as sns 
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image

from .constants import METRICS_PATH, DECOMP_MODEL_NM, BLOB, TOPK_DIR

# ─────────────────────────────────── paths & style ───────────────────────────
FIG_DIR = (BLOB / "internal" / "figures" /
           "model_evaluation" / "enamine_decomposer")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = FIG_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RUN_ROOT = (BLOB / "internal" / "training_runs" / "enamine_decomposer")

TRAIN_RUNS = [
    "cos_loss_canonical",
    "mse_loss_canonical",
    "pearson_topk_loss_canonical",
    "pearson_topk_loss",
    "cos_loss",
    "mse_loss",
    "mse_pearson_topk_loss",
    "cos_pearson_topk_loss",
]


RUN_MAPPING = {
    "cos_loss_canonical": "cos_canonical_long",
    "mse_loss_canonical": "mse_canonical_long",
    "pearson_topk_loss_canonical": "ref_corr_k_canonical_long",
    "cos_pearson_topk_loss_canonical_order": "cos_ref_corr_k_canonical_long",
    "pearson_topk_loss": "ref_corr_k_long",
    "cos_loss": "cos_long",
    "mse_loss": "mse_long",
    "cos_pearson_topk_loss": "cos_ref_corr_k_long",
    "mse_pearson_topk_loss": "mse_ref_corr_k_long",
}

K_TRAIN = 10

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

# ╭─────────────────────────── data loading helpers ───────────────────────────╮
def _load_metrics():
    """Return (cutoffs, accuracy-dict, precision-dict)."""
    m = torch.load(METRICS_PATH)
    return m["cutoffs"], m["accuracy"], m["precision"]

def _load_knn_dfs():
    """Return (queries_df, ground_truth_df, {key:df,…})."""
    q = pd.read_parquet(TOPK_DIR / "queries.parquet")
    gt = pd.read_parquet(TOPK_DIR / "ground_truth.parquet")
    tbls = {p.stem: pd.read_parquet(p) for p in TOPK_DIR.glob("*_bb[12].parquet")}
    return q, gt, tbls


def _sizes_from_config():
    cfg = AutoConfig.from_pretrained(DECOMP_MODEL_NM, trust_remote_code=True)
    return sorted(cfg.comp_sizes), sorted(cfg.output_sizes)


def _aggregate(curves: dict[str, list[float]],
               fixed_as: str,
               fixed_vals: list[int],
               avg_over: list[int]) -> dict[int, list[float]]:
    """
    Average curves across the 'other' dimension.

    Parameters
    ----------
    curves     key = "in->out", value = list[float]
    fixed_as   "input" or "output"
    fixed_vals values along the fixed dimension (x-axis legends)
    avg_over   values to average across (the other dim)
    """
    out: dict[int, list[float]] = {}
    for val in fixed_vals:
        if fixed_as == "input":
            stack = [curves[f"{val}->{o}"] for o in avg_over]
        else:                                     # fixed_as == "output"
            stack = [curves[f"{i}->{val}"] for i in avg_over]
        out[val] = torch.tensor(stack).mean(0).tolist()
    return out
# ╰─────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── molecule-panel helpers ────────────────────────╮
# COLORS = {k: f"#{v.lower()}" for k, v in
        #   {"blue": "0C5DA5", "green": "00B945", "red": "FF2C00"}.items()}

COLORS = {
    "blue": "008cf9",
    "green": "006e00",
    "red": "b80058"
}

COLORS = {k:f"#{v.lower()}" for k,v in COLORS.items()}

def _smiles_img(smiles: str, w: int = 350, h: int = 350) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    d2d = Draw.MolDraw2DCairo(w, h)
    Draw.PrepareAndDrawMolecule(d2d, mol); d2d.FinishDrawing()
    return Image.open(BytesIO(d2d.GetDrawingText()))

def _border(ax, color: str | None, lw: int = 3):
    for sp in ax.spines.values():
        if color is None:
            sp.set_visible(False)
        else:
            sp.set_linewidth(lw); sp.set_color(color)

def _product_smiles(row):          # works for either col name
    return row.get("product", row.get("item"))

def _panel_to_axes(fig, gs, queries_df, gt_df, topk_tbls,
                   panel_no: int, idx: int, in_sz: int, out_sz: int,
                   k: int, cell_px: int, show_col_hdr: bool):
    font_sz = 18
    font_sz_small = 14
    font_sz_large = 22
    prod = _product_smiles(queries_df.iloc[idx])
    gt1, gt2 = gt_df.loc[idx, ["bb1", "bb2"]]
    if len(gt2) <= len(gt1):              # canonical order: shorter first
        gt1, gt2 = gt2, gt1

    base = f"{in_sz}->{out_sz}"
    bb1 = topk_tbls[f"{base}_bb1"].iloc[idx].tolist()[:k - 1]
    bb2 = topk_tbls[f"{base}_bb2"].iloc[idx].tolist()[:k - 1]

    # ── product cell (spans both rows) ────────────────────────────────────
    ax_prod = fig.add_subplot(gs[:, 0])
    ax_prod.imshow(_smiles_img(prod, 2 * cell_px, 2 * cell_px))
    ax_prod.text(0.5, 0.05, f"Query {panel_no} ({in_sz}-dim)", # \n{in_sz} → {out_sz}",
                 transform=ax_prod.transAxes,
                 ha="center", va="bottom", fontsize=font_sz,
                 bbox=dict(facecolor="white", edgecolor="none"))
    if show_col_hdr:
        ax_prod.set_title("Query", fontsize=font_sz, y=1.37)
    _border(ax_prod, None)

    # ── neighbour grids (two rows) ────────────────────────────────────────
    rows_hits, gts, flips = [bb1, bb2], [gt1, gt2], [gt2, gt1]
    for r, hits in enumerate(rows_hits):
        found_any, bb_axes = False, []
        hits = [gts[r]] + hits                         # prepend GT column
        for c, smi in enumerate(hits, start=1):        # start=1 → skip leftmost
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(_smiles_img(smi, cell_px, cell_px))

            if c == 1:                                 # target column
                if r == 0 and show_col_hdr:
                    ax.set_title("Target", fontsize=font_sz)
                ax.text(0.5, 0.05, f"BB{r+1} ({out_sz}-dim)",
                        transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=font_sz,
                        bbox=dict(facecolor="white", edgecolor="none"))
                _border(ax, None)
            else:
                if r == 0 and show_col_hdr:
                    ax.set_title(f"Top {c-1} Result", fontsize=font_sz)

                if smi == gts[r]:
                    _border(ax, COLORS["green"]); found_any = True
                elif smi == flips[r]:
                    _border(ax, COLORS["blue"]);  found_any = True
                    ax.text(0.96, 0.04, f"len={len(smi)}",
                            transform=ax.transAxes, fontsize=font_sz_small,
                            ha="right", va="bottom",
                            bbox=dict(facecolor="white", edgecolor="none"))
                else:
                    _border(ax, None)
                bb_axes.append(ax)

        if not found_any:                              # red ‘X’ row marker
            for ax in bb_axes:
                ax.text(0.96, 0.05, "X",
                        transform=ax.transAxes, color=COLORS["red"],
                        fontsize=font_sz_large, ha="right", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none"))

def plot_decomposer_panels(panels, plot_data, k: int = 5, cell_px: int = 350):
    """
    panels : list of tuples  (query_idx, input_size, output_size)
    Returns matplotlib Figure with panels stacked vertically.
    """
    plt.rcParams.update({
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    queries_df, gt_df, topk_tbls = plot_data
    n = len(panels)
    base_w = 1.1
    gs_master = GridSpec(nrows=2 * n, ncols=k + 1,
                         width_ratios=[1.3] + [base_w] * k,
                         height_ratios=[1] * (2 * n),
                         hspace=0.0, wspace=0.04)
    fig = plt.figure(figsize=((k + 1) * 3, n * 6))

    for p, (q_idx, inp, outp) in enumerate(panels):
        sub = GridSpecFromSubplotSpec(2, k + 1,
                                      subplot_spec=gs_master[2*p:2*p+2, :],
                                      width_ratios=[1.3] + [base_w] * k,
                                      hspace=0.00, wspace=0.04)
        _panel_to_axes(fig, sub,
                       queries_df, gt_df, topk_tbls,
                       p+1, q_idx, inp, outp,
                       k, cell_px, show_col_hdr=(p == 0))

    for ax in fig.get_axes():
        ax.set_xticks([]); ax.set_yticks([])

    return fig
# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭────────────────────────────── plotting core ───────────────────────────────╮
def _maybe_newline(lbl, txt):
    if lbl:
        # lbl.append("\n"+txt)
        lbl.append("+\n "+txt)
    else:
        lbl.append(txt)
    return lbl 

def _nice_label(run: str) -> str:
    lbl = []
    if "cos" in run:
        lbl = _maybe_newline(lbl, "Cos")
    if "mse" in run:
        lbl = _maybe_newline(lbl, "MSE")
    if "pearson" in run:
        lbl = _maybe_newline(lbl, "Chem")
    if "canonical" in run:
        lbl.append(" (R)")
        # lbl = _maybe_newline(lbl, "(R)")
        # lbl = _maybe_newline(lbl, "Rxn Ord")
    else:
        lbl.append(" (L)")
        # lbl = _maybe_newline(lbl, "(L)")
    return " ".join(lbl) or run

def _gather_train_metrics() -> dict[str, dict[int, dict[int, float]]]:
    """
    metrics[run][input_size][output_size]  (precision@K_TRAIN)
    pulled from trainer_state.json
    """
    store = defaultdict(lambda: defaultdict(dict))
    patt = f"p@{K_TRAIN}"
    for run in TRAIN_RUNS:
        for ts in (RUN_ROOT / RUN_MAPPING[run]).rglob("trainer_state.json"):
            last = json.loads(ts.read_text())["log_history"][-1]
            for key, val in last.items():
                if patt not in key:
                    continue
                # key example: eval_128->64_p@10
                sizes = key.replace("eval_", "").split("_p@")[0]
                inp, out = map(int, sizes.split("->"))
                store[run][inp][out] = val
    return store


def _aggregate_mean(metrics: dict[str, dict[int, dict[int, float]]]):
    """
    Returns
    -------
    med_run_score  : average over *all* in/out pairs  (bar-chart)
    med_by_input   : {run: [vals over INPUT_SIZES_CANON]}  (line-plot)
    """
    med_run, med_input = {}, {}
    ins, outs = _sizes_from_config()
    for run, grid in metrics.items():
        df = pd.DataFrame(grid).T.reindex(index=ins,
                                          columns=ins)
        med_run[run]   = float(df.mean().mean())
        med_input[run] = df.mean(axis=1).values.tolist()
    return med_run, med_input

def _add_lines(ax, xs, curves: dict[int, list[float]], *,
               legend_fmt="{:d}-dim", title: str, ylab: str):
    for dim, vec in curves.items():
        ax.plot(xs, vec, label=legend_fmt.format(dim))
    ax.set_xlabel("K Values"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend()

def _curves_to_df(xs: list[int], curves: dict[int, list[float]]) -> pd.DataFrame:
    """
    xs      list of x-values (cut-offs, input sizes, …)
    curves  {legend_key: [y0, y1, …]}
    Returns a DataFrame with first column = x-axis & one column per legend key.
    """
    data = {"x": xs}
    for key, ys in curves.items():
        data[str(key)] = ys
    return pd.DataFrame(data)

def _write_csv(df: pd.DataFrame | dict, name: str):
    """Convenience wrapper - always writes inside FIG_DIR/raw/."""
    if isinstance(df, dict):
        df = pd.DataFrame(list(df.items()), columns=["key", "value"])
    out = RAW_DIR / name
    df.to_csv(out, index=False)
    print("✓ wrote", out)

def _len_based_curves(retrieval_k: int = 10):
    """
    Returns
    -------
    flip_df, found1_df, found2_df   - already aggregated for plotting
    """
    _, gt_df, topk_tbls = _load_knn_dfs()
    sizes = [32, 64, 128, 256, 512, 768]

    # order BBs by shortest-first for a canonical ground-truth
    def _order(row):
        b1, b2 = row["bb1"], row["bb2"]
        return (b2, b1) if len(b2) <= len(b1) else (b1, b2)

    gt_df[["bb1_ord", "bb2_ord"]] = gt_df.apply(_order, axis=1, result_type="expand")
    gt_df["bb1_ord_len"] = gt_df["bb1_ord"].str.len()
    gt_df["bb2_ord_len"] = gt_df["bb2_ord"].str.len()
    gt_df["length_delta"] = (gt_df["bb1_ord_len"] - gt_df["bb2_ord_len"]).abs()

    recs = []                                                             # collect rows
    for i_sz in sizes:
        for o_sz in sizes:
            key = f"{i_sz}->{o_sz}"
            bb1_pred = topk_tbls[f"{key}_bb1"].values[:, :retrieval_k]
            bb2_pred = topk_tbls[f"{key}_bb2"].values[:, :retrieval_k]

            bb1_gt = gt_df["bb1_ord"].values[:, None]
            bb2_gt = gt_df["bb2_ord"].values[:, None]

            flip = ((bb1_pred == bb2_gt).any(-1) |
                    (bb2_pred == bb1_gt).any(-1))

            found1 = ((bb1_pred == bb1_gt).any(-1) |
                      (bb2_pred == bb1_gt).any(-1))
            found2 = ((bb1_pred == bb2_gt).any(-1) |
                      (bb2_pred == bb2_gt).any(-1))

            tmp = gt_df[["bb1_ord_len", "bb2_ord_len",
                          "length_delta"]].copy()
            tmp["flipped"] = flip
            tmp["found1"]  = found1
            tmp["found2"]  = found2
            recs.append(tmp)

    big = pd.concat(recs, ignore_index=True)

    flip_df   = big.groupby("length_delta")["flipped"].mean().reset_index()
    miss1_df  = (1.0 - big.groupby("bb1_ord_len")["found1"].mean()).reset_index()
    miss1_df.columns = ["len", "miss"]
    miss2_df  = (1.0 - big.groupby("bb2_ord_len")["found2"].mean()).reset_index()
    miss2_df.columns = ["len", "miss"]

    return flip_df, miss1_df, miss2_df

def make_flip_miss_panel(retrieval_k: int = 10
                         ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """1x 2 panel showing (A) flip-rate vs Δ-length and (B) miss-rate vs length."""
    flip_df, miss1_df, miss2_df = _len_based_curves(retrieval_k)

    fig, axs = plt.subplots(1, 2, figsize=(12*2, 5*2))
    ax_flip, ax_miss = axs
    ax_map = {"flip": ax_flip, "miss": ax_miss}

    # --- flipped predictions ------------------------------------------------
    ax_flip.plot(flip_df["length_delta"], flip_df["flipped"],
                 marker="o")
    ax_flip.set_xlabel("Absolute SMILES Length Difference BB2 - BB1")
    ax_flip.set_ylabel(f"Flip Rate (k = {retrieval_k})")
    ax_flip.set_title("Flipped-Prediction Rate\nvs Length Difference")

    # --- miss-rate ----------------------------------------------------------
    ax_miss.plot(miss1_df["len"], miss1_df["miss"],
                 marker="o", label="Building Block 1")
    ax_miss.plot(miss2_df["len"], miss2_df["miss"],
                 marker="s", label="Building Block 2")
    ax_miss.set_xlabel("SMILES Length")
    ax_miss.set_ylabel(f"Miss Rate (k = {retrieval_k})")
    ax_miss.set_title("Retrieval Miss-Rate\nvs SMILES Length")
    ax_miss.legend()

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(axs.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    _write_csv(flip_df,   "flip_rate_vs_len_delta.csv")
    _write_csv(miss1_df,  "miss_rate_bb1_vs_len.csv")
    _write_csv(miss2_df,  "miss_rate_bb2_vs_len.csv")

    fig.tight_layout()
    return fig, ax_map

def make_prec_acc_panel() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Main 2 x 2 metrics (accuracy / precision) panel."""
    cutoffs, acc, prec = _load_metrics()
    ins, outs = _sizes_from_config()

    acc_in   = _aggregate(acc,  "input",  ins,  outs)
    acc_out  = _aggregate(acc,  "output", outs, ins)
    prec_in  = _aggregate(prec, "input",  ins,  outs)
    prec_out = _aggregate(prec, "output", outs, ins)

    fig, axs = plt.subplots(2, 2, figsize=(12*2, 10*2))
    ax = {"acc_in":  axs[0, 0], "acc_out":  axs[0, 1],
          "prec_in": axs[1, 0], "prec_out": axs[1, 1]}

    _add_lines(ax["acc_in"],  cutoffs, acc_in,
               title="Retrieval Accuracy - by Input Size",
               ylab="Retrieval Accuracy",
               legend_fmt="{:d}-dim Input")
    _add_lines(ax["acc_out"], cutoffs, acc_out,
               title="Retrieval Accuracy - by Output Size",
               ylab="Retrieval Accuracy",
               legend_fmt="{:d}-dim Output")
    _add_lines(ax["prec_in"], cutoffs, prec_in,
               title="Precision - by Input Size",
               ylab="Retrieval Precision",
               legend_fmt="{:d}-dim Input")
    _add_lines(ax["prec_out"], cutoffs, prec_out,
               title="Precision - by Output Size",
               ylab="Retrieval Precision",
               legend_fmt="{:d}-dim Output")

    # shared y-ranges
    for k in ("acc_in", "acc_out"):
        ax[k].set_ylim(0.0, 1.0)
    for k in ("prec_in", "prec_out"):
        ax[k].set_ylim(0.0, 1.0)

    # add subplot labels
    labels = ["a", "b", "c", "d"]
    for axi, label in zip(axs.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    _write_csv(_curves_to_df(cutoffs, acc_in),  "acc_by_input.csv")
    _write_csv(_curves_to_df(cutoffs, acc_out), "acc_by_output.csv")
    _write_csv(_curves_to_df(cutoffs, prec_in), "prec_by_input.csv")
    _write_csv(_curves_to_df(cutoffs, prec_out),"prec_by_output.csv")

    fig.tight_layout()
    return fig, ax

def _matrix_from_curves(curves: dict[str, list[float]],
                        cutoffs: list[int],
                        k_val: int,
                        ins: list[int],
                        outs: list[int]) -> np.ndarray:
    """
    Build a square matrix  [len(ins) , len(outs)]
    where cell (i,j) = metric(input=ins[i], output=outs[j]) @ k_val.
    """
    k_idx = cutoffs.index(k_val)
    mat   = np.ones((len(ins), len(outs)))
    for i_sz in ins:
        for o_sz in outs:
            mat[ins.index(i_sz), outs.index(o_sz)] = curves[f"{i_sz}->{o_sz}"][k_idx]
    return mat


def make_prec_acc_heatmap_panel(k_val: int = 10
                                ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    1 x 2 panel - left = accuracy heat-map, right = precision heat-map,
    *without* input/output aggregation.
    """
    cutoffs, acc, prec = _load_metrics()
    ins, outs = _sizes_from_config()

    acc_mat  = _matrix_from_curves(acc,  cutoffs, k_val, ins, outs)
    prec_mat = _matrix_from_curves(prec, cutoffs, k_val, ins, outs)

    fig, axs = plt.subplots(1, 2, figsize=(12*2, 5*2))
    ax_acc, ax_prec = axs
    ax_map = {"acc_hmap": ax_acc, "prec_hmap": ax_prec}

    sns.heatmap(acc_mat,  ax=ax_acc,  cmap="viridis", annot=True,
                xticklabels=outs, yticklabels=ins, linewidths=.5,
                cbar_kws=dict(label=f"Accuracy @k={k_val}"), fmt='.3g')
    ax_acc.set_xlabel("Output Embedding Size"); ax_acc.set_ylabel("Input Embedding Size")
    ax_acc.set_title("Retrieval Accuracy")

    sns.heatmap(prec_mat, ax=ax_prec, cmap="viridis", annot=True,
                xticklabels=outs, yticklabels=ins, linewidths=.5,
                cbar_kws=dict(label=f"Precision @k={k_val}"), fmt='.3g')
    ax_prec.set_xlabel("Output Embedding Size"); ax_prec.set_ylabel("Input EmbeddingSize")
    ax_prec.set_title("Retrieval Precision")

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(axs.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    # ── raw CSV exports ────────────────────────────────────────────────
    _write_csv(_curves_to_df(outs, {i: acc_mat[r]  for r,i in enumerate(ins)}),
               f"acc_heatmap_k{k_val}.csv")
    _write_csv(_curves_to_df(outs, {i: prec_mat[r] for r,i in enumerate(ins)}),
               f"prec_heatmap_k{k_val}.csv")

    fig.tight_layout()
    return fig, ax_map

def _tight_save(axis: plt.Axes, path: Path):
    fig = axis.get_figure()
    fig.canvas.draw()
    bbox = axis.get_tightbbox(fig.canvas.get_renderer()
                              ).transformed(fig.dpi_scale_trans.inverted()
                              ).expanded(1.03, 1.05)
    fig.savefig(path, dpi=300, bbox_inches=bbox)
    print("✓ wrote", path)

def make_train_panel() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    ins, outs = _sizes_from_config()
    raw = _gather_train_metrics()
    med_bar, med_input = _aggregate_mean(raw)

    _write_csv(med_bar, "train_loss_bar.csv")
    ins_series = []
    for run, vec in med_input.items():
        ins_series.append(
            pd.DataFrame({"input_size": ins,
                          "run": run,
                          "precision": vec})
        )
    _write_csv(pd.concat(ins_series, ignore_index=True),
               "train_precision_by_input.csv")

    # fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(14, 5))
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12*2, 5*2),
        gridspec_kw={"width_ratios": [1.0, 1]}
    )
    ax_bar, ax_curve = axs

    ax_map = {"loss": ax_bar, "curve": ax_curve}

    # --- bar chart ---------------------------------------------------------
    ordered = sorted(med_bar.items(), key=lambda x: x[1], reverse=True)
    names   = [_nice_label(n) for n, _ in ordered]
    vals    = [v for _, v in ordered]

    ax_bar.bar(range(len(names)), vals)
    ax_bar.set_xticks(range(len(names)), names, rotation=0, ha="center", fontsize=16)
    ax_bar.set_ylabel(f"Average Precision at k={K_TRAIN}")
    ax_bar.set_title("Decomposer Loss Comparison")
    for i, v in enumerate(vals):
        ax_bar.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=16)
    ax_bar.margins(x=.03)

    # --- input-size curve --------------------------------------------------
    for run, vec in med_input.items():
        ax_curve.plot(ins, vec, marker="o",
                      label=_nice_label(run).replace("\n", ""))
    ax_curve.set_xlabel("Input Embedding Size")
    ax_curve.set_ylabel(f"Average Precision at k={K_TRAIN}")
    ax_curve.set_title("Precision by Input Size")
    ax_curve.set_xticks(ins)
    ax_curve.legend()

    # add subplot labels
    labels = ["a", "b"]
    for axi, label in zip(axs.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                 ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    return fig, ax_map

def save_figures(fig: plt.Figure,
                 axes: dict[str, plt.Axes],
                 extra: dict[str, plt.Axes] | None = None,
                 heatmap_k: int=10,
                 ) -> None:
    """Save a panel *and* every axis it contains."""
    if {"acc_in", "acc_out"} <= axes.keys():               # main metrics panel
        panel_name = "decomposer_prec_acc_panel_2x2.png"
    elif {"flip", "miss"} <= axes.keys():                  # length-effects panel
        panel_name = "decomposer_len_effect_panel_1x2.png"
    elif {"acc_hmap", "prec_hmap"} <= axes.keys():         # heat-map panel
        panel_name = f"decomposer_prec_acc_heatmap_k{heatmap_k}_1x2.png"
    else:                                                  # training panel
        panel_name = "decomposer_panel_train.png"

    fig.savefig(FIG_DIR / panel_name, dpi=300)
    print("✓ wrote", FIG_DIR / panel_name)

    indiv = {
        "acc_by_input.png":      "acc_in",
        "acc_by_output.png":     "acc_out",
        "prec_by_input.png":     "prec_in",
        "prec_by_output.png":    "prec_out",
        "flip_vs_len_delta.png": "flip",
        "miss_vs_len.png":       "miss",
        "loss_comparison.png":   "loss",
        "input_curve.png":       "curve",
        "acc_heatmap.png":  "acc_hmap",
        "prec_heatmap.png": "prec_hmap",
    }

    ax_all = axes.copy()
    if extra:
        ax_all.update(extra)

    for fname, key in indiv.items():
        if key not in ax_all:
            continue
        _tight_save(ax_all[key], FIG_DIR / fname)

def _save_example_panels():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    queries_df, gt_df, topk_tbls = _load_knn_dfs()
    plot_data = (queries_df, gt_df, topk_tbls)

    examples = [
        ([(42, 256, 128), (99, 128, 64), (420, 32, 128)], "decomposer_panels_set1.png"),
        ([(1044, 32, 128), (2230, 256, 256), (3412, 768, 768)], "decomposer_panels_set2.png"),
        ([(3855, 768, 512), (2536, 128, 512), (1371, 512, 512)], "decomposer_panels_set3.png")
    ]

    for plist, fname in examples:
        fig = plot_decomposer_panels(plist, plot_data, k=6)
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print("✓ wrote", FIG_DIR / fname)
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭────────────────────────────────── CLI ──────────────────────────────────────╮

def main():
    # metric grid
    fig_metrics, axes_metrics = make_prec_acc_panel()
    save_figures(fig_metrics, axes_metrics)

    # heat-map panel
    k_val = 10
    fig_hmap, axes_hmap = make_prec_acc_heatmap_panel(k_val=k_val)
    save_figures(fig_hmap, axes_hmap, heatmap_k=k_val)

    # flip / miss panel
    fig_len, axes_len = make_flip_miss_panel()
    save_figures(fig_len, axes_len)

    # training-run panel
    fig_train, axes_train = make_train_panel()
    save_figures(fig_train, {}, extra=axes_train)

    # make molecule plots
    _save_example_panels()


if __name__ == "__main__":
    main()
