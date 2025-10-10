"""
Generate evaluation plots for the embedding-compression study.

"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json, os

import torch, numpy as np, pandas as pd, seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots

from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image

from .constants import SIZES

# ╭──────────────────────────────────────── constants ─────────────────────────╮
BLOB          = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()
RUN_ROOT      = BLOB / "internal" / "training_runs" / "embedding_compression"
DATA_PATH     = BLOB / "internal" / "data_analysis" / "model_evaluation" / "embedding_compression"
PREC_PATH     = BLOB / "internal" / "data_analysis" / "model_evaluation" / \
                "embedding_compression" / "precision.pt"
FIG_DIR       = BLOB / "internal" / "figures" / "model_evaluation" / \
                "embedding_compression"
PARQUET_DIR   = BLOB / "internal" / "data_analysis" / "model_evaluation" / \
                "embedding_compression" / "topk"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = FIG_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Training runs to compare
TRAIN_RUNS = [
    "pearson_topk_loss",
    "mse_topk_loss",
    "mse_loss",
    "pearson_loss",
    "mse_topk_loss_large",
    "pearson_topk_loss_large",
]

RUN_MAPPING = {
    "pearson_topk_loss": "pearson_topk_lr1e3_wd001_1_1",
    "mse_topk_loss": "mse_topk_lr1e3_wd001_1_1",
    "mse_loss": "mse_lr1e3_wd001_1_1",
    "pearson_loss": "pearson_lr1e3_wd001_1_1",
    "mse_topk_loss_large": "mse_topk_lr3e4_wd001_4_4",
    "pearson_topk_loss_large": "pearson_topk_lr1e3_wd001_4_4",
}

LOSS_COMPARISON  = [
    "pearson_topk_loss",
    "mse_topk_loss",
    "mse_loss",
    "pearson_loss",
]
SIZE_COMPARISON  = [
    "pearson_topk_loss",
    "mse_topk_loss",
    "mse_topk_loss_large",
    "pearson_topk_loss_large",
]

PRECISION_K   = 100

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

# COLORS = {
#     "blue": "0C5DA5",
#     "green": "00B945",
#     "red": "FF2C00"
# }

COLORS = {
    "blue": "008cf9",
    "green": "006e00",
    "red": "b80058"
}

COLORS = {k:f"#{v.lower()}" for k,v in COLORS.items()}
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────── helper - run-name → nice label ────────────────────╮
def _run_to_label(run: str, add_size: bool = False) -> str:
    mapping = {
        "pearson": "ChemRank",
        "pearson_topk": "ChemRank Top-K",
        "mse": "MSE",
        "mse_topk": "MSE Top-K",
    }
    key = run.split("_loss")[0]
    label = mapping.get(key, key.capitalize()) + " Loss"
    if add_size:
        label += " Large" if "large" in run else " Small"
    return label
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────── load training metrics ───────────────────────────╮
def _gather_eval_metrics() -> dict[str, dict[int, dict[int, float]]]:
    """
    Returns
    -------
    metrics[run_name][k][size] -> precision value
    """
    metrics = defaultdict(lambda: defaultdict(dict))
    for run in TRAIN_RUNS:
        for ts_file in (RUN_ROOT / RUN_MAPPING[run]).rglob("trainer_state.json"):
            print(ts_file)
            state = json.loads(ts_file.read_text())
            # last entry in log_history is the final eval pass
            last_metrics = state["log_history"][-1]
            for k, v in last_metrics.items():
                if not k.startswith("eval_p@"):
                    continue
                try:
                    _, rest = k.split("p@", 1)
                    k_val, sz_val = rest.split("_")
                    k_int, sz_int = int(k_val), int(sz_val)
                except ValueError:
                    continue
                metrics[run][k_int][sz_int] = v
    return metrics
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────────── load K-NN precision ────────────────────────────╮
def _load_precision_pt() -> dict:
    if not PREC_PATH.exists():
        raise FileNotFoundError(f"{PREC_PATH} missing - run generate_data first.")
    return torch.load(PREC_PATH)

def _load_knn_dfs():
    queries_df = pd.read_parquet(PARQUET_DIR / "queries.parquet")
    topk_tbls  = {sz: pd.read_parquet(PARQUET_DIR / f"topk_{sz}.parquet")
                  for sz in SIZES}
    return queries_df, topk_tbls 
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────────── plotting helpers ──────────────────────────────╮

def _write_csv(df: pd.DataFrame | list[dict] | dict, fname: str):
    """Always writes into FIG_DIR/raw/."""
    if isinstance(df, list):
        df = pd.DataFrame(df)
    elif isinstance(df, dict):
        df = pd.DataFrame(list(df.items()), columns=["key", "value"])
    out = RAW_DIR / fname
    df.to_csv(out, index=False)
    print("✓ wrote", out)

def mol_to_img(smiles: str, w: int = 400, h: int = 400) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    d2d = Draw.MolDraw2DCairo(w, h)
    Draw.PrepareAndDrawMolecule(d2d, mol)
    d2d.FinishDrawing()
    return Image.open(BytesIO(d2d.GetDrawingText()))

def _plot_training_curves(metrics: dict,
                          runs: list[str],
                          title: str,
                          ax,
                          add_size: bool = False) -> None:
    for run in runs:
        data = sorted(metrics[run][PRECISION_K].items())      # (size, value)
        x, y = zip(*data) if data else ([], [])
        ax.plot(x, y, marker="o",
                label=_run_to_label(run, add_size=add_size))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Compression Size")
    ax.set_ylabel(f"Precision at {PRECISION_K}")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xticks(x)


def _plot_prec_to_native(prec_data: dict, ax) -> None:
    for size, vals in prec_data["to_gt"].items():
        ax.plot(prec_data["cutoffs"], vals, label=f"{size}→768")
    ax.set_xlabel("K Value")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision")
    ax.set_title("Compressed → Native precision")
    ax.legend()


def _plot_agreement_heatmap(prec_data: dict, ax) -> None:
    k_idx = prec_data["cutoffs"].index(PRECISION_K)
    heat = np.ones((len(SIZES), len(SIZES)))

    for key, vals in prec_data["agreement"].items():
        s1, s2 = map(int, key.split("_"))
        i, j = SIZES.index(s1), SIZES.index(s2)
        heat[i, j] = heat[j, i] = vals[k_idx]

    sns.heatmap(heat, ax=ax, annot=True, cmap="viridis",
                xticklabels=SIZES, yticklabels=SIZES,
                linewidths=.5)
    ax.set_title(f"Inter-size precision at k={PRECISION_K}")
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel("Embedding Size")
    ax.tick_params(axis="y", rotation=0)

def make_nn_panel(queries_df: pd.DataFrame,
                  topk_tbls: dict,
                  query_idx: int,
                  k_mols: int = 5,
                  cell_inch: float = 3.0,
                  out_name: str | None = None):
    plt.rcParams.update({
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    font_sz = default_font_size
    # font_sz = 20
    # font_sz_small = 14

    # ---------- load data -------------------------------------------------
    query_smiles = queries_df.iloc[query_idx]["item"]
    native_list  = topk_tbls[768].iloc[query_idx].tolist()

    plot_sizes   = sorted(list(SIZES), reverse=True)
    n_rows       = len(plot_sizes) + 1

    # ---------- canvas + GridSpec -----------------------------------------
    fig_w, fig_h = k_mols * cell_inch, n_rows * cell_inch
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=n_rows, ncols=k_mols,
        height_ratios=[1] + [1]*len(plot_sizes),
        hspace=0.02, wspace=0.02,
        left=0.02, right=0.98, top=0.98, bottom=0.02
    )
    spines = ["top", "right", "bottom", "left"]

    # ---------- 1) query cell ---------------------------------------------
    ax_q = fig.add_subplot(gs[0, :])
    ax_q.imshow(mol_to_img(query_smiles, 400, 400))
    ax_q.set_title(f"Query Molecule", fontsize=font_sz)
    ax_q.set_xticks([]); ax_q.set_yticks([]); ax_q.tick_params(length=0)
    for sp in spines:
        ax_q.spines[sp].set_visible(False)

    # ---------- 2) neighbour grid -----------------------------------------
    # row_lbl_kwargs = dict(fontsize=font_sz_small, weight="bold", va="center")

    for r, size in enumerate(plot_sizes, start=1):
        row_mols = topk_tbls[size].iloc[query_idx].tolist()[:k_mols]

        for c, smi in enumerate(row_mols):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(mol_to_img(smi, 400, 400))
            ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(length=0)

            # FIRST COL: row label (once per size row)
            if c == 0:
                ax.set_ylabel(f"{size}-dim", fontsize=font_sz)

            # CAPTION COL HEADERS (top grid row only)
            if r == 1:
                ax.set_title(f"Top {c+1} Result", fontsize=font_sz)

            # ---------------- correctness colouring ------------------------
            correct_any   = smi in native_list[:k_mols]
            should_rank   = native_list.index(smi)+1 if smi in native_list else None 
            this_rank     = c+1

            if correct_any and this_rank == should_rank:
                # perfect match → spine invisible
                for sp in spines:
                    ax.spines[sp].set_visible(False)
            else:
                col = COLORS["blue"] if correct_any else COLORS["red"]
                for sp in spines:
                    ax.spines[sp].set_linewidth(2)
                    ax.spines[sp].set_color(col)

                # rank annotation (bottom-right)
                disp = f"{this_rank}→{should_rank if should_rank else 'X'}"
                ax.text(
                    0.96, 0.04, disp,
                    transform=ax.transAxes,
                    fontsize=font_sz, ha="right", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none")
                )

    if out_name is None:
        out_name = f"compression_nn_grid_Q{query_idx}.png"
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("✓ wrote", out_path)

# ╰─────────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────────── main entry ──────────────────────────────────╮
def main() -> None:
    metrics = _gather_eval_metrics()
    rows = []
    for run, k_dict in metrics.items():
        for size, prec in k_dict[PRECISION_K].items():
            rows.append({"run": run, "embed_size": size,
                         "precision": prec})
    _write_csv(rows, "training_precision_at100.csv")

    prec_pt = _load_precision_pt()
    to_native = {"cutoff": prec_pt["cutoffs"]}
    for size, vals in prec_pt["to_gt"].items():
        to_native[str(size)] = vals
    _write_csv(pd.DataFrame(to_native),
               "precision_to_native_curves.csv")
    
    agree_rows = []
    k_idx = prec_pt["cutoffs"].index(PRECISION_K)
    for pair, vals in prec_pt["agreement"].items():
        s1, s2 = map(int, pair.split("_"))
        agree_rows.append({"size1": s1, "size2": s2,
                           "precision": vals[k_idx]})
    _write_csv(agree_rows, "precision_agreement_k100.csv")

    # ------------------------------------------------------------------ 4-panel
    fig, axs = plt.subplots(2, 2, figsize=(12*2, 10*2))
    _plot_training_curves(metrics, LOSS_COMPARISON,
                          "Effect of Loss Function", axs[0, 0], add_size=False)
    _plot_training_curves(metrics, SIZE_COMPARISON,
                          "Effect of Model Size",    axs[0, 1], add_size=True)
    _plot_prec_to_native(prec_pt, axs[1, 0])
    _plot_agreement_heatmap(prec_pt, axs[1, 1])

    # add subplot labels
    labels = ["a", "b", "c", "d"]
    for axi, label in zip(axs.flatten(), labels):
        axi.text(-0.1, 1.1, label, transform=axi.transAxes,
                ha="right", va="top", fontsize=28, fontweight="bold")

    fig.tight_layout()
    panel_path = FIG_DIR / "compression_prec_panel_4up.png"
    fig.savefig(panel_path, dpi=300)
    print(f"✓ wrote {panel_path}")

    # ------------------------------------------------------- individual files
    individual = {
        "train_loss_comparison.png":  axs[0, 0],
        "model_size_comparison.png":  axs[0, 1],
        "precision_to_native.png":    axs[1, 0],
        "precision_agreement_heatmap.png": axs[1, 1],
    }

    for fname, ax in individual.items():
        fig_parent = ax.get_figure()

        # draw once so that tight-bbox knows exact text extents
        fig_parent.canvas.draw()
        renderer = fig_parent.canvas.get_renderer()

        # full bbox (axis + tick labels + title)
        tight_bbox = ax.get_tightbbox(renderer)

        # convert to figure units and add a tiny margin
        bbox_inches = tight_bbox.transformed(
            fig_parent.dpi_scale_trans.inverted()).expanded(1.02, 1.04)

        out_path = FIG_DIR / fname
        fig_parent.savefig(out_path, dpi=300, bbox_inches=bbox_inches)
        print(f"✓ wrote {out_path}")

    queries_df, topk_tbls = _load_knn_dfs()
    plot_idxs = [20, 200, 1620, 4078]
    for plot_idx in plot_idxs:
        make_nn_panel(queries_df, topk_tbls, plot_idx)


if __name__ == "__main__":
    main()
