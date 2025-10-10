from __future__ import annotations
from pathlib import Path
import math
from collections import defaultdict
from typing import Sequence
from string import ascii_uppercase
import math 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from io import BytesIO
from scipy.stats import pearsonr, spearmanr
from rich.console import Console

console = Console()

from .constants import (
    BBKNN_CSV, NATPROD_CSV, EGFR_CSV,
    RAW_DIR,
    FIG_DIR,
    ENAMINE_CLASS,
    EGFR_CLASS
)

# ───────────────────────────── helpers ─────────────────────────────────
def _rank_q(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)

def _save_raw(df: pd.DataFrame, name: str):
    out = RAW_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print("✓ wrote", out)

def _tight_save(axis: plt.Axes, path: Path):
    fig = axis.get_figure()
    fig.canvas.draw()
    bbox = axis.get_tightbbox(fig.canvas.get_renderer()
                              ).transformed(fig.dpi_scale_trans.inverted()
                              ).expanded(1.03, 1.05)
    fig.savefig(path, dpi=300, bbox_inches=bbox)
    print("✓ wrote", path)

def _corr_cut(df: pd.DataFrame,
              cut_col: str,
              x: str, y: str,
              n: int):
    cuts = np.log10(np.logspace(0, 1, num=n))
    qs, p_arr, s_arr = [], [], []
    xval, yval, qval = df[x].to_numpy(), df[y].to_numpy(), df[cut_col].to_numpy()

    for c in cuts:
        m = qval >= c
        if m.sum() >= 2:
            qs.append(c)
            p_arr.append(pearsonr(xval[m], yval[m]).statistic)
            s_arr.append(spearmanr(xval[m], yval[m]).statistic)

    return np.asarray(qs), np.asarray(p_arr), np.asarray(s_arr)

def _cc_labels(cc_vals, newline=True):
    replace_strs = [
        " and steroid derivatives",
        " and derivatives",
        " and substituted derivatives"
    ]
    labels = []
    for c in cc_vals:
        for s in replace_strs:
            c = c.replace(s, "")
        if c=="D4":
            c = "Lyu 140M"

        labels.append(c)
        
    if newline:
        labels = [i.replace(' ', '\n') for i in labels]
    return labels

# ---------- molecule helpers ---------
def _img_from_smiles(smi: str, size=400) -> Image.Image:
    mol = Chem.MolFromSmiles(smi)
    d2d = Draw.MolDraw2DCairo(size, size)
    Draw.PrepareAndDrawMolecule(d2d, mol); d2d.FinishDrawing()
    return Image.open(BytesIO(d2d.GetDrawingText()))

def _pair_grid(df: pd.DataFrame,
               rows: int,
               cols: int,
               tag: str,
               *,
               random_state: int = 42,
               label_by_class: bool = False,
               label_by_score: bool = False,
               save_dir: str = None):
    assert cols % 2 == 0, "cols must be even - one axis per molecule"

    cell_in  = 5.0
    fig_w, fig_h = cols * cell_in, rows * cell_in
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.02, hspace=0.02,
                         left=0.02, right=0.98,
                         top=0.98, bottom=0.02)
    )
    axes = np.atleast_2d(axes)
    pairs   = rows * (cols // 2)

    n_per_query = math.ceil(df["query"].unique().shape[0] / pairs)
    sample = (df.groupby(["query"])
              .head(n_per_query)
              .sample(n=min(pairs, len(df)), random_state=random_state))

    pair_meta = [] 

    itr = sample.itertuples()
    for r in range(rows):
        for p in range(cols // 2):
            c_left         = p * 2
            left_ax        = axes[r, c_left]
            right_ax       = axes[r, c_left + 1]
            row            = next(itr)

            # ── images ───────────────────────────────────────────────
            left_ax.imshow (_img_from_smiles(row.query))
            right_ax.imshow(_img_from_smiles(row.result))
            for ax in (left_ax, right_ax):
                ax.set_xticks([]); ax.set_yticks([])

            pair_id = ascii_uppercase[r * (cols // 2) + p]

            # ---- build dynamic corner-labels --------------------------------
            lbl_q = f"{pair_id}1"
            lbl_r = f"{pair_id}2"

            if ("chemical_class" in row._fields) and label_by_class:
                cls = getattr(row, "chemical_class", "")
                if cls not in ("Enamine Molecules", "EGFR"):     # nat-prod classes
                    lbl_q += f": {cls}"

            if ({"query_score", "result_score"} <= set(row._fields)) and label_by_score:
                lbl_q += f" - Query Score {-1*row.query_score:5.2f}"
                lbl_r += f" - Result Score {-1*row.result_score:5.2f}"

            if "D4" in lbl_q:
                lbl_q = lbl_q.replace("D4", "Lyu 140M")

            left_ax .text(0.02, 0.95, lbl_q, transform=left_ax .transAxes,
                          ha="left", va="top", fontsize=16, fontweight="bold")
            right_ax.text(0.02, 0.95, lbl_r, transform=right_ax.transAxes,
                          ha="left", va="top", fontsize=16, fontweight="bold")


            # --------------------------------------

            # ── store caption to add *after* tight-layout ────────────
            caption = (
                f"Cosine {row.cosine_similarity:.3f} (Quantile {row.cos_quantile:.3f}) | "
                f"Tanimoto {row.tanimoto_similarity:.3f} (Quantile {row.tani_quantile:.3f})"
            )
            pair_meta.append((left_ax, right_ax, caption))

    for ax in axes.ravel():
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── add captions in figure coordinates ───────────────────────────
    fig.canvas.draw()                           # need renderer for bbox
    for l_ax, r_ax, txt in pair_meta:
        bb_l, bb_r = l_ax.get_position(), r_ax.get_position()
        x_mid      = (bb_l.x1 + bb_r.x0) / 2
        y_txt      = bb_l.y0
        fig.text(x_mid, y_txt+0.02, txt, ha="center", va="top", fontsize=16)

    if save_dir is not None:
        fpath = FIG_DIR / save_dir / f"{tag}_grid.png"
    else:
        fpath = FIG_DIR / f"{tag}_grid.png"
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    console.log(f"[green]✓ wrote {fpath}")

# ╭───────────────────────────────────── I/O helpers ─────────────────────────╮
def _corr_path(lat_sz: int, cuts: int) -> Path:
    return RAW_DIR / f"corr_arrays_size{lat_sz}_cuts{cuts}.npz"

def load_data(latent_size: int, cuts: int
              ) -> tuple[pd.DataFrame,
                         tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]:
    """
    Returns
    -------
    df       - similarity DataFrame (cached CSV)
    (q_cos, q_tani)
             - each tuple = (quantile-cutoffs, pearson, spearman)
               cached as .npz to avoid recomputation
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_path   = RAW_DIR / f"similarity_values_size{latent_size}.csv"
    corr_path = _corr_path(latent_size, cuts)

    # ── DataFrame ────────────────────────────────────────────────────────
    if df_path.exists():
        console.log("[cyan]loading cached similarity CSV")
        df = pd.read_csv(df_path)
    else:
        console.log("[cyan]building similarity DataFrame")
        keep_cols = [
            "query", "in_size", "out_size", "max_rank", "result",
            "cosine_similarity", "tanimoto_similarity", "chemical_class"
        ]
        # skip chemical class, add scores
        egfr_cols = keep_cols[:-1] + ["query_score", "result_score"]

        df_bbknn   = pd.read_csv(BBKNN_CSV,   usecols=keep_cols)
        df_natprod = pd.read_csv(NATPROD_CSV, usecols=keep_cols)
        df_egfr    = pd.read_csv(EGFR_CSV,    usecols=egfr_cols)

        df_bbknn["query_score"]    = None
        df_bbknn["result_score"]   = None
        df_natprod["query_score"]  = None
        df_natprod["result_score"] = None
        df_egfr["chemical_class"]  = EGFR_CLASS

        df = pd.concat([
            df_bbknn,
            df_natprod,
            df_egfr
        ])

        df = (df[(df.in_size == latent_size) & (df.out_size == latent_size)]
                .reset_index(drop=True))
        df["cos_quantile"]  = _rank_q(df["cosine_similarity"])
        df["tani_quantile"] = _rank_q(df["tanimoto_similarity"])
        df["quantile_diff"] = df.cos_quantile - df.tani_quantile

        df.to_csv(df_path, index=False)
        console.log(f"[green]✓ wrote {df_path.name}")

    # ── correlation cut-curves ───────────────────────────────────────────
    if corr_path.exists():
        z = np.load(corr_path)
        q_cos = tuple(z[f"cos_{k}"] for k in ("q", "p", "s"))
        q_tan = tuple(z[f"tan_{k}"] for k in ("q", "p", "s"))
        p_total = z["p_total"]
        s_total = z["s_total"]
    else:
        console.log("[cyan]calculating correlation curves")
        q_cos = _corr_cut(df, "cos_quantile",
                          "cosine_similarity", "tanimoto_similarity", cuts)
        q_tan = _corr_cut(df, "tani_quantile",
                          "tanimoto_similarity", "cosine_similarity", cuts)
        p_total = np.array(pearsonr(df["cosine_similarity"], 
                                    df["tanimoto_similarity"]).statistic)
        s_total = np.array(spearmanr(df["cosine_similarity"], 
                                     df["tanimoto_similarity"]).statistic)
        np.savez_compressed(
            corr_path,
            cos_q=q_cos[0], cos_p=q_cos[1], cos_s=q_cos[2],
            tan_q=q_tan[0], tan_p=q_tan[1], tan_s=q_tan[2],
            p_total=p_total, s_total=s_total
        )
        console.log(f"[green]✓ wrote {corr_path.name}")

    return df, (q_cos, q_tan, p_total, s_total)
