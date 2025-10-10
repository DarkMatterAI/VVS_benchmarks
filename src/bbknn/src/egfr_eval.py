"""
egfr_eval.py
─────────────────────────────────────────────────────────────────────
Benchmark BB-KNN on top 10 high-affinity EGFR ligands and rescore all
enumerated products with the internal docking-RPC service.

Output CSV:
    {BLOB}/internal/bbknn/egfr_eval/data.csv
"""

from __future__ import annotations
import argparse, time, yaml, random
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rich.console import Console
from rich.table   import Table

from .reaction_utils import _remove_stereo
from .bbknn          import BBKNN, eval_bbknn_similarity, _chunk
from .score_rpc      import RPCScore
from .constants      import (
    BLOB,
    EGFR_CSV,
    BB_EMB_PTH,
    BB_CSV_PTH,
    EMB_MODEL_NM,
    DECOMP_MODEL_NM,
    DEVICE,
)

console  = Console()
CFG_PATH = Path(__file__).parent / "config.yaml"

# ╭──────────────────────── defaults / CLI ───────────────────────────────╮
def _load_defaults() -> dict:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("egfr_eval", {})

def _cli_args(defs):
    p = argparse.ArgumentParser()
    p.add_argument("--n_queries", type=int, default=defs.get("n_queries", 100))
    p.add_argument("--sizes",     nargs="+", type=int, default=defs.get("sizes", [256]))
    p.add_argument("--k",         type=int, default=defs.get("k", 10))
    p.add_argument("--cpus",      type=int, default=defs.get("cpus", 60))
    p.add_argument("--chunk",     type=int, default=defs.get("score_chunk", 500),
                   help="RPC-scoring batch size.")
    p.add_argument("--timeout",   type=int, default=defs.get("rpc_timeout", 400))
    return p.parse_args()
# ╰────────────────────────────────────────────────────────────────────────╯


def main():
    args = _cli_args(_load_defaults())

    # ── pretty config table ─────────────────────────────────────────
    tbl = Table(title="EGFR BB-KNN Configuration")
    for k, v in vars(args).items():
        tbl.add_row(k, str(v), style="cyan")
    console.print(tbl)

    # ── read ligand set & choose top N by pIC50 ─────────────────────
    console.log("[bold]Loading EGFR ligand set …")
    lig_df = pd.read_csv(EGFR_CSV)
    lig_df = lig_df.iloc[lig_df.pIC50.nlargest(args.n_queries).index].reset_index(drop=True)
    lig_df["item"] = lig_df["item"].map(_remove_stereo)

    # ── initialise RPC scorer (docking) ─────────────────────────────
    console.log("[bold]Initialising docking RPC scorer …")
    scorer = RPCScore("docking_6lud", timeout=args.timeout, batch_size=512, device="cpu")

    # ── score the queries themselves (to filter out failures) ───────
    console.log("Scoring query molecules …")
    q_scores = []
    for batch in _chunk(lig_df.item.tolist(), args.chunk):
        q_scores.extend(scorer(batch))
    lig_df["query_score"] = q_scores
    lig_df = lig_df[lig_df.query_score > 0].reset_index(drop=True)
    if lig_df.empty:
        console.log("[red]No queries passed the docking filter - exiting.")
        return
    console.log(f"{len(lig_df)} / {args.n_queries} queries retained.")

    # ── BB-KNN setup ────────────────────────────────────────────────
    console.log("[bold]Loading BB-KNN models …")
    bbknn = BBKNN(
        EMB_MODEL_NM, 
        DECOMP_MODEL_NM,
        BB_EMB_PTH,
        BB_CSV_PTH,
        DEVICE,
    )

    # ── enumeration ────────────────────────────────────────────────
    console.log("[bold]Enumerating reactions …")
    t0 = time.time()
    prod_df = bbknn.smiles_query_multiscale(
        lig_df.item.tolist(),
        k=args.k,
        input_sizes=args.sizes,
        output_sizes=args.sizes,
        reaction_cpus=args.cpus,
    )
    console.log(f"[green]✓ enumeration done in {time.time()-t0:.1f}s  "
                f"→ {len(prod_df):,} products")

    # ── similarity evaluation (RoBERTa / ECFP) ──────────────────────
    console.log("[bold]Similarity evaluation …")
    prod_df = eval_bbknn_similarity(prod_df, bbknn)

    # ── docking scores for queries + products ───────────────────────
    console.log("[bold]Docking scores for products …")
    all_smiles = list(set(prod_df["query"]).union(prod_df["result"]))
    score_map  = {}

    for i, batch in enumerate(_chunk(all_smiles, args.chunk)):
        console.log(f"   batch {i+1} / { (len(all_smiles)+args.chunk-1)//args.chunk }")
        for smi, sc in zip(batch, scorer(batch)):
            score_map[smi] = sc

    scorer.close()

    prod_df["query_score"]   = prod_df["query"].map(score_map)
    prod_df["result_score"]  = prod_df["result"].map(score_map)

    # ── save CSV ────────────────────────────────────────────────────
    out_path = BLOB / "internal" / "bbknn" / "egfr_eval" / "data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prod_df.to_csv(out_path, index=False)
    prod_df = prod_df.drop(["query_idx", "result_idx"], axis=1)
    console.log(f"[bold green]Results saved → {out_path}")


if __name__ == "__main__":
    main()
