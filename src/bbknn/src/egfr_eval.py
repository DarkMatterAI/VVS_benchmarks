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


def main2():
    args = _cli_args(_load_defaults())

    scorer = RPCScore(
        "docking_6lud",
        timeout=args.timeout,
        batch_size=512,
        device="cpu",
    )

    pairs = [
                (
                    "CN1CCC(c2ccc(-c3ccc4ncn(C(C(=O)Nc5nccs5)c5ccccc5)c(=O)c4c3)cc2)CC1",
                    "Cc1ccn2c(N)c(-c3ccc(N(C(=O)c4ccccc4)c4ccc(O)cc4)cc3)nc2c1"
                ),
                (
                    "OCc1ccc(-c2cc3c(NC(CO)c4ccccc4)ncnc3[nH]2)cc1",
                    "O=C(Nc1ccc(-c2nc3cc(Br)ccc3s2)cc1)NC(CCCO)c1ccccc1"
                ),
                (
                    "COc1cc(N2CCC(N(C)C)CC2)ccc1Nc1ncc(C(=O)OC(C)C)c(-c2c[nH]c3ccccc23)n1",
                    "Cc1nc(Nc2ccc(N3CCC(O)CC3)cc2C)c2cnn(-c3cccc(F)c3)c2n1"
                ),
                (
                    "Cn1cnc2cc3c(Nc4cccc(Br)c4)ncnc3cc21",
                    "Cn1ccc2cnc(Nc3cccc(I)c3)nc21"
                ),
                (
                    "CNc1cc2c(Nc3cccc(Cl)c3)ncnc2cn1",
                    "COc1cc2c(Nc3cccc(CO)c3)ccnc2cc1F"
                ),
                (
                    "C#CCNCC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)c(C#N)cnc2cc1OCC",
                    "C=C=CCCNC(=O)Nc1ccc2ccn(Cc3ccc(CO)cc3)c2c1"
                ),
                (
                    "OCC(O)CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
                    "Clc1cc2ncnc(Nc3cncc(Br)c3)c2cc1I"
                ),
                (
                    "COc1cc2ncnc(Nc3cccc(C(F)(F)F)c3)c2cc1OC",
                    "COc1cc2nc(Nc3cc(C(F)(F)F)ccc3Cl)nc(N)c2cc1OC"
                ),
                (
                    "C=CC(=O)Nc1nc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1C=CCCN1CCOCC1",
                    "C=CC1(CC)COCCN1c1ccc(-c2ncn(Cc3ccccc3)c2N)cc1Cl"
                ),
                (
                    "Oc1cc2ncnc(Nc3cccc(Br)c3)c2cc1O",
                    "Clc1cc2ncnc(Cl)c2cc1NCCc1cccc(Br)c1"
                ),
            ]

    pair_scores = scorer.score_pairs(pairs, save_dir="egfr_pairs")

    # ── report ──────────────────────────────────────────────────
    for i, ((q_smi, r_smi), (q_sc, r_sc)) in enumerate(zip(pairs, pair_scores)):
        console.log(
            f"Pair {i}: query={q_sc:+.2f}  result={r_sc:+.2f}  "
            f"Δ={r_sc - q_sc:+.2f}"
        )

    scorer.close()


if __name__ == "__main__":
    # main()
    main2()
