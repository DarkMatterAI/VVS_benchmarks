"""
natural_product_eval.py
─────────────────────────────────────────────────────────────────────
Benchmark BB-KNN on a *natural-products* slice of the Coconut-Lite
database.

Workflow
--------
1.  Load the processed natural-products CSV.
2.  Apply phys-chem / class filters.
3.  Pick the **top-N classes** by frequency and sample
    *M* query molecules per class (default 10 x 10 = 100).
4.  Enumerate BB-KNN reactions (sizes 32/256/768, *k*=10).
5.  Score cosine & Tanimoto similarity.
6.  Write results to ::

        {BLOB}/internal/bbknn/natural_product_eval/data.csv
"""

from __future__ import annotations
import argparse, time, yaml, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rich.console import Console
from rich.table   import Table

from .bbknn          import BBKNN, eval_bbknn_similarity
from .reaction_utils import _remove_stereo
from .constants      import (
    BLOB,
    NATPROD_CSV,
    BB_EMB_PTH,
    BB_CSV_PTH,
    EMB_MODEL_NM,
    DECOMP_MODEL_NM,
    DEVICE,
)

console  = Console()
CFG_PATH = Path(__file__).parent / "config.yaml"


# ╭──────────────────────── defaults & CLI ───────────────────────────────╮
def _load_yaml_defaults() -> dict:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("natural_product_eval", {})

def _parse_cli(defaults: dict):
    p = argparse.ArgumentParser()
    p.add_argument("--n_classes",        type=int, default=defaults.get("n_classes", 10),
                   help="Number of chemical classes to sample.")
    p.add_argument("--n_per_class",      type=int, default=defaults.get("n_per_class", 10),
                   help="Number of queries per class.")
    p.add_argument("--sizes", nargs="+", type=int,
                   default=defaults.get("sizes", [32, 256, 768]))
    p.add_argument("--k",                type=int, default=defaults.get("k", 10))
    p.add_argument("--cpus",             type=int, default=defaults.get("cpus", 60))
    p.add_argument("--seed",             type=int, default=defaults.get("seed", 42))
    p.add_argument("--iterations",       type=int, default=defaults.get("iterations", 1))
    return p.parse_args()
# ╰────────────────────────────────────────────────────────────────────────╯


def main():
    args = _parse_cli(_load_yaml_defaults())

    # ── pretty config table ─────────────────────────────────────────
    tbl = Table(title="Natural-Product BB-KNN Configuration")
    for k, v in vars(args).items():
        tbl.add_row(k, str(v), style="cyan")
    console.print(tbl)

    # ── load & filter dataset ───────────────────────────────────────
    console.log("[bold]Loading Coconut-Lite dataframe …")
    df = pd.read_csv(NATPROD_CSV)

    console.log("Applying phys-chem filters …")
    df = df[
        (~df.contains_sugar) & 
        (df.number_of_minimal_rings>0) & 
        (df.heavy_atom_count.between(15, 30))
    ]

    # ── pick top-N classes ──────────────────────────────────────────
    top_classes = (
        df.chemical_class.value_counts()
          .nlargest(args.n_classes)
          .index
    )
    df = df[df.chemical_class.isin(top_classes)]

    # ── sample queries per class ────────────────────────────────────
    total_per_cls = args.n_per_class * args.iterations
    console.log(f"Sampling {total_per_cls} per class …")
    queries = (df.groupby("chemical_class")
                 .sample(n=total_per_cls, random_state=args.seed)
                 .reset_index(drop=True)
                 .sample(frac=1)
                 .reset_index(drop=True)
                 [["item", "external_id", "chemical_class"]])

    # strip stereo (improves decomposer robustness)
    queries["item"] = queries["item"].map(_remove_stereo)

    # quick lookup dicts for later join
    cls_map  = dict(zip(queries.item, queries.chemical_class))
    id_map   = dict(zip(queries.item, queries.external_id))

    print(queries.shape)
    chunk_size = args.n_per_class * args.n_classes
    print(chunk_size)
    q_chunks = [queries.iloc[i*chunk_size:(i+1)*chunk_size]
                for i in range(args.iterations)]
    print([i.shape for i in q_chunks])
    
    # ── initialise BB-KNN ───────────────────────────────────────────
    console.log("[bold]Loading BB-KNN models …")
    bbknn = BBKNN(
        EMB_MODEL_NM, 
        DECOMP_MODEL_NM,
        BB_EMB_PTH,
        BB_CSV_PTH,
        DEVICE,
    )
    
    # ---------- iterate over chunks ----------------------------------------
    out_path = (BLOB / "internal" / "bbknn" /
                "natural_product_eval" / "data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.log(f"Running {args.iterations} iterations "
                f"(chunk size = {chunk_size} / class)")

    for idx, chunk in enumerate(q_chunks, 1):
        q_smiles = chunk.item.tolist()
        console.rule(f"[bold]Iteration {idx} / {args.iterations}")

        t0 = time.time()
        prod_df = bbknn.smiles_query_multiscale(
            q_smiles,
            k=args.k,
            input_sizes=args.sizes,
            output_sizes=args.sizes,
            reaction_cpus=args.cpus,
        )
        console.log(f"[green]✓ enumeration in {time.time()-t0:.1f}s  "
                    f"→ {len(prod_df):,} products")

        # attach meta-data
        prod_df["chemical_class"] = prod_df["query"].map(cls_map)
        prod_df["external_id"]    = prod_df["query"].map(id_map)

        # similarity --------------------------------------------------------
        t1 = time.time()
        prod_df = eval_bbknn_similarity(prod_df, bbknn
                    ).drop(columns=["query_idx","result_idx"])
        console.log(f"[green]✓ similarity in {time.time()-t1:.1f}s")

        # append to CSV -----------------------------------------------------
        header_needed = not out_path.exists()
        prod_df.to_csv(out_path, mode="a",
                       header=header_needed, index=False)
        console.log(f"[cyan]Appended results to {out_path.name}")

if __name__ == "__main__":
    main()
