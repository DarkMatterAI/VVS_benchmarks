"""
bbknn_eval.py
─────────────────────────────────────────────────────────────────────
Entry-point for the “plain” BB-KNN benchmark (Enamine-Assembled).

* draws **N** random products as queries
* enumerates reactions & candidates with BB-KNN
* scores cosine + Tanimoto similarities
* writes results CSV →
      {BLOB}/internal/bbknn/bbknn_eval/data.csv
"""

from __future__ import annotations
import argparse, time, yaml
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table   import Table
import duckdb 

from .reconstruct_validation import build_validation_csv
from .reaction_utils import _remove_stereo
from .bbknn     import BBKNN, eval_bbknn_similarity
from .constants import (
    BLOB,
    DB_PATH,
    TBL_NAME,
    ASSM_CSV,
    BB_EMB_PTH,
    BB_CSV_PTH,
    EMB_MODEL_NM,
    DECOMP_MODEL_NM,
    DEVICE,
)

console = Console()
CFG_PATH = Path(__file__).parent / "config.yaml"

VALID_CSV = build_validation_csv()

# ╭──────────────────────── argument parsing & defaults ───────────────────╮
def _load_yaml_defaults() -> dict:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("bbknn_eval", {})

def _parse_cli(defaults: dict):
    p = argparse.ArgumentParser()
    p.add_argument("--n_queries",  type=int, default=defaults.get("n_queries", 10))
    p.add_argument("--sizes",      nargs="+", type=int,
                   default=defaults.get("sizes", [32, 64, 128, 256, 512, 768]))
    p.add_argument("--k",          type=int, default=defaults.get("k", 5))
    p.add_argument("--cpus",       type=int, default=defaults.get("cpus", 60))
    p.add_argument("--iterations", type=int, default=defaults.get("iterations", 1))
    return p.parse_args()
# ╰─────────────────────────────────────────────────────────────────────────╯

def _sample_smiles(db_path: Path, table_name: str, sample_size: int):
    con = duckdb.connect(str(db_path), read_only=True)
    sql = f"""
        SELECT item
        FROM {table_name}
        USING SAMPLE reservoir({sample_size})
    """
    tbl = con.execute(sql)
    df = tbl.df()
    return df

def main():
    args = _parse_cli(_load_yaml_defaults())

    # pretty summary table
    tbl = Table(title="BB-KNN Eval Configuration")
    tbl.add_column("Parameter"); tbl.add_column("Value", style="cyan")
    for k, v in vars(args).items():
        tbl.add_row(k, str(v))
    console.print(tbl)

    console.log("[bold]Loading models …")
    bbknn = BBKNN(
        EMB_MODEL_NM, 
        DECOMP_MODEL_NM,
        BB_EMB_PTH,
        BB_CSV_PTH,
        DEVICE,
    )

    # Validation set ----------------------------------------------------------
    val_full = (pd.read_csv(VALID_CSV)
                  .groupby(["bb1","bb2"]).head(1))           # unique BB-pairs
    val_sample = val_full.sample(
        n=args.iterations*args.n_queries, random_state=0
    ).rename(columns={"product":"query"})
    chunks_val = [val_sample[i:i+args.n_queries]
                  for i in range(0, len(val_sample), args.n_queries)]

    # D4 sample ---------------------------------------------------------------
    d4_df = _sample_smiles(DB_PATH, TBL_NAME, args.iterations*args.n_queries
            ).rename(columns={"item":"query"})
    chunks_d4 = [d4_df[i:i+args.n_queries]
                 for i in range(0, len(d4_df), args.n_queries)]

    # -------------------------------------------------------------------------
    console.log(f"Running {args.iterations} iterations (both Validation & D4)")
    for iteration, (df_val, df_d4) in enumerate(zip(chunks_val, chunks_d4), 1):
        for label, q_df in [("Enamine Validation", df_val),
                            ("D4",                 df_d4)]:

            console.log(f"[bold]{label} — iter {iteration}/{args.iterations}")
            q_df = q_df.reset_index(drop=True)
            q_df["query"] = q_df["query"].map(_remove_stereo)
            query_smiles = q_df["query"].tolist()


            print(len(query_smiles))

            # ── BB-KNN enumeration ──────────────────────────────────────────
            console.log("[bold]Running BB-KNN + reaction enumeration …")
            t0 = time.time()
            df = bbknn.smiles_query_multiscale(
                query_smiles,
                k=args.k,
                input_sizes=args.sizes,
                output_sizes=args.sizes,
                reaction_cpus=args.cpus,
            )
            console.log(f"[green]✓ enumeration finished in {time.time()-t0:.1f}s  "
                        f"→ {len(df):,} products")

            # ── similarity evaluation ───────────────────────────────────────
            console.log("[bold]Calculating similarities …")
            t1 = time.time()
            df = eval_bbknn_similarity(df, bbknn)
            df = df.drop(["query_idx", "result_idx"], axis=1)
            df["chemical_class"] = label
            print(df.shape)
            console.log(f"[green]✓ similarity done in {time.time()-t1:.1f}s")

            # ── save CSV ────────────────────────────────────────────────────
            out_path = BLOB / "internal" / "bbknn" / "bbknn_eval" / "data.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, mode="a", header= not out_path.exists(), index=False)

            # df.to_csv(out_path, index=False)
            console.log(f"[bold green]Results saved → {out_path}")

if __name__ == "__main__":
    main()
