"""
CLI
---
python -m src.benchmarks.best_hyperparams [--k_best 3]
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

from collections import defaultdict
import numpy as np
import pandas as pd

from .constants import SWEEPS, SWEEP_DIR, TOPKS, RANK_KS, console 
from .utils     import parse_run_name, load_one_run, summarise_scores

# ╭────────────────────── main routine ───────────────────────────────────────╮
def gather_all(dir_dict: dict) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Returns
    -------
    summary_df    one row / (method, run_id, replica)
    params_store  {(method, run_id, replica): params_dict}
    """
    records, param_store, df_store = [], {}, {}
    for method, root in dir_dict.items():
        if not root.exists():
            console.log(f"[yellow]⚠  {root} missing - skipped")
            continue
        for run_dir in root.iterdir():
            df, prm = load_one_run(run_dir)

            if df is None:
                continue

            if "is_bb" in df:                              # post-process BB flags
                df["all_scores"] = df["score"]
                df["score"] = df["score"] * (~df["is_bb"])
                df["score"] = (df["score"]
                            .replace(0, np.nan)
                            .ffill()
                            .replace(np.nan, 0))


            name, gid, rep = parse_run_name(run_dir.name)
            run_id = f"{name}_{gid}"

            rec = dict(method=method, run_id=run_id, run_replica=rep)
            summ = summarise_scores(df)
            if "score_name" in prm:
                summ["score"] = prm["score_name"]
            else:
                summ["score"] = prm["plugin"]
            rec.update(summ)
            param_key = hash(json.dumps({k:v for k,v in prm.items()
                                                if k not in ["rng_seed", "run_name"]}))
            rec["param_key"] = param_key
            records.append(rec)

            param_store[param_key] = prm
            # param_store[(method, run_id, rep)] = prm
            df_store[(method, run_id, rep)] = df

    summary = pd.DataFrame(records)
    console.log(f"[cyan]Collected {len(summary)} run replicas.")
    return summary, param_store, df_store 


def rank_and_select(summary: pd.DataFrame,
                    params: Dict[str, dict],
                    k_best: int = 3
                    ) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """
    • average by method+grid   • rank   • pick k_best per method
    """
    # replica-mean aggregation ------------------------------------------------
    agg_cols = ["runtime"] + [f"top{k}" for k in TOPKS]
    agg_df = (summary
              .groupby(["method", "param_key"])[agg_cols]
              .agg(["mean", "std"]))
    agg_df.columns = ["_".join(c).rstrip("_") for c in agg_df.columns]
    agg_df = agg_df.reset_index()

    # ranking (Top-1,Top-10,Top-100) -----------------------------------------
    rank_cols = [f"top{k}_mean" for k in RANK_KS]
    ranks = np.stack([agg_df[c].rank(ascending=False) for c in rank_cols]).mean(0)
    agg_df["rank"] = ranks

    # best-of-k per method ----------------------------------------------------
    best_rows, best_params = [], defaultdict(list)
    for meth in agg_df["method"].unique():
        sub = agg_df[agg_df.method == meth].nsmallest(k_best, "rank")
        best_rows.append(sub)
        for _, row in sub.iterrows():
            key = row["param_key"]
            best_p = params[key]
            best_p["rank"] = row["rank"]
            best_params[meth].append(best_p)

    best_df = pd.concat(best_rows, ignore_index=True)
    best_df = best_df.sort_values(["method", "rank"])
    return best_df, best_params, agg_df


def save_outputs(best_df: pd.DataFrame,
                 best_params: Dict[str, List[dict]],
                 aggregate_df: pd.DataFrame,
                 out_dir: Path):
    best_df.to_csv(out_dir / "best_hyperparams.csv", index=False)
    aggregate_df.to_csv(out_dir / "benchmarks_aggregate_stats.csv", index=False)
    for meth, plist in best_params.items():
        out = out_dir / f"{meth}_best_params.json"
        json_str = json.dumps(plist, indent=2)
        out.write_text(json_str)
    console.log(f"[green]✓ wrote summary CSVs & params → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_best", type=int, default=3,
                    help="How many top grids to keep per method")
    args = ap.parse_args()

    summ, prm, dfs = gather_all(SWEEPS)
    best, best_params, agg = rank_and_select(summ, prm, k_best=args.k_best)
    save_outputs(best, best_params, agg, SWEEP_DIR)


if __name__ == "__main__":
    main()
