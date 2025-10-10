# rad_runner.py
"""
Run RAD traversals according to the sweep YAML provided on the command line.

Example
-------
python -m src.rad_runner --cfg /code/sweep_space.yaml --run_type sweep
python -m src.rad_runner --cfg /code/debug_space.yaml --run_type debug
"""

from __future__ import annotations
import argparse, json, os, time, uuid, multiprocessing as mp
from pathlib import Path
from itertools import product
from typing import Any, Dict, List
from datetime import datetime
from functools import cache
import duckdb 
import pandas as pd 
from multiprocessing import Manager

import yaml
import numpy as np
from rich.console import Console
from rich.table import Table

from usearch.index import Index
from rad.traverser import RADTraverser
from .score_rpc import BatchScorer

console = Console()
INDICES = {}
EXCHANGE   = os.getenv('RABBITMQ_EXCHANGE_NAME')
BLOB = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
RAD_DIR = BLOB / "internal" / "processed" / "rad"
DB_DIR  = BLOB / "internal" / "processed"

# ════════════════════════════════════════════════════════════════════════════
# yaml helpers
# ════════════════════════════════════════════════════════════════════════════

def _expand_grid(param_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Cartesian-product expansion exactly like in the Synthemol benchmark.

    Scalars stay as-is; lists generate sweep axes.
    """
    names, axes = zip(*[
        (k, v if isinstance(v, list) else [v])
        for k, v in param_dict.items()
    ]) if param_dict else ([], [])

    return [dict(zip(names, combo)) for combo in product(*axes)] or [{}]


def parse_yaml(path: Path) -> List[Dict[str, Any]]:
    """
    Returns **one job-dict per (grid-combo x score x replica)**.

    Each job-dict has keys
        • name          unique string
        • dataset       eg "d4_138m"
        • score_cfg     dict with {score_name, score_timeout}
        • run_cfg       dict (inference_budget, num_proc, replica_id, …)
    """
    raw_runs = yaml.safe_load(path.read_text())["runs"]
    jobs: List[Dict[str, Any]] = []

    for section in raw_runs:
        base          = section["name"]
        grid_confs    = _expand_grid(section.get("params", {}))
        score_confs   = section.get("score_params", [])
        run_const     = section.get("run_params", {})
        replicas      = run_const.get("replicas", 1)

        g_id = 0
        for grid in grid_confs:                       # ➊ sweep grid
            g_id += 1
            for s_id, score_cfg in enumerate(score_confs, 1):  # ➋ each score
                for r in range(replicas):             # ➌ replicas
                    job_name = f"{base}_{g_id:02d}-{s_id:02d}_{r}"
                    job = {
                        "name"      : job_name,
                        "dataset"   : grid["dataset"],     # the only grid field we use here
                        "score_cfg" : score_cfg,           # single dict
                        "run_cfg"   : run_const |          # copy + add seed/replica
                                        {"replica_id": r,
                                         "seed": int(grid.get("seed", 0)) + r}
                    }
                    jobs.append(job)
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# RAD helper functions
# ════════════════════════════════════════════════════════════════════════════
def _load_hnsw(dataset: str, debug: bool) -> Index:
    fname = f"{dataset}{'_debug' if debug else ''}.hnsw"
    if fname in INDICES:
        return INDICES[fname]

    hnsw_path  = RAD_DIR / fname
    json_path  = RAD_DIR / fname.replace('.hnsw', '.json')

    for path in (hnsw_path, json_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with open(json_path, 'r') as f:
        hnsw_json = json.load(f)

    hnsw_config = hnsw_json['config']
    hnsw = Index(
        ndim=hnsw_config["fp_size"], 
        dtype="b1", 
        metric="tanimoto",
    )
    hnsw.load(str(hnsw_path))
    INDICES[fname] = hnsw
    return hnsw


class RPCScore:
    """
    Parameters
    ----------
    • db_file    : Path to DuckDB with a table that has    item   column.
    • table_name : Name of that table.
    • budget     : Hard limit on # remote evaluations.
    • plugin     : score-plugin name   (becomes part of routing-key).
    • timeout    : seconds to wait for an RPC reply before raising.
    """
    def __init__(self,
                 *,
                 db_file      : str,
                 table_name   : str,
                 budget       : int,
                 plugin       : str,
                 timeout      : int,
                 shared_log,
                 device       : str,
                 runtime_limit: int=None):

        self.budget   = budget
        self.used     = 0
        self.shared_records = shared_log
        self.local_records  = []
        self.device = device

        # --- DuckDB (read-only) ------------------------------------------
        self.db_file = db_file
        self.table = table_name
        self.con = None

        # --- open RPC channel once (BatchScorer manages reply-queue) -----
        self.exchange = EXCHANGE
        self.plugin = plugin
        self.timeout = timeout
        self.runtime_limit = runtime_limit
        self.start_time    = None
        self.rcp = None

    # ---------------------------------------------------------------------
    def open_conns(self):
        group_key = 'benchmark_score' if self.device=='cpu' else 'benchmark_score_gpu'
        self.con   = duckdb.connect(self.db_file, read_only=True)
        self.rpc = BatchScorer(
            self.exchange,
            routing_key = f"request.{group_key}.score.{self.plugin}.internal.internal",
            timeout     = self.timeout
        )
    
    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None
        if self.rpc is not None:
            self.rpc.close()
            self.rpc = None

    # ---------------------------------------------------------------------
    @cache
    def _score_smiles(self, smiles: str) -> float | None:
        """RPC call → float | None (invalid / timeout)."""
        res = self.rpc.score_batch([{"item_data": {"item": smiles}}])[0]
        self.used += 1
        if isinstance(res, Exception) or res is None or not res.get("valid", False):
            return None 
        return float(res["score"])

    # ---------------------------------------------------------------------
    def _check_limits(self):
        if self.used >= self.budget:
            raise RuntimeError("call-budget exhausted")

        if self.runtime_limit is not None:
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            elif now - self.start_time > self.runtime_limit:
                raise RuntimeError("runtime limit exceeded")

    def __call__(self, key: int) -> float | None:
        self._check_limits()
            
        if self.con is None:
            self.open_conns()

        # note: rowid col is 0-based even though standard DuckDB is 1-based
        # we deliberatly search `key`, not `key+1`
        row_id = int(key)
        smile = self.con.execute(
            f"SELECT item FROM {self.table} WHERE rowid = ?", [row_id]
        ).fetchone()[0]
        score = self._score_smiles(smile)

        score_data = {"ts": str(datetime.now()),
                      "item": smile,
                      "score": score}
        self.local_records.append(score_data)
        if self.shared_records is not None:
            self.shared_records.append(score_data)

        # None scores (ie failed docking) converted to inf
        # Score sign flipped because RAD expects smaller = better 
        score = 1e6 if score is None else -1*score

        return score

def run_single(job_json: dict,
               run_type: str) -> None:
    console.log(job_json)

    run_name   = job_json["name"]
    dataset    = job_json["dataset"]
    budget     = job_json["run_cfg"]["inference_budget"]
    plugin     = job_json["score_cfg"]["score_name"]
    timeout    = job_json["score_cfg"]["score_timeout"] 
    time_limit = job_json["run_cfg"]["runtime_limit"]
    n_workers  = 8
    device = 'gpu' if plugin == 'erbb1_mlp' else 'cpu'
    debug     = run_type=="debug"

    console.log(f"[bold blue]▶  {run_name}  dataset={dataset}  budget={budget}")

    # -------- data paths --------------------------------------------------
    db_file   = DB_DIR / dataset / "database.db"
    table     = dataset          # table name == dataset in our prep script
    hnsw      = _load_hnsw(dataset, debug)
    print(table)

    # -------- scorer ------------------------------------------------------
    shared_log = Manager().list()
    scorer = RPCScore(db_file=str(db_file),
                      table_name=table,
                      budget=budget+1, # rad will stop itself when budget reached
                      plugin=plugin,
                      timeout=timeout,
                      shared_log=shared_log,
                      device=device)

    # -------- RAD traverser ----------------------------------------------
    trav = RADTraverser(hnsw=hnsw,
                        scoring_fn=scorer)
    
    # clear cache from previous run 
    for k in ("pq", "scored:list", "scored:set", "visited"):
        trav.redis_client.delete(k)

    results = []
    for key, score in trav.scored_set:
        results.append((key,score))
    assert len(results)==0, (f"Found {len(results)} at start of fresh run, " \
                             f"RAD cache from previous run failed to clear")

    t0 = time.time()
    console.log(f"[bold blue]▶  {run_name}  starting RAD run")
    trav.prime()
    scorer.close()
    console.log(f"[bold blue]▶  {run_name}  RAD primed")
    console.log(f"[bold blue]▶  {run_name}  starting RAD traversal")
    try:
        trav.traverse(n_workers=n_workers, n_to_score=budget, timeout=time_limit)
    finally:
        results = []
        for key, score in trav.scored_set:
            results.append((key,score))

        # clear cache 
        for k in ("pq", "scored:list", "scored:set", "visited"):
            trav.redis_client.delete(k)
        trav.shutdown()
        scorer.close()

    records = list(shared_log) or scorer.local_log
    console.log(scorer.used, len(records), len(results))
    runtime = time.time() - t0
    console.log(f"[green]✓  {run_name}  finished  in {runtime/60:.1f} min")

    # ------------------------------------------------------------------ persist
    out_dir = (BLOB / "internal" / "benchmarks" /
               "rad" / run_type / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "score_log.csv", index=False)

    with open(out_dir / "params.json", "w") as f:
        json.dump(job_json, open(out_dir / "params.json", "w"))

    # ---- roll-up summary (append) ----------------------------------------
    summary = {
        "run_name": run_name,
        "dataset": dataset,
        "score": plugin,
        "inference_budget": budget,
        "n_inference": df.shape[0],
        "n_results": df.shape[0],
        "runtime": runtime,
    }
    summ_path = out_dir.parent / "summary.csv"
    pd.DataFrame([summary]).to_csv(
        summ_path, mode="a", header=not summ_path.exists(), index=False
    )
    console.log(f"[✓] finished {run_name} - results in {out_dir}")


# ════════════════════════════════════════════════════════════════════════════
# main entry
# ════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--run_type", choices=["debug", "sweep", "final"], default="sweep")
    args = p.parse_args()

    cfg_path = Path(args.cfg)
    jobs = parse_yaml(cfg_path)

    # nice overview table
    tbl = Table(title="RAD jobs")
    tbl.add_column("name")
    tbl.add_column("index")
    tbl.add_column("budget", justify="right")
    for j in jobs:
        tbl.add_row(j["name"], j["dataset"], f"{j['run_cfg']['inference_budget']:,}")
    console.print(tbl)

    for i, job in enumerate(jobs):
        console.log(f"Job {i+1}/{len(jobs)}")
        run_single(job, args.run_type)
    console.log(f"[✓] All Jobs Complete")


if __name__ == "__main__":
    main()
