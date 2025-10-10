from __future__ import annotations
import argparse, yaml, json, time, random, gc
from pathlib import Path
from typing  import Dict, Any, Optional

import numpy as np, pandas as pd, torch

from vvs_local.constants     import BLOB, D4_DB_PATH, D4_TBL_NAME, console
from vvs_local.model_store   import ModelStore
from vvs_local.reaction_assembly import ReactionAssembly
from vvs_local.bbknn         import BBKNN
from vvs_local.knn           import KNN
from vvs_local.vvs           import VVS
from vvs_local.score_rpc     import RPCScore
from .grid_utils             import expand_grid
from .utils                  import(_select_batch, 
                                    _smarts_list, 
                                    _make_bbknn, 
                                    _make_knn)

import torch.nn.functional as F

# ---------- VVS single run -------------------------------------------
def run_single(params: Dict[str, Any],
               store : ModelStore,
               engine_type: str,
               root_out : Path,
               run_name : str) -> None:
    
    # -------- engine ---------------------------------------------------
    if engine_type == "bbknn":
        assembly = ReactionAssembly(rxn_smarts=_smarts_list(),
                                    **params["engine_params"]["reaction_pool"])
        engine   = _make_bbknn(params["engine_params"], assembly, store)
        dataset  = "enamine"
    else:
        engine   = _make_knn(params["engine_params"])
        dataset  = "d4_138m"

    # -------- scorer / VVS --------------------------------------------
    scorer = RPCScore(
        plugin        = params["plugin"],
        timeout       = params["timeout"],
        batch_size    = params["score_batch_size"],
        concurrency   = params["concurrency"],
        device        = params["device"],
        budget        = params["inference_budget"],
        runtime_limit = params["runtime_limit"],
        check_lmt     = False # limit checked in VVS class instead
    )
    vvs = VVS(engine, scorer, update_type=params["update_type"])

    # -------- learning-rate tensor ------------------------------------
    lrs = torch.tensor(params["lrs"])

    # ---------------- primary optimisation loop --------------------
    start_ts     = time.time()
    round_id     = 0
    prod_df_round: Optional[pd.DataFrame] = None
    store_results: list[pd.DataFrame] = []   # collect every round’s DF

    try:
        while True:
            # --- pick next batch --------------------------------------
            round_id += 1
            smiles = _select_batch(prod_df_round,
                                   params["vvs_batch_size"],
                                   params["exploit_percent"],
                                   params["db_path"],
                                   params["db_table"])
            queries = store.encode_compress(smiles).cpu()

            # --- VVS search -------------------------------------------
            _ = vvs.search(queries, params["iterations"], lrs, params["k_nn"], check_lmt=True)

            # after *iterations* we have the last product DataFrame
            prod_df_round = vvs.result_dfs[-1]           # VVS stores each iter
            for df in vvs.result_dfs:
                df.assign(round=round_id)
                store_results.append(df)

            console.log(f"[green]Round {round_id:02d}  "
                        f"queries={len(smiles):4d}  "
                        f"budget_used={scorer.used:,}")
    except RuntimeError:
        console.log(f"[green]Run exited")
        # inference / time budget hit 
        pass

    # -------- collect scorer log --------------------------------------
    runtime = time.time() - start_ts
    score_log = pd.DataFrame(scorer.records)
    score_log.to_csv(root_out / "score_log.csv", index=False)

    if store_results:
        all_prod = pd.concat(store_results, ignore_index=True)
        all_prod.to_parquet(root_out / "prod_df.parquet", index=False)

    params["db_path"] = str(params["db_path"])
    with open(root_out / "params.json","w") as fh:
        json.dump(params, fh, indent=2)

    summary = {
        "run_name"        : run_name,
        "dataset"         : dataset,
        "engine"          : engine_type,
        "score"           : params["plugin"],
        "inference_budget": params["inference_budget"],
        "n_inference"     : scorer.used,
        "n_results"       : score_log.shape[0],
        "runtime"         : runtime,
    }

    summ = root_out.parent / "summary.csv"
    pd.DataFrame([summary]).to_csv(
        summ, mode="a", header=not summ.exists(), index=False
    )

    vvs.close()
    del vvs 
    del engine 
    del scorer 
    console.log(f"[green]✓ {run_name} finished in " \
                f"{int(runtime // 60):02}:{int(runtime % 60):02}\n\n")

# ---------- main ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",
                    default="bbknn_debug.yaml")
    ap.add_argument("--run_type", default="debug")
    args = ap.parse_args()

    cfg_root = Path(f"/code/hyperparam_configs/{args.cfg}")
    runs_yaml = yaml.safe_load(cfg_root.read_text())

    # ----- global fixed things ----------------------------------------
    engine_type = runs_yaml["engine_params"].get("engine","bbknn").lower()

    # create a single ModelStore shared by all runs
    embed_size  = (runs_yaml["engine_params"]["embedding"]
                   .get("input_size",
                        runs_yaml["engine_params"]["embedding"].get("size")))
    store = ModelStore(embed_size)

    out_root = BLOB / "internal" / "benchmarks" / "vvs_local" / args.run_type
    out_root.mkdir(parents=True, exist_ok=True)

    # iterate over YAML blocks -----------------------------------------
    for run_block in runs_yaml["runs"]:
        base_name, configs = expand_grid(run_block)
        replicas = run_block["run_params"]["replicas"]
        n_runs = len(configs) * replicas 
        run_count = 1

        for cfg_id, prm in configs:
            for r in range(replicas):
                run_name = f"{base_name}_{cfg_id}_{r+1}"
                run_dir  = out_root / run_name

                if (run_dir / "score_log.csv").exists():
                    console.log(f"[red]Skipping run {run_name} - results exist")
                    run_count += 1
                    continue 

                run_dir.mkdir(parents=True, exist_ok=True)

                # attach DB paths & engine params for convenience
                prm["db_path"]   = D4_DB_PATH
                prm["db_table"]  = D4_TBL_NAME
                prm["engine_params"] = runs_yaml["engine_params"]

                # RNG seed
                prm["rng_seed"] = random.randrange(1_000_000_000)
                torch.manual_seed(prm["rng_seed"])
                np.random.seed (prm["rng_seed"])

                console.log(f"[green]Starting run {run_name}, {run_count}/{n_runs}")
                print(engine_type)
                run_single(prm, 
                           store,
                           engine_type,
                           run_dir, 
                           run_name)
                run_count += 1
                gc.collect()

    console.rule("[bold green]ALL RUNS COMPLETE")

if __name__ == "__main__":
    main()
