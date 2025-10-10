"""
rxnflow_runner.py
Run one or many RxnFlow jobs described by a YAML sweep file.
"""

from __future__ import annotations
import argparse, yaml, random, math, time, json, gc 
from pathlib import Path
import torch, numpy as np
import pandas as pd

from rdkit          import Chem 
from rdkit.Chem     import Mol as RDMol
from gflownet       import ObjectProperties
from rxnflow.config import Config, init_empty
from rxnflow.base   import BaseTask, RxnFlowTrainer

from .env_builder    import ensure_env
from .yaml_expander  import expand_grid
from .score_rpc      import RPCScore
from .constants      import (ENV_DIR, ENV_BASE, BLOB, console)

# ────────────────────────────────────────────────────────────────
ensure_env()

class RPCTask(BaseTask):
    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, torch.Tensor]:
        smiles = [Chem.MolToSmiles(i) for i in mols]
        scores = self.scorer(smiles)
        scores = torch.tensor(scores).float()
        is_valid = (~scores.isinf())
        scores = scores.reshape(-1, 1)[is_valid]
        return ObjectProperties(scores), is_valid

class RPCTrainer(RxnFlowTrainer):
    def setup_task(self):
        self.task = RPCTask(self.cfg)

    def set_scorer(self, scorer):
        # the task object is created in `setup_task`
        self.task.scorer = scorer


# ----------------------------------------------------------------------
def make_trainer(cfg_dict: dict, inference_budget: int) -> RPCTrainer:
    # 1.2 padding to ensure we run full inference budget 
    # (ie there may be duplicates across batches)
    steps      = int(1.2 * math.ceil(inference_budget / cfg_dict["batch_size"]))
    cfg        = init_empty(Config())
    cfg.log_dir               = str(ENV_BASE / "logs" / cfg_dict["run_name"])
    cfg.env_dir               = str(ENV_DIR)
    cfg.overwrite_existing_exp= True
    cfg.num_training_steps    = steps
    cfg.checkpoint_every      = steps
    cfg.print_every           = 5
    cfg.num_workers_retrosynthesis = 4

    # algorithm tweaks -------------------------------------------------
    cfg.algo.action_subsampling.sampling_ratio = cfg_dict["act_subsample_ratio"]
    cfg.cond.temperature.sample_dist = "uniform"
    cfg.cond.temperature.dist_params = [0,64]
    cfg.algo.train_random_action_prob = cfg_dict["rand_act_prob"]
    cfg.algo.max_len           = 2
    cfg.algo.num_from_policy   = cfg_dict["batch_size"]

    # replay buffer
    cfg.replay.use         = cfg_dict["replay_batch_size"]>0
    cfg.replay.capacity    = cfg_dict["batch_size"] * 200
    cfg.replay.warmup      = int(cfg_dict["warmup_percent"]*inference_budget)
    cfg.replay.num_from_replay = cfg_dict["replay_batch_size"]

    return RPCTrainer(cfg), steps*cfg_dict["batch_size"]+1

# ----------------------------------------------------------------------
def run_single(run_type: str, run_name: str, prm: dict):
    console.rule(f"[bold cyan]{run_name}")

    random.seed(prm["rng_seed"])
    np.random.seed(prm["rng_seed"])
    torch.manual_seed(prm["rng_seed"])

    trainer, inference_budget = make_trainer(prm, prm["inference_budget"])

    scorer = RPCScore(plugin        = prm["score_name"],
                      timeout       = prm["timeout"],
                      batch_size    = prm["score_batch_size"],
                      concurrency   = prm["concurrency"],
                      device        = prm["device"],
                      budget        = inference_budget,
                      runtime_limit = prm["runtime_limit"])

    trainer.task.scorer = scorer
    t0 = time.time()
    try:
        trainer.run()
    except RuntimeError as e:
        console.log(f"[red]run stopped: {e}")
    runtime = time.time() - t0
    scorer.close()

    # ------------ persist ---------------------------------------------
    out_dir = (BLOB / "internal" / "benchmarks" /
               "rxnflow" / run_type / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(scorer.records).to_csv(out_dir / "score_log.csv", index=False)
    with open(out_dir / "params.json","w") as fh:
        json.dump(prm, fh, indent=2)

    summary = {
        "run_name" : run_name,
        "dataset": "enamine",
        "score"    : prm["score_name"],
        "inference_budget": prm["inference_budget"],
        "n_inference": scorer.used,
        "n_results" : len(scorer.records),
        "runtime"   : runtime
    }

    summ = out_dir.parent / "summary.csv"
    pd.DataFrame([summary]).to_csv(
        summ, mode="a", header=not summ.exists(), index=False
    )
    scorer.close()
    del trainer 
    gc.collect()
    torch.cuda.empty_cache()
    console.log(f"[green]✓ finished {run_name}")

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--run_type", default="debug",
                    choices=["debug","sweep","final"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    tasks = []
    for block in cfg["runs"]:
        base, configs = expand_grid(block)
        rpl = block["run_params"]["replicas"]
        for cfg_id, prm in configs:
            for r in range(rpl):
                run_name = f"{base}_{cfg_id}_{r+1}"
                prm = prm.copy() | {"run_name": run_name,
                                    "rng_seed": random.randrange(1_000_000_000)}
                tasks.append((args.run_type, run_name, prm))

    for t in tasks:
        run_single(*t)

if __name__ == "__main__":
    main()
