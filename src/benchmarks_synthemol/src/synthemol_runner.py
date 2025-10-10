import os, json, time, uuid
from pathlib import Path
from datetime import datetime
from functools import cache
import pandas as pd
import numpy as np 
from synthemol.generate import Generator
from synthemol.reactions import (
    load_and_set_allowed_reaction_building_blocks,
    set_all_building_blocks,
)
import pickle, glob
from .score_rpc import BatchScorer

BLOB          = Path("/code/blob_store").resolve()
ENAMINE_DIR   = BLOB / "internal" / "processed" / "enamine"
CSV_PATH      = ENAMINE_DIR / "data.csv"
RXN_PKL       = next(Path(ENAMINE_DIR).glob("*_reactions.pkl"))
RXN_BB_PKL    = next(Path(ENAMINE_DIR).glob("*_reaction_to_bb.pkl"))

# -------------------------------------------------------------------------- data
enamine_df = pd.read_csv(CSV_PATH)
bb_smiles_to_id = dict(zip(enamine_df.item, enamine_df.external_id))
bb_id_to_smiles = dict(zip(enamine_df.external_id, enamine_df.item))

with open(RXN_PKL, "rb") as f:
    rxn_set = pickle.load(f)

set_all_building_blocks(rxn_set, set(bb_smiles_to_id))
load_and_set_allowed_reaction_building_blocks(rxn_set, RXN_BB_PKL)

# ----------------------------------------------------------------------- scorer
EXCHANGE   = os.getenv('RABBITMQ_EXCHANGE_NAME')

class RPCScore:
    def __init__(self, 
                 budget, 
                 plugin, 
                 timeout, 
                 device="cpu",
                 runtime_limit=None):
        group_key = 'benchmark_score_gpu' if device=='gpu' else 'benchmark_score'
        self.rpc = BatchScorer(
            exchange=EXCHANGE,
            routing_key=f"request.{group_key}.score.{plugin}.internal.internal",
            timeout=timeout,
        )
        self.budget  = budget
        self.used    = 0
        self.records = []
        self.runtime_limit = runtime_limit
        self.start_time    = None

    def close(self):
        self.rpc.close()

    @cache
    def _evaluate(self, smile):
        res = self.rpc.score_batch([{"item_data":{"item":smile}}])[0]
        if isinstance(res, Exception) or res is None or not res["valid"]:
            score = None
        else:
            score = res["score"]
        self.used += 1
        self.records.append(
            {"ts": str(datetime.now()), "item": smile, "score": score}
        )
        score = -1e6 if score is None else score 
        return score 
    
    def _check_limits(self):
        if self.used >= self.budget:
            raise RuntimeError("call-budget exhausted")

        if self.runtime_limit is not None:
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            elif now - self.start_time > self.runtime_limit:
                raise RuntimeError("runtime limit exceeded")

    def __call__(self, smile):
        self._check_limits()
        score = self._evaluate(smile)
        if self.used%500 == 0:
            print(f"Inference {self.used} / {self.budget}")
        return score 

# ------------------------------------------------------------------- run helper
def run_synthemol(run_type: str, run_name: str, params: dict):
    out_dir = BLOB / "internal" / "benchmarks" / "synthemol" / run_type / run_name
    if out_dir.exists():
        print("run already finished")
        return 
    
    device = 'gpu' if params["score_name"]=='erbb1_mlp' else 'cpu'
    scorer = RPCScore(
        budget  = params["inference_budget"],
        plugin  = params["score_name"],
        timeout = params["score_timeout"],
        device  = device,
        runtime_limit = params["runtime_limit"]
    )

    g = Generator(
        building_block_smiles_to_id=bb_smiles_to_id,
        max_reactions=params["max_reactions"],
        scoring_fn=scorer,
        explore_weight=params["explore_weight"],
        num_expand_nodes=params["num_expand_nodes"],
        optimization="maximize",
        reactions=rxn_set,
        rng_seed=params["rng_seed"],
        no_building_block_diversity=params["no_diversity"],
        store_nodes=False,
        verbose=False,
    )

    t0 = time.time()
    try:
        # run rolllouts until inference budget is exhausted
        _ = g.generate(n_rollout=1_000_000)
    except RuntimeError:   # budget exhausted
        pass
    runtime = time.time() - t0
    scorer.close()

    # --------------------------- persist -----------------------------------
    out_dir = BLOB / "internal" / "benchmarks" / "synthemol" / run_type / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(scorer.records)
    result_df["is_bb"] = result_df.item.isin(enamine_df.item)
    result_df.to_csv(out_dir / "score_log.csv", index=False)
    with open(out_dir / "params.json", 'w') as f:
        json.dump(params, f)

    summary = {
        "run_name": run_name,
        "dataset": "enamine",
        "score": params["score_name"],
        "inference_budget": params["inference_budget"],
        "n_inference": scorer.used,
        "n_results": result_df.shape[0],
        "runtime": runtime
    }
    summ_path = out_dir.parent / "summary.csv"
    pd.DataFrame([summary]).to_csv(
        summ_path, mode="a", header=not summ_path.exists(), index=False
    )

    print(f"[✓] finished {run_name} - results in {out_dir}")
