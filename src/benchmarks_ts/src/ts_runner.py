import sys
sys.path.append('/code/TS/')

from typing import List, Optional 
import time, math, random, pickle, glob, os, json
from functools import cache 
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from thompson_sampling import ThompsonSampler
from reagent import Reagent
from disallow_tracker import DisallowTracker

from .score_rpc import BatchScorer

# ──────────────────────────────────────────────────────────────  data loading
BLOB  = Path("/code/blob_store").resolve()
ENA   = BLOB / "internal" / "processed" / "enamine"
CSV   = ENA / "data.csv"
RXN   = next(ENA.glob("*_id_to_reaction.pkl"))

enamine_df = pd.read_csv(CSV)

with open(RXN, "rb") as f:
    id_to_rxn = pickle.load(f)
    rxn_smarts = list(set(id_to_rxn.values()))

# pre-build reactant slots for substructure matching
slots = [[], []]
for sm in rxn_smarts:
    rxn = AllChem.ReactionFromSmarts(sm); rxn.Initialize()
    for i, r in enumerate(rxn.GetReactants()):
        slots[i].append(r)

# build reagent lists ---------------------------------------------------------
smiles_lists = [[], []]
for smi, rid in zip(enamine_df.item, enamine_df.external_id):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    for i, react_patterns in enumerate(slots):
        if any(mol.HasSubstructMatch(p) for p in react_patterns):
            smiles_lists[i].append((rid, smi))
            # reagent_lists[i].append(Reagent(rid, smi))

# ───────────────────────────────────────────────────  RPC score interface ----
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

    def evaluate(self, mol):
        self._check_limits()
        smile = Chem.MolToSmiles(Chem.RemoveHs(mol))
        score = self._evaluate(smile)
        if self.used%500 == 0:
            print(f"Inference {self.used} / {self.budget}")
        score = -1*np.inf if score is None else score 
        return score

# ───────────────────────────────────────────────────────── Thompson wrapper --
class ReactionWrapper():
    # wrapper for rdkit reaction because we need to run 
    # `Chem.RemoveHs` before TS processes the result
    def __init__(self, smarts):
        self.smarts = smarts
        self.reaction = AllChem.ReactionFromSmarts(smarts)
        self.reaction.Initialize()
        self.reactants = self.reaction.GetReactants()
        
    def RunReactants(self, reagents):
        prod = self.reaction.RunReactants(reagents)
        prod = [[Chem.RemoveHs(i) for i in out] for out in prod]
        return prod
    
    def check_reactants(self, mols):
        for (mol, p) in zip(mols, self.reactants):
            if not mol.HasSubstructMatch(p):
                return False
        return True 
    
class MultiReactionTS(ThompsonSampler):
    """
    Thompson-sampler that supports an arbitrary list of two-component
    reactions and a global inference budget.
    """
    def __init__(
        self,
        reaction_smarts: List[str],
        mode="maximize", 
        log_filename: Optional[str] = None
    ):
        super().__init__(mode=mode, log_filename=log_filename)
        self.reactions = [ReactionWrapper(s) for s in reaction_smarts]

    def read_reagents(self, reagents_list):
        self.reagent_lists = reagents_list
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def evaluate(self, choice_list):
        # iterate reactions in a random order until a match is found
        rxn_idxs = list(range(len(self.reactions)))
        random.shuffle(rxn_idxs)

        mols = []
        for idx, choice in enumerate(choice_list):
            mols.append(self.reagent_lists[idx][choice].mol)
        
        for rxn_idx in rxn_idxs:
            self.reaction = self.reactions[rxn_idx]        # plug into parent helper
            self.reaction_idx = rxn_idx

            if not self.reaction.check_reactants(mols):
                continue 

            results = super().evaluate(choice_list)
            if results[0] != 'FAIL':
                return results
            
        return "FAIL", 'FAIL', np.nan

    def warm_up(self, n_warmup):
        """Randomly sample pairs until n_warmup molecules have been evaluated."""
        warmup_scores, warmup_results = [], []
        rng = np.random.default_rng()

        while self.evaluator.used < n_warmup:
            choice = self._disallow_tracker.sample()
            smi, name, score = self.evaluate(choice)
            if np.isfinite(score):
                warmup_scores.append(score)
                warmup_results.append([score, smi, name])
            self._disallow_tracker.update(choice)

        # -------- prior initialisation (changed from the paper) ------------
        μ0, σ0 = float(np.mean(warmup_scores)), float(np.std(warmup_scores))
        self._warmup_std = σ0
        for reag_list in self.reagent_lists:
            for r in reag_list:
                # keep every reagent - even if it never received a score
                if r.initial_scores:
                    r.init_given_prior(μ0, σ0)
                else:                      # untouched in warm-up ⇒ no scores
                    r.current_phase = "search"
                    r.current_mean  = μ0
                    r.current_std   = σ0
                    r.known_var     = σ0**2
        self.logger.info(f"Warm-up done: {len(warmup_scores)} scores, "
                         f"prior μ={μ0:.3f}, σ={σ0:.3f}")
        return warmup_results

# ────────────────────────────────────────── driver callable by run_benchmark
def run_ts(run_type, run_name, params):
    rng_seed = params["rng_seed"]
    random.seed(rng_seed); np.random.seed(rng_seed)
    reagent_lists = [[Reagent(rid, smi) for rid, smi in i ] for i in smiles_lists]
    score_name = params["score_name"]
    device = "gpu" if score_name == "erbb1_mlp" else "cpu"

    scorer = RPCScore(params["inference_budget"], 
                      score_name, 
                      params["score_timeout"],
                      device,
                      runtime_limit=params["runtime_limit"])
    sampler = MultiReactionTS(rxn_smarts)
    sampler.evaluator = scorer
    sampler._top_func = lambda xs: max(xs) if xs else [0, None, None]
    sampler.read_reagents(reagent_lists)

    warm = int(params["inference_budget"] * params["warmup_percent"])
    print(f"{run_name} - warmup")
    t0 = time.time()
    sampler.warm_up(warm)
    print(f"{run_name} - search")
    try:
        sampler.search(100_000_000)      # run search until inference budget
    except RuntimeError:
        pass
    runtime = time.time() - t0
    scorer.close()

    # --------------------------- persist -----------------------------------
    out_dir = BLOB / "internal" / "benchmarks" / "ts" / run_type / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(scorer.records)
    result_df = result_df[~result_df.item.isin(enamine_df.item)] # remove building block scores
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
        "runtime": runtime,
    }
    summ_path = out_dir.parent / "summary.csv"
    pd.DataFrame([summary]).to_csv(
        summ_path, mode="a", header=not summ_path.exists(), index=False
    )

    print(f"[✓] finished {run_name} - results in {out_dir}")