"""
vvs_local.lr_sweep.run
────────────────────────────────────────────────────────────────────────────
One benchmark script that can drive either

    • BB-KNN  (reaction enumeration)      - engine: bbknn
    • plain K-NN (pre-built HNSW index)   - engine: knn

Parameters come from lr_sweep.yaml; results are written to a per-engine /
per-scorer sub-folder inside

    blob_store/internal/data_analysis/vvs_local/lr_sweep/
"""

from __future__ import annotations
import argparse, yaml, pickle, time
from pathlib import Path
from typing  import Dict, Any, List, Tuple, Optional

import duckdb, torch, torch.nn.functional as F
import numpy  as np
import pandas as pd
from rich.console import Console
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from usearch.index import Index   

from vvs_local.constants import (
    BLOB, DEVICE, D4_DB_PATH, D4_TBL_NAME,
    ZNC_DB_PATH, ZNC_TBL_NAME, INDEX_PATH,
    EMB_MODEL_NM, DECOMP_MODEL_NM,
    BB_EMB_PTH, BB_CSV_PTH, REACTION_PKL,
)
from vvs_local.reaction_assembly import ReactionAssembly
from vvs_local.bbknn             import BBKNN
from vvs_local.knn               import KNN
from vvs_local.model_store       import ModelStore
from vvs_local.score_rpc         import RPCScore
from vvs_local.vvs               import VVS

console = Console()


# ╭───────────────────────── YAML helper ─────────────────────────╮
def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ╭──────────────── reaction SMARTS helper (BB-KNN only) ─────────╮
def _smarts_list() -> List[str]:
    with open(REACTION_PKL, "rb") as fh:
        id2s = pickle.load(fh)
    return list({s for s in id2s.values()})


# ╭──────────────── DuckDB sampler ───────────────────────────────╮
def _sample_smiles(db: Path, tbl: str, n: int) -> List[str]:
    sql = f"SELECT item FROM {tbl} USING SAMPLE reservoir({n});"
    return (duckdb.connect(str(db), read_only=True)
                  .execute(sql)
                  .fetchnumpy()["item"].tolist())


# ╭──────────────────── embed+compress helpers ───────────────────╮
def _row_cos(mat: torch.Tensor, vecs: torch.Tensor) -> np.ndarray:     # vecs row-aligned
    return F.cosine_similarity(mat, vecs, dim=1).cpu().numpy()


# ╭────────────────────── engine factories ───────────────────────╮
def _make_bbknn(cfg: dict,
                assembly: ReactionAssembly,
                store: ModelStore) -> BBKNN:

    in_sz  = cfg["embedding"]["input_size"]
    out_sz = cfg["embedding"]["output_size"]

    # Load only the light artefacts here
    bb_state   = torch.load(BB_EMB_PTH, map_location="cpu")
    bb_table   = bb_state[f"{out_sz}.weight"].float()
    bb_smiles  = pd.read_csv(BB_CSV_PTH)["item"].tolist()

    # Pass references to the store’s models / tok / coll
    return BBKNN(tok       = store.tok,
                 collator  = store.coll,
                 encoder   = store.enc,
                 decomposer= store.dec,
                 bb_table  = bb_table,
                 bb_smiles = bb_smiles,
                 assembly  = assembly,
                 input_size= in_sz,
                 output_size=out_sz,
                 device    = store.device)


def _make_knn(cfg: dict) -> KNN:
    size       = cfg["embedding"]["size"]
    idx_path   = INDEX_PATH / f"{size}" / "index.usearch"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    index = Index.restore(str(idx_path))
    console.log(f"[cyan]HNSW index {size}-d → {index.size} vectors loaded")

    return KNN(index=index,
               db_path=str(ZNC_DB_PATH),
               db_tbl=ZNC_TBL_NAME)


# ╭─────────────────── batch-level workhorse ─────────────────────╮
def _run_batches(cfg: dict,
                 engine,                     # BBKNN | KNN
                 store: ModelStore,
                 scorer_cfg: dict,
                 lrs: torch.Tensor,
                 out_dir: Path,
                 rank_caps: List[int],
                 use_bb_ranks: bool) -> None:

    scorer = RPCScore(**scorer_cfg)
    vvs    = VVS(engine, scorer, update_type="standard")

    bs        = cfg["batching"]["batch_size"]
    n_batches = cfg["batching"]["num_batches"]
    k_nn      = cfg["knn"]["k_nn"]

    prod_parts: List[pd.DataFrame] = []
    oid_parts : List[torch.Tensor] = []
    n_orig = n_query = 0

    for b in range(n_batches):
        console.log(f"[cyan]{scorer_cfg['plugin']} - batch {b+1}/{n_batches}")

        # ---------- query sampling & gradient ----------------------------
        smiles   = _sample_smiles(D4_DB_PATH, D4_TBL_NAME, bs)
        q0 = store.encode_compress(smiles).cpu()

        base_idx = torch.arange(q0.size(0))
        _, (q, _, g) = vvs.search_iteration(
            q0, base_idx, lrs, k_nn,
            prev_grads=None, skip_score=False
        )

        gq, orig_idx = vvs._expand_queries(q, base_idx, lrs, g, False)

        prod_df, prod_emb = engine.run(gq, k_nn=k_nn)

        # ---------- enrich & global re-index ----------------------------
        lr_vec = lrs[None].repeat_interleave(q0.size(0), 0).reshape(-1).cpu().numpy()
        prod_df["lr"]       = lr_vec[prod_df.query_idx]
        prod_df["orig_idx"] = orig_idx[prod_df.query_idx].cpu().numpy()

        prod_df["cos_to_query"] = _row_cos(prod_emb, q0[prod_df.orig_idx])
        prod_df["cos_to_grad"]  = _row_cos(prod_emb, gq[prod_df.query_idx])

        # continuous indices
        o_u = prod_df.orig_idx.nunique()
        q_u = prod_df.query_idx.nunique()
        prod_df["orig_idx"]  += n_orig
        prod_df["query_idx"] += n_query
        orig_idx += n_orig
        n_orig   += o_u
        n_query  += q_u

        prod_parts.append(prod_df)
        oid_parts.append(orig_idx)

    prod_df  = pd.concat(prod_parts, ignore_index=True)
    orig_idx = torch.cat(oid_parts)

    # ---------- STATISTICS ----------------------------------------------
    if use_bb_ranks:
        mask_fn = lambda k: (prod_df.bb1_rank < k) & (prod_df.bb2_rank < k)
    else:
        mask_fn = lambda k: (prod_df["rank"] < k)

    rank_mask = {k: mask_fn(k) for k in rank_caps}

    # baseline result sets (lr==0) per-orig_idx
    baseline = {
        k: {oid: set(g.result)
            for oid, g in prod_df[(prod_df.lr == 0) & m].groupby("orig_idx")}
        for k, m in rank_mask.items()
    }

    def _recall(level_idx: pd.Index,
                sets: pd.Series,
                k_val: int,
                is_qidx: bool):
        oid = (orig_idx[level_idx].numpy() if is_qidx else level_idx.values)
        base_sets = [baseline[k_val].get(i, set()) for i in oid]
        return np.array([len(s & b)/max(len(b), 1) for s, b in zip(sets, base_sets)])

    stats = []
    for k, m in rank_mask.items():
        df_k = prod_df[m]

        gq_sets = df_k.groupby(["lr", "query_idx"])["result"].agg(set)
        gq = (df_k.groupby(["lr","query_idx"])
                   .agg(cos_q_mean=("cos_to_query","mean"),
                        cos_q_std =("cos_to_query","std"),
                        cos_g_mean=("cos_to_grad" ,"mean"),
                        cos_g_std =("cos_to_grad" ,"std"),
                        n_results=("result","nunique"))
                   .assign(overlap_frac=_recall(
                               gq_sets.index.get_level_values(1), gq_sets, k, True),
                           k=k, level="grad_query")
                   .reset_index())

        orig_sets = df_k.groupby(["lr", "orig_idx"])["result"].agg(set)
        oq = (df_k.groupby(["lr","orig_idx"])
                   .agg(cos_q_mean=("cos_to_query","mean"),
                        cos_q_std =("cos_to_query","std"),
                        cos_g_mean=("cos_to_grad" ,"mean"),
                        cos_g_std =("cos_to_grad" ,"std"),
                        n_results=("result","nunique"))
                   .assign(overlap_frac=_recall(
                               orig_sets.index.get_level_values(1), orig_sets, k, False),
                           k=k, level="orig_query")
                   .reset_index())

        overall = (df_k.groupby("lr")
                         .agg(cos_q_mean=("cos_to_query","mean"),
                              cos_q_std =("cos_to_query","std"),
                              cos_g_mean=("cos_to_grad" ,"mean"),
                              cos_g_std =("cos_to_grad" ,"std"),
                              n_results=("result","nunique"))
                         .assign(overlap_frac=np.nan,
                                 k=k, level="overall")
                         .reset_index())

        stats.extend([gq, oq, overall])

    stats_df = pd.concat(stats, ignore_index=True)

    # ---------- persist ---------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    prod_df .to_parquet(out_dir / "prod_df.parquet",  index=False)
    stats_df.to_parquet(out_dir / "stats_df.parquet", index=False)
    np.save(out_dir / "orig_idx.npy", orig_idx.cpu().numpy())
    with open(out_dir / "params.yaml", "w") as fh:
        yaml.safe_dump({"config": cfg, "scorer": scorer_cfg}, fh)

    console.log(f"[green]✓ saved → {out_dir}")
    scorer.close()


# ╭─────────────────────────────────── main() ─────────────────────────╮
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="lr_sweep.yaml")
    args = ap.parse_args()
    cfg  = _load_cfg(Path(f"/code/{args.cfg}"))

    root = (BLOB / "internal" / "data_analysis" / "vvs_local" / "lr_sweep")
    root.mkdir(parents=True, exist_ok=True)

    # ───── load models ----------------------------------------------------
    comp_size = cfg["embedding"].get("input_size",
                                    cfg["embedding"].get("size"))   # BB-KNN vs KNN
    store = ModelStore(comp_size)

    # ───── choose engine --------------------------------------------------
    engine_type = cfg.get("engine", "bbknn").lower()
    rank_caps   = cfg.get("rank_caps", [1,3,5,7,10])

    if engine_type == "bbknn":
        assembly = ReactionAssembly(rxn_smarts=_smarts_list(),
                                    **cfg["reaction_pool"])
        engine   = _make_bbknn(cfg, assembly, store)   # <── changed
        use_bb_ranks = True
    elif engine_type == "knn":
        engine   = _make_knn(cfg)
        use_bb_ranks = False
    else:
        raise ValueError(f"Unknown engine '{engine_type}'")

    # shared LR grid
    lrs = torch.linspace(cfg["learning_rates"]["min"],
                         cfg["learning_rates"]["max"],
                         cfg["learning_rates"]["steps"] + 1)
    
    for scorer_cfg in cfg["scorers"]:
        console.rule(f"[bold cyan]{engine_type.upper()}  -  {scorer_cfg['plugin']}")
        out_dir = root / engine_type / scorer_cfg["plugin"]
        _run_batches(cfg, engine, store, scorer_cfg, lrs, out_dir,
                     rank_caps, use_bb_ranks)

    console.rule("[bold green]DONE")


if __name__ == "__main__":
    main()


