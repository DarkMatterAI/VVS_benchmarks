from __future__ import annotations
import pickle, random 
from pathlib import Path
from typing  import List, Optional

import duckdb, torch
import pandas as pd
from rdkit import Chem 
from usearch.index import Index   

from vvs_local.constants import (
    D4_DB_PATH, D4_TBL_NAME, INDEX_PATH,
    BB_EMB_PTH, BB_CSV_PTH, REACTION_PKL,
    PROC_PTH, console 
)
from vvs_local.reaction_assembly import ReactionAssembly
from vvs_local.bbknn             import BBKNN
from vvs_local.knn               import KNN
from vvs_local.model_store       import ModelStore

INDICES = {}

def _remove_stereo(smi: str) -> str:
    """Strip all stereo from a SMILES (RDKit)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def _sample_smiles(db: Path, tbl: str, n: int) -> List[str]:
    sql = f"SELECT item FROM {tbl} USING SAMPLE reservoir({n});"
    smiles = (duckdb.connect(str(db), read_only=True)
                  .execute(sql)
                  .fetchnumpy()["item"].tolist())
    smiles = [_remove_stereo(i) for i in smiles]
    return smiles 

def _smarts_list() -> List[str]:
    with open(REACTION_PKL, "rb") as fh:
        id2s = pickle.load(fh)
    return list({s for s in id2s.values()})

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
    size     = cfg["embedding"]["size"]
    idx_path = PROC_PTH / "vvs_local" / "d4_index_128" / "index.usearch"

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    
    if idx_path in INDICES:
        index = INDICES[idx_path]
    else:
        index = Index.restore(str(idx_path))
        INDICES[idx_path] = index 
        
    console.log(f"[cyan]HNSW index {size}-d → {index.size} vectors loaded")

    return KNN(index=index,
               db_path=str(D4_DB_PATH),
               db_tbl=D4_TBL_NAME)

def _select_batch(
        prod_df: Optional[pd.DataFrame],
        batch_size: int,
        exploit_percent: float,
        db_path: Path,
        db_table: str,
) -> list[str]:
    """Return a list of SMILES for the next round."""
    if prod_df is None or prod_df.empty or exploit_percent <= 0.0:
        return _sample_smiles(db_path, db_table, batch_size)

    exploit_bs = int(batch_size * exploit_percent)
    explore_bs = batch_size - exploit_bs

    top_hits = (prod_df
                .drop_duplicates("result")
                .nlargest(exploit_bs, "score", keep="all")["result"]
                .tolist())

    top_hits += _sample_smiles(db_path, db_table, explore_bs)
    random.shuffle(top_hits)
    return top_hits
