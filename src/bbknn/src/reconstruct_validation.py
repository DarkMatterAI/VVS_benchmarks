"""
Reconstructs the Enamine validation set (first 1 M rows of the HF dataset)
into full product SMILES using the stored BB-IDs & reaction templates.
Writes a CSV once - idempotent on re-runs.
"""
from __future__ import annotations
import multiprocessing as mp, pickle as pkl
from collections import defaultdict
from pathlib import Path

import pandas as pd
import datasets as ds
from rdkit import Chem
from rdkit.Chem import AllChem
from rich.console import Console

console = Console()

from .constants import BLOB                # already imported elsewhere

# ---------- paths ----------
DATA_DIR   = BLOB / "internal" / "training_datasets" / "enamine_assembled"
HF_DS      = DATA_DIR / "enamine_assembled.hf"
ENAMINE_CSV = BLOB / "internal" / "processed" / "enamine" / "data.csv"
RXN_PKL     = BLOB / "internal" / "processed" / "enamine" / "enamine_id_to_reaction.pkl"
OUT_CSV     = BLOB / "internal" / "bbknn" / "bbknn_eval" / "validation_dataset.csv"

VALID_SZ   = 1_000_000
CPU        = max(1, mp.cpu_count() - 2)

# -------------------------------------------------------------------------
def _prepare_assets():
    console.log("[cyan]Loading HF dataset slice & lookup tables")
    full = ds.load_from_disk(str(HF_DS))
    valid = full.select(range(VALID_SZ)).remove_columns("embedding").to_pandas()

    ena = pd.read_csv(ENAMINE_CSV, usecols=["item"])
    bb1 = ena.iloc[valid.bb1_id].reset_index(drop=True).rename(columns={"item": "bb1"})
    bb2 = ena.iloc[valid.bb2_id].reset_index(drop=True).rename(columns={"item": "bb2"})
    pairs = pd.concat([bb1, bb2], axis=1)
    pairs = pairs.drop_duplicates().reset_index(drop=True)

    with open(RXN_PKL, "rb") as f:
        id_to_smarts = pkl.load(f)

    reactions = {}
    for rid, smarts in id_to_smarts.items():
        rxn = AllChem.ReactionFromSmarts(smarts); rxn.Initialize()
        reactions[rid] = {"reaction": rxn,
                          "reactants": list(rxn.GetReactants())}
    return pairs.to_dict("records"), reactions

def _run_reaction(m1, m2, rxn_dict):
    rxn, (r1, r2) = rxn_dict["reaction"], rxn_dict["reactants"]
    products = set()
    for a, b in ((m1, m2), (m2, m1)):            # both orientations
        if a.HasSubstructMatch(r1) and b.HasSubstructMatch(r2):
            for plist in rxn.RunReactants((a, b)):
                for p in plist:
                    products.add(Chem.MolToSmiles(Chem.RemoveHs(p)))
    return products

def _react_pair(pair, reactions):
    mol1 = Chem.AddHs(Chem.MolFromSmiles(pair["bb1"]))
    mol2 = Chem.AddHs(Chem.MolFromSmiles(pair["bb2"]))

    out = []
    for rxn in reactions.values():
        for prod in _run_reaction(mol1, mol2, rxn):
            out.append({"product": prod,
                        "bb1": pair["bb1"],
                        "bb2": pair["bb2"]})
    return pd.DataFrame(out)

# -------------------------------------------------------------------------
def build_validation_csv(force: bool = False) -> Path:
    if OUT_CSV.exists() and not force:
        console.log("[cyan]Validation CSV already present - skipping rebuild")
        return OUT_CSV

    pairs, reactions = _prepare_assets()
    console.log(f"[cyan]Running reactions on {len(pairs):,} BB-pairs …")

    with mp.Pool(processes=CPU) as pool:
        dfs = pool.starmap(_react_pair, [(p, reactions) for p in pairs])

    pd.concat(dfs, ignore_index=True).to_csv(OUT_CSV, index=False)
    console.log(f"[green]✓ wrote validation dataset → {OUT_CSV}")
    return OUT_CSV
