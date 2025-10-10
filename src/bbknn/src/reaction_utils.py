"""
Utility helpers for running two-building-block reactions *en masse* and
for fast Tanimoto similarity calculation.
"""
from __future__ import annotations
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
import pickle as pkl
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity

from .constants import REACTION_PKL

# ──────────────────────────────────────────────────────────── load reactions
with open(REACTION_PKL, "rb") as f:
    _id_to_smarts: Dict[int, str] = pkl.load(f)

REACTIONS = {}
for rid, smarts in _id_to_smarts.items():
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    REACTIONS[rid] = {
        "reaction":   rxn,
        "reactants":  list(rxn.GetReactants()),
    }

# ───────────────────────────────────────────────────── tanimoto helper
def compute_fps(smiles, radius=2, fp_size=1024, num_threads=32):
    fpg = GetMorganGenerator(radius=radius, fpSize=fp_size)
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    fps = fpg.GetFingerprints(mols, numThreads=num_threads)
    return fps 

def tanimoto_similarity(idxs, fps):
    """Fast single-pair Tanimoto - used inside map-loops."""
    q_idx, r_idx = idxs
    return TanimotoSimilarity(fps[q_idx], fps[r_idx])


# ───────────────────────────────────────────────────── reaction helpers
def _remove_stereo(smi: str) -> str:
    """Strip all stereo from a SMILES (RDKit)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def _run_reaction(m1, m2, rxn_dict):
    rxn, (r1, r2) = rxn_dict["reaction"], rxn_dict["reactants"]
    prods = set()
    for a, b in ((m1, m2), (m2, m1)):
        if a.HasSubstructMatch(r1) and b.HasSubstructMatch(r2):
            for plist in rxn.RunReactants((a, b)):
                for p in plist:
                    prods.add(Chem.MolToSmiles(Chem.RemoveHs(p)))
    return prods

def _react_pair(pair_info: dict) -> List[str]:
    m1 = Chem.AddHs(Chem.MolFromSmiles(pair_info["bb1_item"]))
    m2 = Chem.AddHs(Chem.MolFromSmiles(pair_info["bb2_item"]))
    out = []
    for rxn in REACTIONS.values():
        out.extend(_run_reaction(m1, m2, rxn))
    return list(set(out))

def _flatten_pair(pair_info: dict) -> List[dict]:
    """Return one dict per product, keeping all metadata."""
    res = _react_pair(pair_info)
    out = []
    for p in res:
        d = pair_info.copy()
        d["result"] = p
        out.append(d)
    return out

def parallel_react(pairs: List[dict], num_proc: int) -> List[dict]:
    """
    Enumerate two-BB reactions in parallel *with a live progress-bar*.

    Parameters
    ----------
    pairs : list[dict]
        Output of ``BBKNN.build_reaction_pairs`` - one dict per BB-pair.
    num_proc : int
        Number of worker processes (falls back to serial if ≤ 1).

    Returns
    -------
    list[dict]  - flattened records, one per enumerated product.
    """
    if num_proc <= 1:                                # ── serial fallback
        nested = [_flatten_pair(p) for p in tqdm(pairs, desc="Reacting")]
    else:                                            # ── multi-proc
        with Pool(processes=num_proc) as pool:
            nested = list(
                tqdm(
                    pool.imap_unordered(_flatten_pair, pairs),
                    total=len(pairs),
                    desc="Reacting",
                )
            )
    # flatten outer list
    return [item for sub in nested for item in sub]
