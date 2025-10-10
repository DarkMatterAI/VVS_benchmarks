"""
scores.openeye
~~~~~~~~~~~~~~~~~~~
OpenEye docking (2zdt, 6lud) and ROCS overlay (2chw ligand).

Code adapted from https://github.com/PatWalters/TS/blob/210d83fa0ce58f7260b163aae1e97c8ef10acff0/evaluators.py
"""

import os, gc
from pathlib import Path
from functools import lru_cache
from typing import List

from openeye import oechem, oeomega, oedocking, oeshape
from .utils import log, TimeoutException

# ────────────────────────────────────────────────────────────────────────────
# paths
BASE_DIR = Path("/code/processed/openeye")

FILES = {
    # docking grids (.oedu)
    "dock_2zdt": BASE_DIR / "docking" / "2zdt_receptor.oedu",
    "dock_6lud": BASE_DIR / "docking" / "6lud.oedu",
    # overlay reference ligand (.sdf)
    "rocs_2chw": BASE_DIR / "rocs" / "2chw_lig.sdf",
}

# ────────────────────────────────────────────────────────────────────────────
# licence check
@lru_cache(maxsize=1)
def _has_license() -> bool:
    lic_path = Path(os.getenv("OE_LICENSE", "/run/secrets/oe_license_file"))
    return lic_path.exists() and lic_path.read_text().strip() != ""

# ────────────────────────────────────────────────────────────────────────────
def _gen_confs(mol, max_confs=10) -> bool:
    om = oeomega.OEOmega()
    om.SetRMSThreshold(0.5) # 0.5 rms thresh
    om.SetStrictStereo(False) # False for strict stereo 
    om.SetMaxConfs(max_confs)
    err_lvl = oechem.OEThrow.GetLevel()
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)  # silence warnings
    ok = om(mol)
    oechem.OEThrow.SetLevel(err_lvl)
    return ok

# ────────────────────────────────────────────────────────────────────────────
# 1) docking helpers
@lru_cache(maxsize=None)
def _read_grid(path: Path):
    du = oechem.OEDesignUnit()
    if not oechem.OEReadDesignUnit(str(path), du):
        raise FileNotFoundError(path)
    dk = oedocking.OEDock()
    dk.Initialize(du)
    return dk

def _dock_one(smiles: str, dock: oedocking.OEDock, max_conf=10):
    mol = oechem.OEMol()
    oechem.OEParseSmiles(mol, smiles)
    if not _gen_confs(mol, max_conf):
        return None
    docked = oechem.OEGraphMol()
    if dock.DockMultiConformerMolecule(docked, mol) != oedocking.OEDockingReturnCode_Success:
        return None
    dock_opts = oedocking.OEDockOptions()
    score_method = dock_opts.GetScoreMethod()
    tag = oedocking.OEDockMethodGetName(score_method)
    oedocking.OESetSDScore(docked, dock, tag)
    return -float(oechem.OEGetSDData(docked, tag))      # higher = better

# ────────────────────────────────────────────────────────────────────────────
# 2) ROCS overlay helpers
@lru_cache(maxsize=None)
def _read_ref_lig(path: Path):
    ifs = oechem.oemolistream(str(path))
    ref = oechem.OEMol()
    oechem.OEReadMolecule(ifs, ref)
    return ref

def _overlay_score(smiles: str, ref_mol, max_conf=10):
    fit = oechem.OEMol()
    oechem.OEParseSmiles(fit, smiles)
    if not _gen_confs(fit, max_conf):
        return None

    prep = oeshape.OEOverlapPrep()
    prep.Prep(ref_mol)
    overlay = oeshape.OEMultiRefOverlay()
    overlay.SetupRef(ref_mol)
    prep.Prep(fit)

    score = oeshape.OEBestOverlayScore()
    overlay.BestOverlay(score, fit, oeshape.OEHighestTanimoto())
    return score.GetTanimotoCombo()

# ────────────────────────────────────────────────────────────────────────────
# generic batch wrappers
def _batch(
    request: List[dict],
    func,                                     # _dock_one or _overlay_score
    resource,                                 # dock grid or ref ligand
):
    out = []
    for rec in request:
        try:
            smi = rec["item_data"]["item"] if "item_data" in rec else rec["item"]
            s = func(smi, resource)
            out.append({"valid": s is not None, "score": float(s) if s is not None else None})
        except TimeoutException as e:
            raise e 
        except Exception as e:
            log(f"[red]Openeye error {e}, {str(e)}, {type(e)}")
            out.append({"valid": False, "score": None})
    gc.collect()
    return out

def _guard(request, job):
    if not _has_license():
        msg = {"valid": False, "score": None}
        return [msg] * len(request)
    return job()

# ────────────────────────────────────────────────────────────────────────────
# 4) public API functions ----------------------------------------------------
def dock_2zdt(request):
    name = "dock_2zdt"
    return _guard(
        request,
        lambda: _batch(request, _dock_one, _read_grid(FILES[name]))
    )

def dock_6lud(request):
    name = "dock_6lud"
    return _guard(
        request,
        lambda: _batch(request, _dock_one, _read_grid(FILES[name]))
    )

def rocs_2chw(request):
    return _guard(
        request,
        lambda: _batch(request, _overlay_score, _read_ref_lig(FILES["rocs_2chw"]))
    )

