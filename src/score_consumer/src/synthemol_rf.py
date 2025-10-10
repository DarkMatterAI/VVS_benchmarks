"""
scores.antibacterial_rf
~~~~~~~~~~~~~~~~~~~~~~~
Random-Forest antibacterial score used in SyntheMol benchmarks.
"""

from pathlib import Path
import pickle, os
from functools import lru_cache
import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from .utils import log 


# ------------------------------------------------------------------------- utils
def sklearn_load(pkl: Path):
    log('SKLearn File Load')
    with pkl.open("rb") as f:
        return pickle.load(f)


def sklearn_predict(model, fps: np.ndarray):
    if isinstance(model, (RandomForestClassifier, MLPClassifier)):
        return model.predict_proba(fps)[:, 1]
    elif isinstance(model, (RandomForestRegressor, MLPRegressor)):
        return model.predict(fps)
    raise TypeError(f"Unsupported model type {type(model)}")


# RDKit 2D-normalized fingerprint generator is expensive to init: cache it
@lru_cache(maxsize=1)
def _fp_gen():
    return rdNormalizedDescriptors.RDKit2DNormalized()


def rdkit_fp(smiles: str) -> np.ndarray:
    vec = _fp_gen().process(smiles)[1:]
    vec = np.nan_to_num(vec, nan=0.0).astype("float32")
    return vec


# ------------------------------------------------------------------- inference
ARTEFACT_DIR = Path(
    os.getenv(
        "ANTIBACTERIAL_RF_PATH",
        "/code/processed/synthemol_rf",
    )
)

@lru_cache(maxsize=1)
def _load_ensemble():
    log(os.listdir(str(ARTEFACT_DIR)))
    pkl_files = sorted(ARTEFACT_DIR.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl models under {ARTEFACT_DIR}")
    return [sklearn_load(p) for p in pkl_files]


def score(request_batch):
    """
    Parameters
    ----------
    request_batch : List[dict]  - each dict has at least key 'item' = SMILES

    Returns
    -------
    List[dict]  - same length, each with keys {valid, score, data}
    """
    smiles = [d["item_data"]["item"] for d in request_batch]
    fps = np.stack([rdkit_fp(smi) for smi in smiles])

    models = _load_ensemble()
    # shape (n_items, n_models)
    preds = np.column_stack([sklearn_predict(m, fps) for m in models])
    mean = preds.mean(axis=1)

    out = [
        {
            "valid": True,
            "score": float(mean[i]),
        }
        for i in range(len(request_batch))
    ]
    return out

