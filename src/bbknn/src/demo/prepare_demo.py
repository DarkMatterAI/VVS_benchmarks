"""
prepare_demo.py
Build the artifacts needed to run the BBKNN demo:
  * <BLOB_STORE>/demo_artifacts/bb_embeddings.pt   (256-dim ModuleDict state dict)
  * <BLOB_STORE>/internal/processed/enamine/enamine_id_to_reaction.pkl

Self-contained: does NOT import the `bbknn` package, so it can run before the
reaction pickle exists (which `bbknn`'s import chain requires).
"""
from __future__ import annotations
import os, json, pickle as pkl
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

# --- paths (mirror constants.py, derived from BLOB_STORE) ---
BLOB         = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
REACTION_PKL = BLOB / "internal" / "processed" / "enamine" / "enamine_id_to_reaction.pkl"
BB_EMB_PATH  = BLOB / "demo_artifacts" / "bb_embeddings.pt"

DEMO_DIR     = Path(__file__).parent
DEMO_BB_CSV  = DEMO_DIR / "demo_bbs.csv"
DEMO_RXN_JS  = DEMO_DIR / "demo_reactions.json"

EMB_MODEL    = "entropy/roberta_zinc_480m"
DECOMP_MODEL = "entropy/roberta_zinc_enamine_decomposer"
SIZE         = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _mean_pool(model, tok, coll, smiles, bs=512):
    outs = []
    for i in range(0, len(smiles), bs):
        batch = smiles[i:i + bs]
        t = coll(tok(batch, truncation=True, max_length=256))
        t = {k: v.to(DEVICE) for k, v in t.items()}
        last = model(**t, output_hidden_states=True).hidden_states[-1]
        mask = t["attention_mask"]
        emb = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        outs.append(emb.cpu())
    return torch.cat(outs)


def build_bb_embeddings():
    df = pd.read_csv(DEMO_BB_CSV)
    assert {"external_id", "item"}.issubset(df.columns), \
        "demo_bbs.csv must contain 'external_id' and 'item' columns"

    tok  = AutoTokenizer.from_pretrained(EMB_MODEL)
    coll = DataCollatorWithPadding(tok, return_tensors="pt")
    embed_model = AutoModel.from_pretrained(EMB_MODEL, add_pooling_layer=False).to(DEVICE).eval()
    decomposer  = AutoModel.from_pretrained(DECOMP_MODEL, trust_remote_code=True).to(DEVICE).eval()

    base = _mean_pool(embed_model, tok, coll, df["item"].tolist())          # [N, 768]
    comp = decomposer.compress(base.to(DEVICE), [SIZE])[SIZE].cpu().float()  # [N, 256]

    # Match the format BBKNN expects: state_dict of nn.ModuleDict({str(size): Embedding})
    module = nn.ModuleDict({str(SIZE): nn.Embedding.from_pretrained(comp, freeze=True)})
    BB_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), BB_EMB_PATH)
    print(f"\u2713 wrote {BB_EMB_PATH}  (shape {tuple(comp.shape)})")


def build_reactions():
    with open(DEMO_RXN_JS) as f:
        rxns = json.load(f)
    rxns = {int(k): v for k, v in rxns.items()}     # {id: SMARTS}
    REACTION_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(REACTION_PKL, "wb") as f:
        pkl.dump(rxns, f)
    print(f"\u2713 wrote {REACTION_PKL}  ({len(rxns)} reactions)")


if __name__ == "__main__":
    build_reactions()
    build_bb_embeddings()