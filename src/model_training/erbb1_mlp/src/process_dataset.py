"""
Create an HF Dataset for ErbB1 pIC50 regression:

* Loads processed CSV   →   blob_store/internal/processed/chembl_erbb1_ic50/data.csv
* Filters / normalises  →   adds `pIC50_norm`
* Embeds with           →   entropy/roberta_zinc_480m  (mean-pooling)
* Saves dataset to      →   blob_store/internal/training_datasets/erbb1_mlp.hf
* Persists mean/std to  →   erbb1_mlp_stats.json  (alongside dataset)
"""
import json, math, os
from pathlib import Path

import datasets as ds
import pandas as pd
import torch
from rich.console import Console
from transformers import (
    AutoTokenizer,
    RobertaModel,
    DataCollatorWithPadding,
)

assert torch.cuda.is_available()
console = Console()
device  = "cuda" if torch.cuda.is_available() else "cpu"

# ─── paths ───────────────────────────────────────────────────────
BLOB_STORE   = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
RAW_CSV      = BLOB_STORE / "internal" / "processed" / "chembl_erbb1_ic50" / "data.csv"
OUT_DIR      = BLOB_STORE / "internal" / "training_datasets" / "erbb1_mlp" / "erbb1_mlp.hf"
STATS_PATH   = OUT_DIR.parent / "erbb1_mlp_stats.json"

EMBED_MODEL  = "entropy/roberta_zinc_480m"
BATCH_SIZE   = 512


# ─── helpers ─────────────────────────────────────────────────────
def _compute_mean_std(series):
    mean = series.mean()
    std  = series.std()
    return mean, std


def _embed_fn(batch, tokenizer, collator, model):
    toks = tokenizer(batch["item"], truncation=True, max_length=256)
    inputs = collator(toks)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]                          # [B,L,768]
        mask = inputs["attention_mask"]
        embed = (last_hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
    return {"embedding": embed.cpu().tolist()}


# ─── main ────────────────────────────────────────────────────────
def main():
    console.rule("[bold]ErbB1 dataset embedding")

    if OUT_DIR.exists():
        console.log("[green]✓ embedded dataset already exists - skipping")
        return

    if not RAW_CSV.exists():
        console.log(f"[red]✗ missing processed CSV: {RAW_CSV}")
        return

    # 1) Load CSV → Dataset ---------------------------------------------------
    df = pd.read_csv(RAW_CSV)  # columns: external_id, item, IC50, pIC50
    mean, std = _compute_mean_std(df["pIC50"])
    df["pIC50_norm"] = (df["pIC50"] - mean) / std
    console.log(f"Dataset size: {len(df):,}    μ={mean:.3f}  σ={std:.3f}")

    ds_raw = ds.Dataset.from_pandas(df, preserve_index=False)

    # 2) Embed ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = RobertaModel.from_pretrained(EMBED_MODEL).to(device).eval()
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    embed_ds = ds_raw.map(
        lambda batch: _embed_fn(batch, tokenizer, collator, model),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["item"],     # keep external_id, IC50, pIC50, pIC50_norm
        desc="embedding SMILES",
    )

    # 3) Save ----------------------------------------------------------------
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    embed_ds.save_to_disk(str(OUT_DIR))
    console.log(f"[green]✓ dataset saved → {OUT_DIR}")

    with open(STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    console.log(f"[green]✓ stats saved  → {STATS_PATH.name}")


if __name__ == "__main__":
    main()
