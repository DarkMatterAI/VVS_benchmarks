"""
benchmarks.build_indices.run
────────────────────────────────────────────────────────────────────
• Embed the Zinc-10 M SMILES set (768-d + all compressed sizes).
• Persist an HF `Dataset` to
      {BLOB}/internal/processed/vvs_local/zinc_10m_embedded.hf
• Build a USearch HNSW index for **every** compressed size and save to
      {BLOB}/internal/processed/vvs_local/zinc_10m_indices/{size}/
      ├─ index.usearch         (binary)
      ├─ params.json           (Index() construction kwargs)
      └─ build_stats.json      (time, memory)

Run via:

    ./run.sh generate_indices
"""
from __future__ import annotations
import os, time, json, math, argparse
from pathlib import Path
from typing  import Dict, List

import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import datasets as ds
from datasets import disable_caching
from rich.console import Console
from usearch.index import Index
# ───────────────── project helpers
from vvs_local.constants        import (
    BLOB, DEVICE,
    EMB_MODEL_NM, DECOMP_MODEL_NM,
)

disable_caching()
console = Console(highlight=False)

# ─────────────────────────── paths ────────────────────────────────────
EMB_DATA_DIR = (BLOB / "internal" / "processed" / "vvs_local" /
                "zinc_10m_embedded.hf")
INDEX_ROOT   = (BLOB / "internal" / "processed" / "vvs_local" /
                "zinc_10m_indices")

# ╭───────────────────-─────────── Helpers ───────────-────────────────────╮
def _load_raw_zinc() -> ds.Dataset:
    """Read the *raw* data CSV → HF Dataset (lazy, streaming friendly)."""
    csv_path = BLOB / "internal" / "processed" / "zinc_10m" / "data.csv"
    console.log(f"[cyan]Loading Zinc-10 M CSV ({csv_path}) …")
    # load into memory first to prevent dataset CSV file copy
    df = pd.read_csv(csv_path)
    dataset = ds.Dataset.from_pandas(df)
    return dataset 

def _embed_dataset() -> ds.Dataset:
    """
    Returns the fully-embedded dataset (loads from disk if present,
    else computes it and saves).
    """
    if EMB_DATA_DIR.exists():
        console.log(f"[green]✓ Embedded dataset found - loading")
        return ds.load_from_disk(str(EMB_DATA_DIR))

    console.rule("[bold]Embedding Zinc-10 M")
    raw_ds = _load_raw_zinc()

    # ── load models once ───────────────────────────────────────────
    tok   = AutoTokenizer.from_pretrained(EMB_MODEL_NM)
    coll  = DataCollatorWithPadding(tok, return_tensors="pt")
    enc   = (AutoModel.from_pretrained(EMB_MODEL_NM,
                                       add_pooling_layer=False)
                    .to(DEVICE).eval())
    dec   = (AutoModel.from_pretrained(DECOMP_MODEL_NM,
                                       trust_remote_code=True)
                    .to(DEVICE).eval())
    comp_sizes = dec.config.comp_sizes

    tok_fn = lambda b: tok(b["item"], truncation=True, max_length=256)

    console.log("→ tokenising")
    raw_ds = raw_ds.map(tok_fn, batched=True, num_proc=max(os.cpu_count()-2, 1))

    @torch.no_grad()
    def embed_and_compress(batch):
        inputs = coll(
            {"input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]}
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        last = enc(**inputs, output_hidden_states=True
                ).hidden_states[-1]
        mask = inputs["attention_mask"]
        base = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)

        comps = dec.compress(base, comp_sizes=dec.config.comp_sizes)
        out = {}
        for s, v in comps.items():
            out[f"emb_{s}"] = v.cpu().to(torch.float16)
        return out

    console.log("→ embedding + compressing")
    emb_ds = raw_ds.map(embed_and_compress,
                        batched=True, batch_size=1_024,
                        remove_columns=["input_ids", "attention_mask"])

    console.log(f"→ saving → {EMB_DATA_DIR}")
    emb_ds.save_to_disk(str(EMB_DATA_DIR))
    return emb_ds


# ---------------------------------------------------------------------
#                  Index build for a single size                       #
# ---------------------------------------------------------------------
def _build_index(ds_emb: ds.Dataset,
                 size: int,
                 *, conn: int = 16,
                 e_add: int = 128,
                 e_search: int = 128,
                 batch: int = 4096):
    """Build a USearch HNSW index for one embedding size."""
    out_dir = INDEX_ROOT / str(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "index.usearch"

    if idx_path.exists():
        console.log(f"[green]✓ Index {size}-d already present - skipping")
        return

    console.rule(f"[bold]Building {size}-d index")
    ds_sub = ds_emb.remove_columns(
        [c for c in ds_emb.column_names if c != f"emb_{size}"])
    ds_sub.set_format(type="numpy")

    index = Index(ndim=size,
                  metric="cos",
                  dtype="f16",
                  connectivity=conn,
                  expansion_add=e_add,
                  expansion_search=e_search,
                  multi=False)

    t0 = time.time()
    for start in range(0, len(ds_sub), batch):
        vecs = ds_sub[start:start+batch][f"emb_{size}"]  # np.ndarray
        ids  = np.arange(start, start+vecs.shape[0], dtype=np.int64)
        index.add(ids, vecs)
        if start and start % (batch*25) == 0:
            console.log(f"  → added {start:,}/{len(ds_sub):,}")

    elapsed = time.time() - t0
    index.save(str(idx_path))
    stats = {
        "build_seconds": round(elapsed, 2),
        "vectors": int(len(ds_sub)),
        "memory_bytes": int(index.memory_usage),
    }
    with open(out_dir / "params.json", "w") as fh:
        json.dump({"ndim": size, "metric": "cos",
                   "connectivity": conn,
                   "expansion_add": e_add,
                   "expansion_search": e_search},
                  fh, indent=2)
    with open(out_dir / "build_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    console.log(f"[green]✓ {size}-d index built "
                f"({elapsed/60:.1f} min, {index.memory_usage/1e9:.2f} GB)")


########################################################################
#                               main                                   #
########################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Restrict to these sizes (default: all)")
    args = parser.parse_args()

    console.rule("[bold]USearch index generation")

    ds_emb = _embed_dataset()
    all_sizes = sorted(
        int(c.split("_")[1]) for c in ds_emb.column_names if c.startswith("emb_")
    )
    sizes = args.sizes or all_sizes
    console.log(f"Embedding sizes: {sizes}")

    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    for s in sizes:
        _build_index(ds_emb, s)

    console.rule("[bold green]ALL DONE")


if __name__ == "__main__":
    main()
