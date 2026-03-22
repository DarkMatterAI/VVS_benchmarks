"""
Stage 1 (CPU)  :  • load the CSVs
                  • canonical-ordering of (bb1_id, bb2_id)
                  • Hugging-Face dataset w/ products and ordered bb-ids
                  • tokenize products, split into GPU-count shards, length-sort

Stage 2 (GPU)  :  • embed products with entropy/roberta_zinc_480m
                  • embed every building-block once with
                    - roberta encoder
                    - compression heads 32/64/128/256/512 → 'canonical 768' size
                  • write
                       .../enamine_assembled/enamine_assembled.hf
                       .../enamine_assembled/bb_embeddings.pt
"""

import os, json, shutil, multiprocessing, time
from pathlib import Path
from functools import partial

import pandas as pd
import ast 
import datasets as ds
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, RobertaModel, DataCollatorWithPadding
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeRemainingColumn

from .modeling_decomposer import get_compression_heads
# ─────────────────────────────────────────────────────────────────────────────
console = Console()
BLOB = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()

BASE          = BLOB / "internal" / "training_datasets" / "enamine_assembled"
CSV_ASM       = BASE / "enamine_assembled.csv"
CSV_BB        = BLOB / "internal" / "processed" / "enamine" / "data.csv"

TOK_PATH      = BASE / "tokenised.hf"
OUT_PROD_HF   = BASE / "enamine_assembled.hf"
OUT_BB_PTH    = BASE / "bb_embeddings.pt"
CACHE_DIR     = BASE / "hf_cache"

EMBED_MODEL   = "entropy/roberta_zinc_480m"
COMP_HEADS_PTH= BLOB / "internal" / "model_weights" / "compression_heads.pt"
BATCH_EMB     = 1024
CPU_PROCS     = max(multiprocessing.cpu_count()-2, 2)
SHUFFLE_SEED  = 42

INPUT_SIZE    = 768
COMP_SIZES    = [32, 64, 128, 256, 512, 768]   # heads + native
COMP_LAYERS   = 4
N_GPUS        = torch.cuda.device_count() or 1

# ----------------------------------------------------------------- stage-0

def harmonise_csvs() -> None:
    """
    • reorder (bb1_id, bb2_id) so that bb1 is always the *shorter* SMILES
    • add a `canonical_order` column  (True = not flipped, False = flipped)
    • write the result incrementally to
          …/training_datasets/enamine_assembled/assembled_ordered.csv
    """
    out_csv = BASE / "assembled_ordered.csv"
    if out_csv.exists():
        console.log("[cyan]✓ ordered CSV already present")
        return

    console.rule("[bold]Canonicalising building-block order")

    # --- 1.  id → SMILES lookup lives happily in memory -------------------
    df_bb  = pd.read_csv(CSV_BB, usecols=["item"])        # ~7 M rows → < 1 GB
    id2sm  = df_bb.item.to_dict()                         # {row_idx: smiles}

    # --- 2.  process assembly CSV chunk-wise ------------------------------
    # CHUNK   = 1_000_000
    CHUNK   = 100_000
    first   = True
    total   = 0
    with Progress(SpinnerColumn(),
                  "[progress.description]{task.description}",
                  TimeRemainingColumn(),
                  console=console) as prog, \
         pd.read_csv(CSV_ASM, chunksize=CHUNK) as reader, \
         out_csv.open("w", newline="") as fh_out:

        writer = None                                     # lazy init
        task   = prog.add_task("ordering", total=None)

        for chunk in reader:
            # map bb-ids → SMILES  (vectorised)
            sm1 = chunk.bb1_id.map(id2sm)
            sm2 = chunk.bb2_id.map(id2sm)

            # Decide which rows need flipping
            canonical = (sm1.str.len() < sm2.str.len())
            new_bb1   = chunk.bb1_id.where(canonical, chunk.bb2_id)
            new_bb2   = chunk.bb2_id.where(canonical, chunk.bb1_id)

            chunk["bb1_id"]         = new_bb1
            chunk["bb2_id"]         = new_bb2
            chunk["canonical_order"]= canonical            # bool
            chunk["reaction_ids"]   = chunk.reaction_ids

            # incremental write
            if writer is None:
                writer = chunk.to_csv(fh_out, index=False, header=first)
                first  = False
            else:
                chunk.to_csv(fh_out, index=False, header=False)

            total += len(chunk)
            prog.update(task, advance=len(chunk))
            break 

    console.log(f"[green]✓ wrote {total:,} rows → {out_csv}")

# ----------------------------------------------------------------- stage-1
def tokenise_products():
    if TOK_PATH.exists():
        console.log("[cyan]✓ tokenised dataset exists")
        return

    harmonise_csvs()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    ds_raw = ds.load_dataset("csv",
                             data_files=str(BASE/"assembled_ordered.csv"),
                             split="train",
                             cache_dir=str(CACHE_DIR))

    tok_fn = lambda batch: tokenizer(batch["product"],
                                     truncation=True, max_length=256)

    ds_tok = ds_raw.map(tok_fn,
                        batched=True, batch_size=1024,
                        # num_proc=CPU_PROCS, remove_columns=["product", "reaction_ids"],
                        num_proc=CPU_PROCS, remove_columns=["product"],
                        desc="tokenising")

    ds_tok = ds_tok.map(lambda x: {"len": len(x["input_ids"])},
                        num_proc=CPU_PROCS)
    ds_tok = ds_tok.map(lambda x: {"reaction_ids": ast.literal_eval(x["reaction_ids"])},
                        num_proc=CPU_PROCS)
    shards=[]
    for i in range(N_GPUS):
        s = ds_tok.shard(num_shards=N_GPUS,index=i,contiguous=True)
        s = s.sort("len", reverse=True).remove_columns("len")
        shards.append(s)
    ds_tok = ds.concatenate_datasets(shards).flatten_indices(num_proc=CPU_PROCS)
    ds_tok.save_to_disk(str(TOK_PATH), num_proc=min(CPU_PROCS,16))
    console.log(f"[green]✓ tokenised dataset → {TOK_PATH}")

# ----------------------------------------------------------------- helpers
def _load_heads() -> nn.ModuleDict:
    """compression heads already trained"""
    heads = get_compression_heads(INPUT_SIZE, COMP_SIZES[:-1], COMP_LAYERS, add_input_identity=True)
    heads.load_state_dict(torch.load(COMP_HEADS_PTH, map_location="cpu", weights_only=True))
    heads.eval()
    return heads

def _embed(batch, model, collator, device):
    inputs = collator({"input_ids": batch["input_ids"]})
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        out  = model(**inputs, output_hidden_states=True)
        last = out.hidden_states[-1]
        mask = inputs["attention_mask"]
        emb  = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
    return {"embedding": emb.cpu().tolist()}

def _embed_shard(gpu_idx: int, n_gpus: int):
    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
    ds_tok = ds.load_from_disk(str(TOK_PATH)
              ).shard(num_shards=n_gpus,index=gpu_idx,contiguous=True)

    console.log(f"GPU{gpu_idx}: rows={len(ds_tok):,}")

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    model     = RobertaModel.from_pretrained(EMBED_MODEL).to(device).eval()

    ds_emb = ds_tok.map(partial(_embed, model=model, collator=collator, device=device),
                        batched=True, batch_size=BATCH_EMB,
                        remove_columns=["input_ids", "attention_mask"],
                        desc=f"embed GPU{gpu_idx}")

    shard_path = OUT_PROD_HF / f"shard_{gpu_idx}"
    ds_emb.save_to_disk(str(shard_path), num_proc=min(CPU_PROCS,16))
    console.log(f"GPU{gpu_idx}: saved {shard_path}")

# ----------------------------------------------------------------- stage-2
def embed_products():
    if OUT_PROD_HF.exists():
        console.log("[green]✓ product embeddings exist")
        return

    tokenise_products()
    OUT_PROD_HF.mkdir(parents=True, exist_ok=True)

    if N_GPUS>1:
        with multiprocessing.get_context("spawn").Pool(N_GPUS) as pool:
            pool.starmap(_embed_shard, [(i,N_GPUS) for i in range(N_GPUS)])
    else:
        _embed_shard(0,1)

    # merge & shuffle shards
    shards=[ds.load_from_disk(str(OUT_PROD_HF/f"shard_{i}")) for i in range(N_GPUS)]
    full=ds.concatenate_datasets(shards).shuffle(seed=SHUFFLE_SEED
            ).flatten_indices(num_proc=CPU_PROCS)
    full.save_to_disk(str(OUT_PROD_HF), num_proc=min(CPU_PROCS,16))
    for p in OUT_PROD_HF.glob("shard_*"): 
        shutil.rmtree(p)
    console.rule(f"[green]✓ products saved → {OUT_PROD_HF}")

# ---------------------------------------------------------------- building-block embeddings
@torch.no_grad()
def embed_building_blocks():
    if OUT_BB_PTH.exists():
        console.log("[green]✓ bb-embedding file exists")
        return

    console.rule("[bold]Embedding building blocks")
    df_bb   = pd.read_csv(CSV_BB)[["external_id","item"]].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    model     = RobertaModel.from_pretrained(EMBED_MODEL).eval().to("cuda:0")

    heads = _load_heads().to("cuda:0")
    emb_dict = {d: torch.empty(len(df_bb), d) for d in COMP_SIZES}

    bs = 1024
    for i in range(0, len(df_bb), bs):
        batch = df_bb.item.iloc[i:i+bs].tolist()
        toks  = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        toks  = {k:v.to("cuda:0") for k,v in toks.items()}
        out  = model(**toks, output_hidden_states=True).hidden_states[-1]
        mask = toks["attention_mask"]
        base = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)  # [b,768]

        for d in COMP_SIZES:
            emb = heads[str(d)](base).cpu()
            emb_dict[d][i:i+len(batch)] = emb

    # nn.Embedding wrappers
    embed_modules = nn.ModuleDict({
        str(d): nn.Embedding.from_pretrained(emb_dict[d], freeze=True)
        for d in COMP_SIZES
    })
    torch.save(embed_modules.state_dict(), OUT_BB_PTH)
    console.log(f"[green]✓ building-block embeddings → {OUT_BB_PTH}")

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    embed_products()
    embed_building_blocks()
    if (BASE/"assembled_ordered.csv").exists():
        (BASE/"assembled_ordered.csv").unlink()
    if TOK_PATH.exists():
        shutil.rmtree(TOK_PATH)
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    console.rule(f"[bold green]ALL DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
