"""
Stage 1  (CPU):  tokenise the 30 M-row CSV with many processes.
Stage 2  (GPU):  each GPU embeds its shard and writes to disk.
Result:
    …/embedding_compression/embedding_compression.hf/
"""

import os, json, shutil, multiprocessing, torch
from pathlib import Path
import time 

import datasets as ds
from transformers import AutoTokenizer, RobertaModel, DataCollatorWithPadding
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeRemainingColumn

console = Console()

# ─── paths ──────────────────────────────────────────────────────
BLOB        = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
BASE        = BLOB / "internal" / "training_datasets" / "embedding_compression"
CSV_PATH    = BASE / "embedding_compression.csv"
TOK_PATH    = BASE / "tokenised.hf"           # intermediate shards
OUT_PATH    = BASE / "embedding_compression.hf"
STATS_PATH  = BASE / "embedding_compression_stats.json"
CACHE_DIR   = BASE / "hf_cache"

EMBED_MODEL  = "entropy/roberta_zinc_480m"
BATCH_EMB    = 1024
CPU_PROCS    = max(multiprocessing.cpu_count() - 2, 2)
SHUFFLE_SEED = 42

# ─── tokenisation (CPU parallel) ───────────────────────────────
def tokenize_dataset(num_shards: int):
    """
    Tokenise the full CSV with CPU workers, then:
      • add a `length` column (number of tokens)
      • split into `num_shards` contiguous shards
      • sort each shard by length (long → short)

    Sorting makes batches within each GPU shard more uniform,
    which reduces padding → fewer wasted FLOPs during embedding.
    We do this on the shard level to give each GPU the same 
    length distribution.
    """
    if TOK_PATH.exists():
        console.log("[cyan]✓ tokenised dataset already cached")
        return

    console.rule("[bold]Tokenising CSV → HF dataset")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    ds_raw = ds.load_dataset(
        "csv",
        data_files=str(CSV_PATH),
        split="train",
        cache_dir=str(CACHE_DIR),
    )

    # CPU-parallel tokenisation
    tok_fn = lambda batch: tokenizer(batch["item"], truncation=True, max_length=256)
    ds_tok = ds_raw.map(
        tok_fn,
        batched=True,
        batch_size=1024,
        num_proc=CPU_PROCS,
        remove_columns=["item"],
        desc="tokenising",
    )

    # store sequence length to sort by it later
    ds_tok = ds_tok.map(
        lambda x: {"length": len(x["input_ids"])},
        num_proc=CPU_PROCS,
    )

    # create length-sorted shards (one per GPU)
    shards = []
    for i in range(num_shards):
        shard = ds_tok.shard(num_shards=num_shards, index=i, contiguous=True)
        shard = shard.sort("length", reverse=True)   # longest first
        shard = shard.remove_columns("length")       # keep dataset slim
        shards.append(shard)

    # concatenate back → deterministic order but shard-wise sorted
    ds_tok = ds.concatenate_datasets(shards).flatten_indices(num_proc=CPU_PROCS)

    ds_tok.save_to_disk(str(TOK_PATH), num_proc=min(CPU_PROCS, 16))
    console.log(f"[green]✓ saved tokenised dataset to {TOK_PATH}")


# ─── embedding helper (one GPU) ─────────────────────────────────
def embed_shard(idx: int, num_shards: int):
    device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    ds_tok = ds.load_from_disk(str(TOK_PATH)).shard(
        num_shards=num_shards, index=idx, contiguous=True
    )
    console.log(f"GPU{idx}: rows = {len(ds_tok):,}")

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)  # just for pad info
    collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    model     = RobertaModel.from_pretrained(EMBED_MODEL).to(device).eval()

    def _embed(batch):
        inputs = collator({"input_ids" : batch["input_ids"]})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out  = model(**inputs, output_hidden_states=True)
            last = out.hidden_states[-1]
            mask = inputs["attention_mask"]
            emb  = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        return {"embedding": emb.cpu().tolist()}

    ds_emb = ds_tok.map(
        _embed,
        batched=True,
        batch_size=BATCH_EMB,
        desc=f"GPU{idx} embedding",
        remove_columns=ds_tok.column_names,
    )

    shard_path = OUT_PATH / f"shard_{idx}"
    ds_emb.save_to_disk(str(shard_path), num_proc=min(CPU_PROCS, 16))
    console.log(f"GPU{idx}: shard saved → {shard_path}")


# ─── main driver ───────────────────────────────────────────────
def main():
    if OUT_PATH.exists():
        console.log("[green]✓ final dataset exists - nothing to do")
        return

    if not CSV_PATH.exists():
        console.log(f"[red]✗ missing {CSV_PATH}")
        return

    start = time.time()
    n_gpus = torch.cuda.device_count() or 1
    tokenize_dataset(n_gpus)

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    console.rule(f"[bold]Embedding with {n_gpus} GPU(s)")

    if n_gpus > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_gpus) as pool:
            pool.starmap(embed_shard, [(i, n_gpus) for i in range(n_gpus)])
    else:
        embed_shard(0, 1)

    # merge shards
    shards = [ds.load_from_disk(str(OUT_PATH / f"shard_{i}")) for i in range(n_gpus)]
    full   = ds.concatenate_datasets(shards)
    full   = full.shuffle(seed=SHUFFLE_SEED) # shuffle after sorting
    full   = full.flatten_indices(num_proc=CPU_PROCS)
    full.save_to_disk(str(OUT_PATH), num_proc=min(CPU_PROCS, 16))

    # cleanup
    for p in OUT_PATH.glob("shard_*"):
        shutil.rmtree(p)
    if TOK_PATH.exists():
        shutil.rmtree(TOK_PATH)
    if CACHE_DIR:
        shutil.rmtree(CACHE_DIR)

    elapsed = time.time() - start 
    json.dump({"rows": len(full), "time": elapsed}, STATS_PATH.open("w"))
    console.log(f"[green]✓ embedding dataset finished in {elapsed/60} minutes")
    console.rule(f"[bold green]Saved → {OUT_PATH}   rows={len(full):,}")

if __name__ == "__main__":
    main()

