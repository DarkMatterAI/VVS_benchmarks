"""
USearch HNSW (128-d) index for the first 100 M molecules of d4_138m.

Launch with:

    ./run.sh build_d4_index  [--rows 100000000] [--batch_rows 1000000]
"""
from __future__ import annotations
import os, time, json, argparse, multiprocessing as mp
from pathlib import Path
from typing import List, Dict

import duckdb, numpy as np, torch
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from rich.console import Console
from usearch.index import Index

# ─ project helpers ----------------------------------------------------------
from vvs_local.constants import (BLOB, D4_DB_PATH, D4_TBL_NAME,
                                 EMB_MODEL_NM, DECOMP_MODEL_NM)

console  = Console(highlight=False)
OUT_DIR  = (BLOB / "internal" / "processed" / "vvs_local" /
            "d4_index_128").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
IDX_PATH = OUT_DIR / "index.usearch"

EMB_SIZE       = 128
EMB_BATCH_SIZE = 2048
DEFAULT_ROWS   = 100_000_000          # first 100 M
DEFAULT_CHUNK  = 2_000_000            # rows pulled from DuckDB at once

# ╭────────────────── worker set-up & embed function ─────────────────────────╮
def _init_worker(embed_batch: int):
    global DEV, COLL, ENC, DEC, EMB_BATCH
    rank       = mp.current_process()._identity[0] - 1
    DEV        = f"cuda:{rank}"
    TOK        = AutoTokenizer.from_pretrained(EMB_MODEL_NM, use_fast=False)
    COLL       = DataCollatorWithPadding(TOK, return_tensors="pt")
    ENC        = (AutoModel.from_pretrained(EMB_MODEL_NM,
                                            add_pooling_layer=False)
                         .to(DEV).eval())
    DEC        = (AutoModel.from_pretrained(DECOMP_MODEL_NM,
                                            trust_remote_code=True)
                         .to(DEV).eval())
    EMB_BATCH  = embed_batch
    console.log(f"[cyan]Worker on {DEV} ready (batch={EMB_BATCH})")

@torch.no_grad()
def _embed_token_batch(batch: Dict[str, List]):
    """Embed *one* tokenised mini-batch (<= EMB_BATCH items)."""
    ex = {"input_ids": batch["input_ids"]}
    inputs = COLL(ex).to(DEV)
    last   = ENC(**inputs, output_hidden_states=True).hidden_states[-1]
    mask   = inputs["attention_mask"]
    emb768 = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
    z128   = DEC.compress(emb768, [EMB_SIZE])[EMB_SIZE]      # [n,128] fp16
    return np.asarray(batch["rowid"], dtype=np.int64), z128.cpu().numpy()

# ╭──────────────────────────── main driver ───────────────────────────────────╮
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--rows",        type=int, default=DEFAULT_ROWS)
    pa.add_argument("--batch_rows",  type=int, default=DEFAULT_CHUNK)
    args = pa.parse_args()

    if IDX_PATH.exists():
        console.log(f"[green]✓ {IDX_PATH} already exists - nothing to do")
        return

    # ------------------------- initialise USearch ---------------------------
    index = Index(ndim=EMB_SIZE, 
                  metric="cos", 
                  dtype="f16",
                  connectivity=16, 
                  expansion_add=128,
                  expansion_search=128, 
                  multi=False)

    # ------------------------- DuckDB connection ---------------------------
    con = duckdb.connect(str(D4_DB_PATH), read_only=True)

    # ------------------------- worker pool ---------------------------------
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus))
    print(gpu_ids)
    n_proc  = max(len(gpu_ids), 1)
    mp.set_start_method("spawn", force=True)

    pool = mp.Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(EMB_BATCH_SIZE,)
    )

    start = time.time()
    added = 0
    chunk_id = 0

    try:
        while added < args.rows:
            limit = min(args.batch_rows, args.rows - added)
            console.log(f"[blue]Fetch rows {added:,} - {added+limit-1:,}")
            df = con.sql(f"""
                SELECT rowid, item
                FROM {D4_TBL_NAME}
                LIMIT {limit} OFFSET {added}
            """).df()

            # -------- tokenise ONCE on CPU ----------------------------------
            tok = AutoTokenizer.from_pretrained(EMB_MODEL_NM)
            tok_out = tok(df["item"].tolist(),
                          truncation=True, max_length=256)
            tok_out["rowid"] = df["rowid"].tolist()

            # -------- slice into GPU-safe batches ---------------------------
            tasks = []
            for i in range(0, len(df), EMB_BATCH_SIZE):
                tasks.append({
                    "rowid": tok_out["rowid"][i:i+EMB_BATCH_SIZE],
                    "input_ids": tok_out["input_ids"][i:i+EMB_BATCH_SIZE],
                })

            # -------- async embed & add to index ----------------------------
            for ids, vecs in pool.imap_unordered(_embed_token_batch, tasks, chunksize=1):
                index.add(ids, vecs)

            added   += len(df)
            chunk_id += 1
            console.log(f"[green]Chunk {chunk_id} done ({added:,}/{args.rows:,})")

    finally:
        pool.close(); pool.join()

    # ------------------------- persist artefacts ---------------------------
    build_s = time.time() - start
    index.save(str(IDX_PATH))
    json.dump({
        "ndim": EMB_SIZE, 
        "metric": "cos",
        "connectivity": 16, 
        "expansion_add": 128,
        "expansion_search": 128
    }, open(OUT_DIR / "params.json", "w"), indent=2)
    json.dump({
        "build_seconds": round(build_s, 2),
        "vectors": added,
        "memory_bytes": int(index.memory_usage)
    }, open(OUT_DIR / "build_stats.json", "w"), indent=2)

    console.rule(f"[bold green]✓ Index built - {added:,} vecs, "
                 f"{index.memory_usage/1e9:.2f} GB, {build_s/60:.1f} min")

if __name__ == "__main__":
    main()
