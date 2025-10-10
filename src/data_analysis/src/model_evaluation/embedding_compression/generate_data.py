#!/usr/bin/env python
"""
Data-generation pipeline for embedding-compression evaluation.

Run:
    python -m model_evaluation.embedding_compression.generate_data \
        [--force]
"""
from __future__ import annotations
import argparse, json, gc, pickle, torch
import duckdb, pandas as pd, pyarrow as pa, datasets as ds
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from .constants import *

console = Console()
ds.disable_caching()

# ------------------------------------------------------------------ helpers
def _sample_smiles(out_file: Path, *, force=False):
    if out_file.exists() and not force:
        console.log("[cyan]✓ sample CSV exists")
        return
    console.rule("[bold]Sampling SMILES from D4")
    con = duckdb.connect(str(QUERY_DB), read_only=True)
    sql = f"""
        SELECT item
        FROM {TABLE}
        USING SAMPLE reservoir({QUERY_SZ + REF_SZ})   -- reproducible & memory-friendly
    """
    tbl = con.execute(sql)
    df = tbl.df()
    df.to_csv(out_file, index=False)
    console.log(f"[green]✓ wrote {len(df):,} rows → {out_file}")

def precision_at_k(query_idx: torch.Tensor,
                   ref_idx:   torch.Tensor,
                   k: int,
                   batch_size: int = 512) -> float:
    q = query_idx[:, :k]
    r = ref_idx[:, :k]
    hits = []
    for q_b, r_b in zip(q.split(batch_size), r.split(batch_size)):
        m  = (r_b.unsqueeze(2) == q_b.unsqueeze(1)).any(dim=2).sum(dim=1)
        hits.append(m)
    hits = torch.cat(hits)
    return (hits.float().mean() / k).item()

def _embed_and_compress(csv_file: Path, out_dir: Path,
                        device="cuda", force=False):
    if out_dir.exists() and not force:
        console.log("[cyan]✓ embeddings dataset present"); return

    console.rule("[bold]Tokenize + Embed + Compress")
    tok  = AutoTokenizer.from_pretrained(EMBED_MODEL)
    coll = DataCollatorWithPadding(tok, return_tensors="pt")

    embed = AutoModel.from_pretrained(EMBED_MODEL, add_pooling_layer=False
                 ).to(device).eval()
    comp  = AutoModel.from_pretrained(COMPRESS_MODEL, trust_remote_code=True
                 ).to(device).eval()
    sizes = comp.config.compression_sizes

    ds_raw = ds.load_dataset("csv", data_files=str(csv_file), split="train")

    tok_fn = lambda b: tok(b["item"], truncation=True, max_length=256)
    ds_tok = ds_raw.map(tok_fn, batched=True, batch_size=2_048, num_proc=os.cpu_count()-2)

    @torch.no_grad()
    def _f(batch):
        inputs = coll({"input_ids": batch["input_ids"]})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last = embed(**inputs, output_hidden_states=True).hidden_states[-1]
        mask = inputs["attention_mask"]
        base = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        out  = {f"emb_{base.size(1)}": base.cpu()}
        cmp  = comp.compress(base, compression_sizes=sizes)
        for k,v in cmp.items():
            out[f"emb_{k}"] = v.cpu()
        return out

    ds_emb = ds_tok.map(_f, batched=True, batch_size=1024, desc="embed")
    ds_emb.save_to_disk(str(out_dir))
    console.log(f"[green]✓ embeddings saved → {out_dir}")

@torch.no_grad
def _knn_and_precision(emb_dir: Path, sample_csv: Path,
                       device="cuda", force=False):
    if all(p.exists() for p in (KNN_PTH, PREC_PTH, TOPK_DIR)) and not force:
        console.log("[cyan]✓ KNN / precision artefacts exist"); return
    console.rule("[bold]K-NN & precision curves")

    ds_emb = ds.load_from_disk(str(emb_dir)).with_format("pt")
    knn = {}
    for sz in SIZES:
        v = ds_emb[f"emb_{sz}"].to(device)
        v = torch.nn.functional.normalize(v, 2, -1)
        q, r = v[:QUERY_SZ], v[QUERY_SZ:]
        idx = []
        for b in q.split(BATCH_KNN):
            idx.append((b @ r.T).topk(K_NN, dim=-1, largest=True).indices + QUERY_SZ)
        knn[sz] = torch.cat(idx).cpu()
        del v, q, r; torch.cuda.empty_cache()
    torch.save(knn, KNN_PTH)

    # ---------- precision to GT ----------
    gt = knn[768]
    prec_to_gt, prec_agree = {}, {}
    for sz in SIZES:
        if sz == 768: continue
        prec_to_gt[sz] = [precision_at_k(knn[sz], gt, k, BATCH_KNN)
                          for k in CUTS]

    # ---------- size-pair agreement ----------
    for i, s1 in enumerate(SIZES):
        for s2 in SIZES[i+1:]:
            prec_agree[f"{s1}_{s2}"] = [
                precision_at_k(knn[s1], knn[s2], k, BATCH_KNN) for k in CUTS
            ]

    torch.save({
        "cutoffs": CUTS,
        "to_gt": prec_to_gt,
        "agreement": prec_agree}, PREC_PTH)

    # ---------- Top-50 neighbour SMILES ----------
    df = pd.read_csv(sample_csv)            # original sample order
    TOPK_DIR.mkdir(parents=True, exist_ok=True)
    df.iloc[:QUERY_SZ].to_parquet(TOPK_DIR / "queries.parquet", index=False)

    k = 50
    for sz, idx in knn.items():
        neigh = pd.DataFrame(df.item.values[idx[:, :k]], columns=[f"n{i}" for i in range(k)])
        neigh.to_parquet(TOPK_DIR / f"topk_{sz}.parquet", index=False)

    console.log(f"[green]✓ saved precision → {PREC_PTH.name}, "
                f"KNN → {KNN_PTH.name}, top-K parquet files")

# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run every stage even if artefacts exist")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE.mkdir(parents=True, exist_ok=True)
    _sample_smiles(RAW_CSV, force=args.force)
    _embed_and_compress(RAW_CSV, EMB_DS, device=device, force=args.force)
    _knn_and_precision(EMB_DS, RAW_CSV, device=device, force=args.force)
    console.rule("[bold green]ALL DONE")

if __name__ == "__main__":
    main()
