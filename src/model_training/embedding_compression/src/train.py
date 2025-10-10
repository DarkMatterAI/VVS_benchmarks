"""
Single-epoch training run for the CompressionHFModel.

Usage (inside container):
    python -m src.train \
        --run_name run1 \
        --input_size 768 \
        --comp_sizes 256 128 64 \
        --encoder_layers 2 \
        --mse 1.0 --topk_mse 2.0 --topk_vals 10 50

If several --run_name values are given the script trains each run in
sequence — handy for quick grid / random sweeps.
"""
import argparse, os
from pathlib import Path
import numpy as np, time, gc

import datasets as ds
import torch
from transformers import (
    Trainer,
    TrainingArguments,
)

from .configuration_compression import CompressionConfig
from .modeling_compression import CompressionHFModel

BLOB       = Path(os.environ["BLOB_STORE"]).resolve()
DATA       = BLOB / "internal" / "training_datasets" / "embedding_compression" / "embedding_compression.hf"
OUTPUT_DIR = BLOB / "internal" / "training_runs" / "embedding_compression"
VAL_SZ     = 1_500_000     # validation subset
VAL_Q      = 500_000       # number of query vectors for validation

def get_dataset(max_train=None, max_valid=None):
    full = ds.load_from_disk(str(DATA))
    valid = full.select(range(VAL_SZ))                 # first 1 M
    train = full.select(range(VAL_SZ, len(full)))      # remaining 29 M

    if max_valid is not None:
        valid = valid.select(range(max_valid))
    if max_train is not None:
        train = train.select(range(max_train))

    train_len = len(train)
    train = train.to_iterable_dataset(num_shards=16)
    train = train.with_format(type="torch")
    valid.set_format(type="torch", columns=["embedding"])
    return train, valid, train_len


# ─── KNN precision eval ──────────────────────────────────────────────────
def knn_metric(inputs, k, queries=10_000, batch_size=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = inputs.to(device)
    inputs = torch.nn.functional.normalize(inputs, p=2, dim=-1)
    q = inputs[:queries]
    r = inputs[queries:]
    indices = []
    for batch in q.split(batch_size):
        knn = (batch @ r.t()).topk(k, dim=-1, largest=True).indices
        indices.append(knn)
    
    indices = torch.cat(indices)
    del q, r, inputs
    gc.collect()
    return indices

def precision_from_indices(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> float:
    # shape broadcasting  →  (Q, k, k)
    matches = (pred_idx.unsqueeze(2) == gt_idx.unsqueeze(1)).any(dim=2)
    # matches[q, j] == True  ⇔  pred_idx[q, j] ∈ gt_idx[q, :]
    hits_per_query = matches.sum(dim=1)                     # [Q]
    precision = hits_per_query.float().mean().item() / gt_idx.size(1)
    return precision

def collate(batch):               # dataset already holds tensors
    return {"embedding": torch.stack([b["embedding"] for b in batch])}

def compute_metrics(pred):
    torch.cuda.empty_cache()
    ks = args.eval_ks
    index_times = {}
    
    base_emb = pred.label_ids
    loss_terms = []
    comp_dicts = []
    
    for batch_results in pred.predictions:
        if len(batch_results)==2:
            loss_term, comp_dict = batch_results
        else:
            loss_term, comp_dict, decoder_output = batch_results
        
        loss_terms.append(loss_term)
        comp_dicts.append(comp_dict)
        
    with torch.inference_mode():
        start = time.time()
        base_emb = torch.cat([torch.from_numpy(i) for i in base_emb])
        queries = args.eval_queries if args.eval_queries else VAL_Q
        queries = min(queries, base_emb.shape[0])
        gt_ids = knn_metric(base_emb, max(ks), queries=queries)
        torch.cuda.empty_cache()
        elapsed = time.time() - start
        index_times["gt_index"] = elapsed
        
        metrics = {}
        for size in comp_dicts[0].keys():
            z = torch.cat([torch.from_numpy(i[size]) for i in comp_dicts])
            start = time.time()
            z_ids = knn_metric(z, max(ks), queries=queries)
            for k in ks:
                prec = precision_from_indices(gt_ids[:,:k], z_ids[:,:k])
                metrics[f"p@{k}_{size}"] = prec
            del z_ids 
            torch.cuda.empty_cache()
            elapsed = time.time() - start 
            index_times[f"size{size}_index"] = elapsed
        
    losses = {}
    for key in loss_terms[0].keys():
        value = np.array([i[key] for i in loss_terms]).mean()
        losses[key] = value
        
    losses.update(metrics)
    losses.update(index_times)
    return losses


# ─── cli -----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--run_name", nargs="+", required=True)
p.add_argument("--input_size", type=int, default=768)
p.add_argument("--comp_sizes", nargs="+", type=int, required=True)
p.add_argument("--encoder_layers", type=int, default=1)
p.add_argument("--decoder_layers", type=int, default=0)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--layer_norm_eps", type=float, default=1e-12)
p.add_argument("--use_layer_norm", type=bool, default=True)
p.add_argument("--mse", type=float, default=0.0)
p.add_argument("--topk_mse", type=float, default=0.0)
p.add_argument("--topk_vals", nargs="*", type=int, default=[])
p.add_argument("--rank_mse_weight", type=float, default=0.0)
p.add_argument("--pearson_loss_weight", type=float, default=0.0)
p.add_argument("--margin_ranking_weight", type=float, default=0.0)
p.add_argument("--margin", type=float, default=0.2)
p.add_argument("--margin_strategy", type=str, default="top1-vs-median")
p.add_argument("--margin_k", type=int, default=5)
p.add_argument("--decoder_cosine_weight",  type=float, default=0.0)
p.add_argument("--decoder_pairwise_weight",  type=float, default=0.0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr",  type=float, default=3e-4)
p.add_argument("--wd",  type=float, default=0.01)
p.add_argument("--warmup_ratio",  type=float, default=0.1)
p.add_argument("--eval_queries",  type=int, default=10_000)
p.add_argument("--eval_ks", nargs="+", type=int, required=True)
p.add_argument("--max_train_size", type=int, default=None,
               help="Optional cap on number of training rows")
p.add_argument("--max_valid_size", type=int, default=None,
               help="Optional cap on number of validation rows")
args = p.parse_args()

train_ds, val_ds, train_len = get_dataset(max_train=args.max_train_size, max_valid=args.max_valid_size)
eval_accum_max = 2048*32
batch_size = 1024*3
eval_accum = eval_accum_max//batch_size

for run in args.run_name:
    if args.use_layer_norm:
        ln = args.layer_norm_eps
    else:
        ln = None 
    cfg = CompressionConfig(
        input_size=args.input_size,
        compression_sizes=args.comp_sizes,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        layer_norm_eps=ln,
        mse_loss_weight=args.mse,
        topk_mse_loss_weight=args.topk_mse,
        topk_values=args.topk_vals,
        rank_mse_weight=args.rank_mse_weight,
        pearson_loss_weight=args.pearson_loss_weight,
        margin_ranking_weight=args.margin_ranking_weight,
        margin=args.margin,
        margin_strategy=args.margin_strategy,
        margin_k=args.margin_k,
        decoder_cosine_weight=args.decoder_cosine_weight,
        decoder_pairwise_weight=args.decoder_pairwise_weight
    )

    model = CompressionHFModel(cfg)
    output_dir = OUTPUT_DIR / run 
    max_steps = args.epochs * (train_len//batch_size)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        run_name=run,
        max_steps=max_steps,
        save_total_limit=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_eval=True,
        eval_do_concat_batches=False,
        label_names=['embedding'],
        eval_strategy="epoch",
        # eval_steps=max_steps,
        eval_accumulation_steps=eval_accum,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=500,
        save_strategy="epoch",
        # save_steps=3000,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_safetensors=True,
        dataloader_drop_last=True,
        dataloader_num_workers=8
    )

    tr = Trainer(
        model,
        targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    tr.train()
