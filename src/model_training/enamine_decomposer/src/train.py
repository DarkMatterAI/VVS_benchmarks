"""
Training run for the DecomposerModel.

Usage (inside container):
    python -m src.train \
        --run_name run1 \
        --input_size 768 \
        --comp_sizes 256 128 64 \
        --output_sizes 256 128 64 \
        --cosine_weight 1.0 --input_corr

If several --run_name values are given the script trains each run in
sequence — handy for quick grid / random sweeps.
"""

import argparse, os
from pathlib import Path
from collections import defaultdict
import numpy as np, time, gc

import datasets as ds
from datasets import disable_caching
disable_caching()
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy("file_system")  

from transformers import (
    Trainer,
    TrainingArguments,
)

from .configuration_decomposer import DecomposerConfig
from .modeling_decomposer import DecomposerModel, FeedForwardLayer, cross_cosine
from .decomposer_conditional import (ConditionalDecomposerModel, 
                                     ConditionalDecomposerConfig, 
                                     ConditionalDecomposerCollator,
                                     REACTION_ID_TO_IDX,
                                     N_REACTIONS)

assert torch.cuda.is_available()

BLOB       = Path(os.environ["BLOB_STORE"]).resolve()
DATA_DIR   = BLOB / "internal" / "training_datasets" / "enamine_assembled"
DATA       = DATA_DIR / "enamine_assembled.hf"
BB_EMBS    = DATA_DIR / "bb_embeddings.pt"
COMP_HDS   = BLOB / "internal" / "model_weights" / "compression_heads.pt"

OUTPUT_DIR = BLOB / "internal" / "training_runs" / "enamine_decomposer"
VAL_SZ     = 1_000_000
COMP_SIZES = [32, 64, 128, 256, 512, 768] # compression heads + native 
N_ENAMINE  = 79324 # number of enamine building blocks 
DATA_WORKERS = 32
N_GPUS = 2

def canonical_order(row):
    is_canonical = row["canonical_order"]
    bb1_id = row["bb1_id"]
    bb2_id = row["bb2_id"]
    order = [bb1_id, bb2_id] if is_canonical else [bb2_id, bb1_id]
    columns = ["bb1_id", "bb2_id"]
    return {columns[i]:order[i] for i in range(len(order))}

def get_dataset(max_train=None, max_valid=None, canonical=False, conditional=False):
    full = ds.load_from_disk(str(DATA))

    valid = full.select(range(VAL_SZ))                 # first {VAL_SZ} M
    train = full.select(range(VAL_SZ, len(full)))      # remaining  50M - VAL_SZ

    if max_valid is not None:
        valid = valid.select(range(max_valid))
    if max_train is not None:
        train = train.select(range(max_train))

    if canonical:
        print("Setting canonical order")
        valid = valid.map(canonical_order, num_proc=DATA_WORKERS)
        train = train.map(canonical_order, num_proc=DATA_WORKERS)

    cols_to_remove = ["canonical_order"]
    if not conditional:
        cols_to_remove.append("reaction_ids")

    valid = valid.remove_columns(cols_to_remove)
    train = train.remove_columns(cols_to_remove)

    train_len = len(train)
    train = train.to_iterable_dataset(num_shards=1024)
    train = train.with_format(type="torch")
    valid.set_format(type="torch")
    print(train)
    print(valid)
    return train, valid, train_len

# ─── load pretrained weights and embs ────────────────────────────────────
def _load_embeddings() -> nn.ModuleDict:
    embed_modules = nn.ModuleDict({
        str(d): nn.Embedding(N_ENAMINE, d, _freeze=True)
        for d in COMP_SIZES
    })
    embed_modules.load_state_dict(torch.load(BB_EMBS, map_location="cpu", weights_only=True))
    embed_modules.eval()
    for p in embed_modules.parameters():
        p.requires_grad = False 
    return embed_modules 

# # ─── KNN precision eval ──────────────────────────────────────────────────
def precision_from_indices(gt_idx: torch.Tensor, # (B, K)
                           pred_idx: torch.Tensor # (N, B, K)
                          ) -> float:
    # shape broadcasting  →  (N, B, K, K)  →  (N, B, K)
    matches = (pred_idx.unsqueeze(-1) == gt_idx.unsqueeze(-2)).any(dim=-1)
    hits_per_query = matches.sum(dim=-1) # (N, )
    precision = hits_per_query.float().mean(-1) / gt_idx.size(1)
    return precision

class RetrievalPrecisionEval():
    def __init__(self, input_sizes, output_sizes, eval_ks, knn_batch_size=512):
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.eval_ks = eval_ks
        self.knn_batch_size = knn_batch_size
        self.ref_embs = _load_embeddings()
        for k,v in self.ref_embs.items():
            # pre-normalize for nearest neighbors 
            v.weight.data = F.normalize(v.weight.data, p=2, dim=-1)
        self.ref_embs.to("cuda:0")
        self.metrics = defaultdict(list)
        self.loss_log = defaultdict(list)
        
    @torch.no_grad
    def __call__(self, eval_pred, compute_result):
        prediction_list, (bb1_ids, bb2_ids) = eval_pred
        loss_log = prediction_list[0]
        predictions = prediction_list[1]
        
        for k,v in loss_log.items():
            if type(v)==float:
                self.loss_log[k].append(v)
            else:
                self.loss_log[k].extend(v.detach().cpu().tolist())
        
        targets = {}
        for size in self.ref_embs.keys():
            embedding = self.ref_embs[size]
            targets[int(size)] = torch.stack([embedding(bb1_ids), embedding(bb2_ids)], dim=1)
            
        for output_size, preds in predictions.items():
            # HF concats outputs from multiple GPUs along first dimension
            # we permuted in the model from [n_sizes, B, ...] -> [B, n_sizes, ...]
            # now we reverse this
            preds = preds.permute(1,0,2,3)
            targs = targets[int(output_size)]
            
            p_flat = preds.flatten(1,2) # [n_sizes, B, 2, d] -> [n_sizes, B*2, d]
            t_flat = targs.flatten(0,1) # [B, 2, d] -> [B*2, d]
            p_flat = F.normalize(p_flat, p=2, dim=-1)
            t_flat = F.normalize(t_flat, p=2, dim=-1)
            
            ref_embs = self.ref_embs[str(output_size)].weight.data
            
            for (p_batch, t_batch) in zip(p_flat.split(self.knn_batch_size, 1), 
                                          t_flat.split(self.knn_batch_size)):
                p_sims = cross_cosine(p_batch, ref_embs)
                t_sims = cross_cosine(t_batch, ref_embs)
                
                p_topk = p_sims.topk(max(self.eval_ks), dim=-1, largest=True).indices
                t_topk = t_sims.topk(max(self.eval_ks), dim=-1, largest=True).indices
                
                for k_val in self.eval_ks:
                    precision = precision_from_indices(t_topk[:, :k_val], p_topk[:, :, :k_val])
                    for i, input_size in enumerate(self.input_sizes):
                        key = f"{input_size}->{output_size}_p@{k_val}"
                        self.metrics[key].append(precision[i].item())
        
        if compute_result:
            result = {k:np.array(v).mean() for k,v in self.metrics.items()}
            loss_log = {k:np.array(v).mean() for k,v in self.loss_log.items()}
            result.update(loss_log)
            return result

# ─── cli -----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--run_name", nargs="+", required=True)
p.add_argument("--input_size", type=int, default=768)
p.add_argument("--comp_sizes", nargs="+", type=int, required=True)
p.add_argument("--output_sizes", nargs="+", type=int, required=True)
p.add_argument("--shared_dim", type=int, default=768)
p.add_argument("--n_shared_layers", type=int, default=8)
p.add_argument("--n_head_layers", type=int, default=1)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--layer_norm_eps", type=float, default=1e-12)
p.add_argument('--use_layer_norm', action="store_true", help="enable layer norm")
p.add_argument("--n_output", type=int, default=2)
p.add_argument("--n_refs_batch", type=int, default=1024)
p.add_argument("--n_refs_total", type=int, default=N_ENAMINE)
p.add_argument("--cosine_weight", type=float, default=0.0)
p.add_argument("--mse_weight", type=float, default=0.0)
p.add_argument('--self_corr', action="store_true", help="enable self correlation loss")
p.add_argument('--ref_corr', action="store_true", help="enable reference embedding correlation loss")
p.add_argument('--input_corr', action="store_true", help="enable input embedding correlation loss")
p.add_argument("--corr_weight", type=float, default=0.0)
p.add_argument("--corr_loss_type", type=str, default="pearson") # pearson or mse
p.add_argument("--corr_k_vals", nargs="*", type=int, default=[])
p.add_argument('--canonical', action="store_true", help="enable layer norm")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr",  type=float, default=3e-4)
p.add_argument("--wd",  type=float, default=0.01)
p.add_argument("--warmup_ratio",  type=float, default=0.1)
p.add_argument("--eval_ks", nargs="+", type=int, required=True)
p.add_argument("--max_train_size", type=int, default=None,
               help="Optional cap on number of training rows")
p.add_argument("--max_valid_size", type=int, default=None,
               help="Optional cap on number of validation rows")
p.add_argument("--conditional", action="store_true")
p.add_argument("--reaction_dim", type=int, default=64)
args = p.parse_args()

train_ds, val_ds, train_len = get_dataset(max_train=args.max_train_size, 
                                          max_valid=args.max_valid_size,
                                          canonical=args.canonical,
                                          conditional=args.conditional)
batch_size = 2048
eval_accum = 64 

for run in args.run_name:
    if args.use_layer_norm:
        ln = args.layer_norm_eps
    else:
        ln = None 

    if args.conditional:
        print("running conditional")
        collator = ConditionalDecomposerCollator(REACTION_ID_TO_IDX)
        cfg = ConditionalDecomposerConfig(
                input_size=args.input_size,
                n_reactions=N_REACTIONS,
                reaction_dim=args.reaction_dim,
                reaction_id_map=REACTION_ID_TO_IDX,
                comp_sizes=args.comp_sizes,
                output_sizes=args.output_sizes,
                shared_dim=args.shared_dim,
                n_shared_layers=args.n_shared_layers,
                n_head_layers=args.n_head_layers,
                dropout=args.dropout,
                layer_norm_eps=ln,
                n_output=args.n_output,
                n_refs_batch=args.n_refs_batch,
                n_refs_total=args.n_refs_total,
                cosine_weight=args.cosine_weight,
                mse_weight=args.mse_weight,
                self_corr=args.self_corr,
                ref_corr=args.ref_corr,
                input_corr=args.input_corr,
                corr_weight=args.corr_weight,
                corr_loss_type=args.corr_loss_type,
                corr_k_vals=args.corr_k_vals
                )
        model = ConditionalDecomposerModel(cfg)
    else:
        collator = None  # default
        cfg = DecomposerConfig(
            input_size=args.input_size,
            comp_sizes=args.comp_sizes,
            output_sizes=args.output_sizes,
            shared_dim=args.shared_dim,
            n_shared_layers=args.n_shared_layers,
            n_head_layers=args.n_head_layers,
            dropout=args.dropout,
            layer_norm_eps=ln,
            n_output=args.n_output,
            n_refs_batch=args.n_refs_batch,
            n_refs_total=args.n_refs_total,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            self_corr=args.self_corr,
            ref_corr=args.ref_corr,
            input_corr=args.input_corr,
            corr_weight=args.corr_weight,
            corr_loss_type=args.corr_loss_type,
            corr_k_vals=args.corr_k_vals
        )
        model = DecomposerModel(cfg)

    # cfg = DecomposerConfig(
    #     input_size=args.input_size,
    #     comp_sizes=args.comp_sizes,
    #     output_sizes=args.output_sizes,
    #     shared_dim=args.shared_dim,
    #     n_shared_layers=args.n_shared_layers,
    #     n_head_layers=args.n_head_layers,
    #     dropout=args.dropout,
    #     layer_norm_eps=ln,
    #     n_output=args.n_output,
    #     n_refs_batch=args.n_refs_batch,
    #     n_refs_total=args.n_refs_total,
    #     cosine_weight=args.cosine_weight,
    #     mse_weight=args.mse_weight,
    #     self_corr=args.self_corr,
    #     ref_corr=args.ref_corr,
    #     input_corr=args.input_corr,
    #     corr_weight=args.corr_weight,
    #     corr_loss_type=args.corr_loss_type,
    #     corr_k_vals=args.corr_k_vals
    # )

    compute_metrics = RetrievalPrecisionEval(cfg.comp_sizes, 
                                             cfg.output_sizes, 
                                             args.eval_ks,
                                             knn_batch_size=1024)


    # model = DecomposerModel(cfg)
    # heads = _load_heads()
    # embs = _load_embeddings()

    model.compression_heads.load_state_dict(torch.load(COMP_HDS, map_location="cpu", weights_only=True))
    model.ref_emb.load_state_dict(torch.load(BB_EMBS, map_location="cpu", weights_only=True))

    # model.compression_heads.load_state_dict(heads.state_dict())
    model.compression_heads.eval()
    for p in model.compression_heads.parameters():
        p.requires_grad = False

    # model.ref_emb.load_state_dict(embs.state_dict())
    model.ref_emb.eval()
    for p in model.ref_emb.parameters():
        p.requires_grad = False

    output_dir = OUTPUT_DIR / run 
    steps_per_epoch = train_len // batch_size // N_GPUS
    max_steps = args.epochs * steps_per_epoch

    targs = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        run_name=run,
        max_steps=max_steps,
        # num_train_epochs=args.epochs,
        save_total_limit=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        do_eval=True,
        eval_do_concat_batches=False,
        label_names=['bb1_id', 'bb2_id'],
        # eval_strategy="epoch",
        eval_strategy="steps",
        eval_steps=train_len//batch_size//N_GPUS,
        batch_eval_metrics=True,
        eval_accumulation_steps=eval_accum,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=500,
        # save_strategy="epoch",
        save_strategy="steps",
        save_steps=steps_per_epoch,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_safetensors=True,
        dataloader_drop_last=True,
        dataloader_num_workers=DATA_WORKERS,
        dataloader_persistent_workers=False,
    )

    tr = Trainer(
        model,
        targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    tr.train()
