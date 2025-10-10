"""
Generate artefacts for the Enamine-decomposer evaluation.

 * Stage-A — uses the pre-embedded HF dataset to compute
             decomposed embeddings, ground-truth BB embeddings,
             and Top-K neighbour indices → saves one HF dataset.

 * Stage-B — tokenises + embeds a 5 k CSV slice and writes only
             Top-K neighbour DataFrames (parquet) for molecule plots.

 * Stage-C — compute accuracy and precision metrics
"""

from __future__ import annotations
import argparse, itertools
from pathlib import Path
import torch, torch.nn.functional as F
import datasets as ds
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeRemainingColumn

from .constants import *

console = Console()
ds.disable_caching()

# ╭──────────────────────────── helper factories ─────────────────────────────╮
def build_decompose_fn(decomp: AutoModel,
                       comp_sizes: list[int],
                       out_sizes:  list[int]):
    @torch.no_grad
    def _fn(batch):
        emb = batch["embedding"].to(DEVICE)
        z   = decomp.compress(emb, comp_sizes)
        de  = decomp.decompose(z, out_sizes)      # {out:[n_in,B,2,d]}
        out = {}
        for o, tensor in de.items():
            t = tensor.cpu()
            for idx, i_size in enumerate(comp_sizes):
                out[f"{i_size}->{o}"] = t[idx]    # [B,2,d]
        return out
    return _fn


def build_groundtruth_fn(bb_lookup: torch.nn.ModuleDict):
    @torch.no_grad
    def _fn(batch):
        bb_ids = torch.stack([batch["bb1_id"], batch["bb2_id"]], 1).to(DEVICE)
        out = {f"{sz}_gt": layer(bb_ids).cpu()
               for sz, layer in bb_lookup.items()}
        return out
    return _fn


def build_retrieval_fn(ref_tables: dict[int, torch.Tensor],
                       comp_sizes: list[int],
                       out_sizes:  list[int],
                       k: int):
    @torch.no_grad
    def _fn(batch):
        out = {}
        for o in out_sizes:
            ref = ref_tables[o]
            # ground truth
            if f"{o}_gt" in batch:
                gt_q = F.normalize(batch[f"{o}_gt"].to(DEVICE), 2, -1)
                gt_idx = (gt_q @ ref.T).topk(k).indices.cpu().long()
                out[f"{o}_gt_idx"] = gt_idx

            # decomposed variants
            for i in comp_sizes:
                q = F.normalize(batch[f"{i}->{o}"].to(DEVICE), 2, -1)
                idx = (q @ ref.T).topk(k).indices.cpu().long()
                out[f"{i}->{o}_idx"] = idx
        return out
    return _fn

@torch.no_grad
def precision_at_k(query_idx: torch.Tensor,
                   ref_idx:   torch.Tensor,
                   k: int,
                   batch_size: int = 512) -> float:
    q = query_idx[:, :k]
    r = ref_idx[:, :k]
    hits = []
    for q_b, r_b in zip(q.split(batch_size), r.split(batch_size)):
        m = (r_b.unsqueeze(2) == q_b.unsqueeze(1)).any(-1).sum(-1)
        hits.append(m)
    return (torch.cat(hits).float().mean() / k).item()


@torch.no_grad
def pair_recovery_at_k(pred_idx: torch.Tensor,
                       bb_targets: torch.Tensor,
                       k: int,
                       batch: int = 512) -> float:
    # pred_idx : [N, 2, K]   targets : [N, 2, 1]
    q = pred_idx[:, :, :k]
    hits = []
    flips = []
    for q_b, t_b in zip(q.split(batch), bb_targets.split(batch)):
        mask1 = q_b == t_b           # correct orientation
        mask2 = q_b == t_b.flip([1]) # swapped orientation
        recovered = (mask1.any(-1).all(-1) | 
                     mask2.any(-1).all(-1))
        hits.append(recovered)
        flips.append(mask2.any(-1).all(-1))
    return (torch.cat(hits).float().mean().item(),
            torch.cat(flips).float().mean().item())

# ╰────────────────────────────────────────────────────────────────────────────╯


def _load_assets():
    decomposer = AutoModel.from_pretrained(DECOMP_MODEL_NM,
                                           trust_remote_code=True).to(DEVICE).eval()
    bb_state = torch.load(BB_EMB_PTH, map_location="cpu")

    bb_lookup = torch.nn.ModuleDict({
        k.split('.')[0]: torch.nn.Embedding.from_pretrained(v, freeze=True)
        for k, v in bb_state.items()
    }).to(DEVICE)
    for emb in bb_lookup.values():
        emb.weight.data = F.normalize(emb.weight.data, 2, -1)

    return decomposer, bb_lookup


def stage_hf(force=False):
    if PROCESSED_DS.exists() and not force:
        console.log("[cyan]✓ HF artefact exists - skipping Stage-A")
        return

    console.rule("[bold]Stage A • HF dataset")

    decomp, bb_lookup = _load_assets()
    comp_sizes, out_sizes = decomp.config.comp_sizes, decomp.config.output_sizes

    ds_hf = ds.load_from_disk(str(HF_DATASET)).select(range(QUERY_SZ))
    ds_hf.set_format("pt")

    # build reusable map functions
    fn_decomp  = build_decompose_fn(decomp, comp_sizes, out_sizes)
    fn_gt      = build_groundtruth_fn(bb_lookup)
    ref_tables = {o: bb_lookup[str(o)].weight.to(DEVICE) for o in out_sizes}
    fn_retr    = build_retrieval_fn(ref_tables, comp_sizes, out_sizes, K_RETRIEVE)

    ds_hf = ds_hf.map(fn_decomp,  batched=True, batch_size=1024, desc="decompose")
    ds_hf = ds_hf.map(fn_gt,      batched=True, batch_size=1024, desc="GT lookup")
    ds_hf = ds_hf.map(fn_retr,    batched=True, batch_size=1024, desc="top-K idx")

    ds_hf.save_to_disk(str(PROCESSED_DS))
    console.log(f"[green]✓ wrote {PROCESSED_DS}")


def stage_csv(force=False):
    if TOPK_DIR.exists() and not force:
        console.log("[cyan]✓ parquet dir exists - skipping Stage-B")
        return

    console.rule("[bold]Stage B • CSV slice")

    decomp, bb_lookup = _load_assets()
    comp_sizes, out_sizes = decomp.config.comp_sizes, decomp.config.output_sizes

    csv_path = (BLOB / "internal" / "training_datasets" /
                "enamine_assembled" / "enamine_assembled.csv")
    df = next(pd.read_csv(csv_path, chunksize=5_000)).rename(
        columns=lambda x: "product" if x == "item" else x)

    # --- embed products ----------------------------------------------------
    tok  = AutoTokenizer.from_pretrained(EMB_MODEL_NM)
    coll = DataCollatorWithPadding(tok, return_tensors="pt")
    embed_model = AutoModel.from_pretrained(EMB_MODEL_NM,
                                            add_pooling_layer=False).to(DEVICE).eval()

    @torch.no_grad 
    def fn_embed(batch):
        t = coll(tok(batch["product"], truncation=True, max_length=256))
        t = {k: v.to(DEVICE) for k, v in t.items()}
        last = embed_model(**t, output_hidden_states=True).hidden_states[-1]
        mask = t["attention_mask"]
        emb  = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        return {"embedding": emb.cpu()}

    ds_csv = ds.Dataset.from_pandas(df)
    ds_csv = ds_csv.map(fn_embed,    batched=True, batch_size=512, desc="embed CSV")
    ds_csv.set_format("pt")

    fn_decomp = build_decompose_fn(decomp, comp_sizes, out_sizes)
    ref_tables = {o: bb_lookup[str(o)].weight.to(DEVICE) for o in out_sizes}
    fn_retr = build_retrieval_fn(ref_tables, comp_sizes, out_sizes, K_RETRIEVE)

    ds_csv = ds_csv.map(fn_decomp, batched=True, batch_size=512, desc="decompose CSV")
    ds_csv = ds_csv.map(fn_retr,   batched=True, batch_size=512, desc="retrieval CSV")

    # --- write parquet -----------------------------------------------------
    TOPK_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"product": df["product"]}).to_parquet(TOPK_DIR / "queries.parquet")

    enamine_smiles = pd.read_csv(
        BLOB / "internal" / "processed" / "enamine" / "data.csv",
        usecols=["item"]
    )["item"].values

    gt_df = pd.DataFrame({
        "bb1": enamine_smiles[df["bb1_id"].values],
        "bb2": enamine_smiles[df["bb2_id"].values],
    })
    gt_df.to_parquet(TOPK_DIR / "ground_truth.parquet")

    for o in out_sizes:
        for i in comp_sizes:
            key = f"{i}->{o}_idx"
            idx = ds_csv[key].numpy()        # [N, 2, K_RETRIEVE]

            # DataFrames: one for each BB position, columns n0 … n{k-1}
            bb1_cols = {f"n{k}": enamine_smiles[idx[:, 0, k]] for k in range(K_RETRIEVE)}
            bb2_cols = {f"n{k}": enamine_smiles[idx[:, 1, k]] for k in range(K_RETRIEVE)}

            pd.DataFrame(bb1_cols).to_parquet(TOPK_DIR / f"{i}->{o}_bb1.parquet")
            pd.DataFrame(bb2_cols).to_parquet(TOPK_DIR / f"{i}->{o}_bb2.parquet")

    console.log(f"[green]✓ wrote queries/ground-truth + {len(comp_sizes)*len(out_sizes)*2} " \
                f"neighbour parquet files → {TOPK_DIR}")


def stage_metrics(force=False):
    if METRICS_PATH.exists() and not force:
        console.log("[cyan]✓ metrics.pt already present - skipping Stage-C")
        return

    console.rule("[bold]Stage C • precision / accuracy curves")
    ds_hf = ds.load_from_disk(str(PROCESSED_DS)).with_format("pt")

    comp_sizes   = set(s.split("->")[0] for s in ds_hf.column_names if "->" in s)
    out_sizes    = set(s.split("->")[1].split("_")[0] for s in ds_hf.column_names if "->" in s)
    comp_sizes   = sorted(map(int, comp_sizes))
    out_sizes    = sorted(map(int, out_sizes))
    cutoffs      = list(range(1, K_RETRIEVE + 1))

    targets = torch.stack([ds_hf["bb1_id"], ds_hf["bb2_id"]], 1).unsqueeze(-1).to(DEVICE)

    prec_dict, acc_dict, flips_dict = {}, {}, {}
    for i_size in comp_sizes:
        console.log(f"• computing curves for input-size {i_size}")
        for o_size in out_sizes:
            key = f"{i_size}->{o_size}"
            pred_idx = ds_hf[f"{key}_idx"].to(DEVICE)       # [N,2,K]
            gt_idx   = ds_hf[f"{o_size}_gt_idx"].to(DEVICE) # [N,2,K]

            prec_curve = []
            acc_curve  = []
            flips_curve = []
            for k in cutoffs:
                prec_curve.append(
                    precision_at_k(pred_idx.flatten(0, 1),
                                    gt_idx.flatten(0, 1), k, 1024)
                )
                acc, flips = pair_recovery_at_k(pred_idx, targets, k, 1024)
                acc_curve.append(acc)
                flips_curve.append(flips)

            prec_dict[key] = prec_curve
            acc_dict[key]  = acc_curve
            flips_dict[key]  = flips_curve

            del pred_idx, gt_idx; torch.cuda.empty_cache()

    torch.save({"cutoffs": cutoffs,
                "precision": prec_dict,
                "accuracy":  acc_dict,
                "flips": flips_dict},
                METRICS_PATH)
    console.log(f"[green]✓ wrote metrics → {METRICS_PATH}")



# ╭───────────────────────────── CLI driver ───────────────────────────────────╮
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-compute artefacts even if they exist")
    args = ap.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    stage_hf(force=args.force)
    stage_csv(force=args.force)
    stage_metrics(force=args.force)
    console.rule("[bold green]ALL DONE")

if __name__ == "__main__":
    main()
