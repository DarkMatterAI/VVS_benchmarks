from pathlib import Path
import shutil, sys, os

comp_prefix = "training_runs/embedding_compression"
decomp_prefix = "training_runs/enamine_decomposer"

TO_EXPORT = [
    # erbb1 mlp
    ("training_runs/erbb1_mlp/checkpoints/",),

    # embedding compression
    (f"{comp_prefix}/mse_lr1e3_wd001_1_1", f"{comp_prefix}/mse_loss"),
    (f"{comp_prefix}/mse_topk_lr1e3_wd001_1_1", f"{comp_prefix}/mse_topk_loss"),
    (f"{comp_prefix}/pearson_lr1e3_wd001_1_1", f"{comp_prefix}/pearson_loss"),
    (f"{comp_prefix}/pearson_topk_lr1e3_wd001_1_1", f"{comp_prefix}/pearson_topk_loss"),
    (f"{comp_prefix}/mse_topk_lr3e4_wd001_4_4", f"{comp_prefix}/mse_topk_loss_large"),
    (f"{comp_prefix}/pearson_topk_lr1e3_wd001_4_4", f"{comp_prefix}/pearson_topk_loss_large"),
    ("model_weights/compression_heads.pt",),

    # decomposer model
    (f"{decomp_prefix}/cos_long", f"{decomp_prefix}/cos_loss"),
    (f"{decomp_prefix}/mse_long", f"{decomp_prefix}/mse_loss"),
    (f"{decomp_prefix}/ref_corr_k_long", f"{decomp_prefix}/pearson_topk_loss"),
    (f"{decomp_prefix}/cos_ref_corr_k_long", f"{decomp_prefix}/cos_pearson_topk_loss"),
    (f"{decomp_prefix}/mse_ref_corr_k_long", f"{decomp_prefix}/mse_pearson_topk_loss"),
    (f"{decomp_prefix}/cos_ref_corr_k_canonical_long", f"{decomp_prefix}/cos_pearson_topk_loss_canonical_order"),

    


    # # training datasets
    # ("training_datasets/embedding_compression/embedding_compression.csv",),
    # ("training_datasets/enamine_assembled/enamine_assembled.csv",),

    # # processed datasets
    # ("processed/chembl_erbb1_ic50",),
    # ("processed/d4_138m/data.csv",),
    # ("processed/enamine/data.csv",),
    # ("processed/openeye",),
    # ("processed/zinc_10m/data.csv",),
    # ("processed/synthemol_rf",),
]

ROOT          = Path(os.environ.get("BLOB_STORE", "../blob_store")).resolve()
SRC_ROOT      = ROOT / "internal"
DST_ROOT      = ROOT / "export"

def copy_entry(src_rel: str, dst_rel: str):
    src = SRC_ROOT / src_rel
    dst = DST_ROOT / dst_rel

    if not src.exists():
        print(f"[!]  source missing → {src}", file=sys.stderr)
        return

    if dst.exists():
        print(f"[=]  already exists  → {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

    print(f"[+]  copied {src_rel} → {dst_rel}")

def main():
    for entry in TO_EXPORT:
        if len(entry) == 1:
            rel = entry[0]
            copy_entry(rel, rel)
        elif len(entry) == 2:
            copy_entry(entry[0], entry[1])
        else:
            print(f"[!] bad entry: {entry}", file=sys.stderr)

if __name__ == "__main__":
    main()
