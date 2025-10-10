"""
Central configuration - change sizes / paths here only.
"""
import torch 
from pathlib import Path
import os

BLOB = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()

# ─── artefact locations ────────────────────────────────────────────────────
BASE             = BLOB / "internal" / "data_analysis" / "model_evaluation" / "enamine_decomposer"
PROCESSED_DS     = BASE / "decomposer_eval.hf"          # full HF dataset incl. indices
TOPK_DIR         = BASE / "topk_csv"                    # parquet dirs for CSV-based eval
METRICS_PATH     = BASE / "metrics.pt"
# (anything under BASE is considered cached & idempotent)

# ─── pretrained model & static inputs ──────────────────────────────────────
HF_DATASET       = BLOB / "internal" / "training_datasets" / "enamine_assembled" / "enamine_assembled.hf"
BB_EMB_PTH       = BLOB / "internal" / "training_datasets" / "enamine_assembled" / "bb_embeddings.pt"
DECOMP_MODEL_NM  = "entropy/roberta_zinc_enamine_decomposer"
EMB_MODEL_NM     = "entropy/roberta_zinc_480m"

# ─── evaluation sizes / constants ──────────────────────────────────────────
QUERY_SZ   = 20_000                 # number of product rows from HF dataset
K_RETRIEVE = 50                     # top-K neighbours to save
BATCH_EMB  = 1_024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
