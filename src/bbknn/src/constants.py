from pathlib import Path
import os
import torch

# Disable HF tokenizer multi-proc noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BLOB         = Path("/code/blob_store").resolve()
DB_PATH      = BLOB / "internal" / "processed" / "d4_138m" / "database.db"
TBL_NAME     = "d4_138m"
ASSM_CSV     = BLOB / "internal" / "training_datasets" / "enamine_assembled" / "enamine_assembled.csv"
BB_EMB_PTH   = BLOB / "internal" / "training_datasets" / "enamine_assembled" / "bb_embeddings.pt"
BB_CSV_PTH   = BLOB / "internal" / "processed" / "enamine" / "data.csv"
REACTION_PKL = BLOB / "internal" / "processed" / "enamine" / "enamine_id_to_reaction.pkl"
NATPROD_CSV  = BLOB / "internal" / "processed" / "natural_products" / "data.csv"
EGFR_CSV     = BLOB / "internal" / "processed" / "chembl_erbb1_ic50" / "data.csv"
OUT_DIR      = BLOB / "internal" / "bbknn"

DECOMP_MODEL_NM = "entropy/roberta_zinc_enamine_decomposer"
EMB_MODEL_NM    = "entropy/roberta_zinc_480m"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
