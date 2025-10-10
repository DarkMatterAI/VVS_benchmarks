from pathlib import Path
import os

# -------------------------------------------------------------------------
BLOB = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()

# location of the BB-KNN results produced earlier
BBKNN_CSV   = BLOB / "internal" / "bbknn" / "bbknn_eval" / "data.csv"
EGFR_CSV    = BLOB / "internal" / "bbknn" / "egfr_eval" / "data.csv"
NATPROD_CSV = BLOB / "internal" / "bbknn" / "natural_product_eval" / "data.csv"

# data-analysis scratch + figure paths (mirrors other modules)
BASE     = BLOB / "internal" / "data_analysis" / "bbknn"
FIG_DIR  = BLOB / "internal" / "figures" / "bbknn"
RAW_DIR  = FIG_DIR / "raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

ENAMINE_CLASS = "Enamine Validation"
D4_CLASS      = "D4"
EGFR_CLASS    = "EGFR"
