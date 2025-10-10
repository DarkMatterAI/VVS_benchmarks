from pathlib import Path
import os

# ─── blob-store root ──────────────────────────────────────────────────────────
BLOB = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()

# ─── data locations ───────────────────────────────────────────────────────────
DATA_DIR  = BLOB / "internal" / "data_analysis" / "vvs_local" / "lr_sweep"
BENCH_DIR = BLOB / "internal" / "benchmarks" / "vvs_local" / "sweep"
ENGINE    = "bbknn"          # folder name inside DATA_DIR
INDEX_PATH = (BLOB / "internal" / "processed" / "vvs_local"
                       / "zinc_10m_indices").resolve()

# ─── figure / raw-csv output ──────────────────────────────────────────────────
FIG_DIR = BLOB / "internal" / "figures" / "vvs_local"
RAW_DIR = FIG_DIR / "raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "0C5DA5",
    "green": "00B945",
    "red": "FF2C00"
}
COLORS = {k:f"#{v.lower()}" for k,v in COLORS.items()}