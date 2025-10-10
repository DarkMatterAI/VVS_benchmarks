from pathlib import Path
import os
from rich.console import Console

console = Console()

# ── centralised paths ─────────────────────────────────────────────────
BLOB = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()

BENCH_DIR = BLOB / "internal" / "benchmarks"
METHODS = ["vvs_local", "synthemol", "rxnflow", "ts"]
SWEEPS = {k: BENCH_DIR / k / "sweep" for k in METHODS}
FINALS = {k: BENCH_DIR / k / "final" for k in METHODS}
ENUM_FINALS = {"vvs_local": BENCH_DIR / "vvs_local" / "final_knn",
               "rad": BENCH_DIR / "rad" / "final"}

DATA_DIR  = BLOB / "internal" / "data_analysis" / "benchmarks"
SWEEP_DIR = DATA_DIR / "sweep"
FINAL_DIR = DATA_DIR / "final"
FIG_DIR   = BLOB / "internal" / "figures" / "benchmarks"
RAW_DIR   = FIG_DIR / "raw"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOPKS   = [1, 5, 10, 100, 1000]
RANK_KS = [1, 10, 100]
