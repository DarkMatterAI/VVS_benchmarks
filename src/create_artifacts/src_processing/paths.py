import os
from pathlib import Path

BLOB_STORE   = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
INTERNAL     = BLOB_STORE / "internal"
RAW_DIR      = INTERNAL / "raw_downloads"
PROCESSED_DIR = INTERNAL / "processed"

# Final destinations
OPENEYE_DOCK = PROCESSED_DIR / "openeye" / "docking"
OPENEYE_ROCS = PROCESSED_DIR / "openeye" / "rocs"
SYNTHEMOL_RF = PROCESSED_DIR / "synthemol_rf"

# Processed dataset folders
ZINC_DIR     = PROCESSED_DIR / "zinc_10m"
D4_DIR       = PROCESSED_DIR / "d4_138m"
ENAMINE_DIR  = PROCESSED_DIR / "enamine"
CHEMBL_DIR   = PROCESSED_DIR / "chembl_erbb1_ic50"
NATPROD_DIR  = PROCESSED_DIR / "natural_products"

# Raw sources
TS_REPO_DATA  = RAW_DIR / "TS" / "data"
MODELS_ZIP    = RAW_DIR / "Models.zip"
DATA_ZIP      = RAW_DIR / "Data.zip"
NATPROD_ZIP   = RAW_DIR / "natural_products.zip"
ZINC_TAR_GZ   = RAW_DIR / "zinc15_10M_2D.tar.gz"
D4_GZ         = RAW_DIR / "D4_screen_table.csv.gz"
PROVIDED_6LUD = BLOB_STORE / "provided_files" / "6lud_docking" / "6lud.oedu"
CHEMBL_CSV   = RAW_DIR / "chembl_erbb1_ic50.csv"
