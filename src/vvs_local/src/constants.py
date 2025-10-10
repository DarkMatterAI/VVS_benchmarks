from pathlib import Path
import os, torch, pika 
from typing import Dict
from rich.console import Console
console  = Console()

BLOB         = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
BB_EMB_PTH   = BLOB / "internal" / "training_datasets" / "enamine_assembled" / "bb_embeddings.pt"
PROC_PTH     = BLOB / "internal" / "processed"
BB_CSV_PTH   = PROC_PTH / "enamine" / "data.csv"
REACTION_PKL = PROC_PTH / "enamine" / "enamine_id_to_reaction.pkl"
D4_DB_PATH   = PROC_PTH / "d4_138m" / "database.db"
D4_TBL_NAME  = "d4_138m"
ZNC_DB_PATH  = PROC_PTH / "zinc_10m" / "database.db"
ZNC_TBL_NAME = "zinc_10m"
INDEX_PATH   = PROC_PTH / "vvs_local" / "zinc_10m_indices"

DECOMP_MODEL_NM = "entropy/roberta_zinc_enamine_decomposer"
EMB_MODEL_NM    = "entropy/roberta_zinc_480m"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

RABBIT: Dict = dict(
    host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=int(os.getenv("RABBITMQ_PORT", 5672)),
    credentials=pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "rabbitmq_user"),
        os.getenv("RABBITMQ_PASS", "rabbitmq_password"),
    ),
    heartbeat=180
)
EXCHANGE = os.getenv("RABBITMQ_EXCHANGE_NAME")