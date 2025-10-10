from pathlib import Path
import os, multiprocessing, pika
from rich.console import Console
console = Console()

# ───────────── base paths ──────────────────────────────────────
BLOB         = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
PROC_PTH     = BLOB / "internal" / "processed"

# ── Enamine building blocks & reactions (already produced earlier)
BB_CSV_PTH   = PROC_PTH / "enamine" / "data.csv"
REACTION_PKL = PROC_PTH / "enamine" / "enamine_id_to_reaction.pkl"

# ── RxnFlow environment root  (will be auto-generated)
ENV_BASE     = PROC_PTH / "rxnflow"
ENV_DIR      = ENV_BASE / "env"          # fp / desc / mask live here
TEMPLATE_TXT = ENV_BASE / "data" / "template.txt"
BB_SMI       = ENV_BASE / "data" / "building_block.smi"

# ───────────── RabbitMQ connection info ───────────────────────
RABBIT = dict(
    host        = os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port        = int(os.getenv("RABBITMQ_PORT", 5672)),
    credentials = pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "rabbitmq_user"),
        os.getenv("RABBITMQ_PASS", "rabbitmq_password"),
    ),
    heartbeat   = 180,
)
EXCHANGE = os.getenv("RABBITMQ_EXCHANGE_NAME")

# ───────────── misc ────────────────────────────────────────────
N_CPU = max(multiprocessing.cpu_count() - 2, 2)
