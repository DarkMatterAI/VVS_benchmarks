"""
Train the ErbB1-MLP using Hugging-Face Trainer.
Run inside the erbb1_mlp Docker image:

    docker run --gpus '"device=0"' -v $PWD/blob_store:/code/blob_store erbb1_mlp \
        python -m src.train
"""
import json, os
from pathlib import Path

import evaluate
import numpy as np
import torch
from rich.console import Console
from transformers import Trainer, TrainingArguments

import datasets as ds

from .configuration_erbb1_mlp import Erbb1MlpConfig
from .modeling_erbb1_mlp import Erbb1MlpModel

assert torch.cuda.is_available()
console = Console()

# ───── paths ────────────────────────────────────────────────────
BLOB = Path(os.environ.get("BLOB_STORE", "/code/blob_store")).resolve()
PARENT_DIR = BLOB / "internal" / "training_datasets" / "erbb1_mlp"
DATASET_DIR = PARENT_DIR / "erbb1_mlp.hf"
STATS_PATH  = PARENT_DIR / "erbb1_mlp_stats.json"
OUTPUT_DIR  = BLOB / "internal" / "training_runs" / "erbb1_mlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ───── prepare dataset ─────────────────────────────────────────
console.log("📚 loading dataset")
ds_all = ds.load_from_disk(str(DATASET_DIR))
ds_all = ds_all.remove_columns(["external_id", "IC50", "pIC50"])
ds_all = ds_all.rename_column("pIC50_norm", "labels")
ds_all.set_format(type="torch", columns=["embedding", "labels"])

console.log("🔀 splitting")
dataset = ds_all.train_test_split(test_size=0.2, seed=42)

# ───── stats for normalisation ─────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)
mean, std = stats["mean"], stats["std"]

# ───── model & config ──────────────────────────────────────────
config = Erbb1MlpConfig(
    d_in=768,
    d_hidden=1024,
    n_layers=4,
    dropout=0.1,
    dataset_mean=mean,
    dataset_std=std,
)

model = Erbb1MlpModel(config)

# ───── metrics ─────────────────────────────────────────────────
r2 = evaluate.load("r_squared")

def compute_metrics(eval_pred):
    (preds, preds_norm), labels_norm = eval_pred
    preds = torch.tensor(preds_norm) * std + mean
    labels = torch.tensor(labels_norm) * std + mean
    return {
        "mse": float(np.mean((preds_norm - labels_norm) ** 2)),
        "r2":  r2.compute(predictions=preds, references=labels),
    }

# ───── training args ───────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=512,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=8,
    report_to=None,
)

# ───── Trainer ────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

console.rule("[bold]Training")
trainer.train()

console.rule("[bold green]Saving final model")
model.save_pretrained(str(OUTPUT_DIR / "hf_model"))
config.save_pretrained(str(OUTPUT_DIR / "hf_model"))
console.log(f"[green]✓ model saved to {OUTPUT_DIR/'hf_model'}")
