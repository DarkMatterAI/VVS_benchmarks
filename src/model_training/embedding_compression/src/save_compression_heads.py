import os
from pathlib import Path 

from .modeling_compression import CompressionHFModel


BLOB          = Path(os.environ["BLOB_STORE"]).resolve()
TRAINING_RUNS = BLOB / "internal" / "training_runs" / "embedding_compression"
TARGET_RUN    = TRAINING_RUNS / "pearson_topk_lr1e3_wd001_4_4" / "checkpoint-9277"
OUTPUT_FILE   = BLOB / "internal" / "model_weights" / "compression_heads.pt"

def main():
    model = CompressionHFModel.from_pretrained(str(TARGET_RUN))
    model.save_encoders(OUTPUT_FILE)


if __name__ == "__main__":
    main()