from pathlib import Path
import os

BLOB = Path(os.getenv("BLOB_STORE", "/code/blob_store")).resolve()

# ---------- I/O ----------
BASE      = BLOB / "internal" / "data_analysis" / "model_evaluation" / "embedding_compression"
RAW_CSV   = BASE / "sample.csv"              # the random D4 sample (parquet is fine too)
TOK_DS    = BASE / "tokenised.hf"            # HF dataset with input_ids / attn_mask
EMB_DS    = BASE / "embeddings.hf"           # HF dataset with all sizes saved as float32
KNN_PTH   = BASE / "knn_indices.pt"          # torch.save({size: LongTensor[B,Q]})
PREC_PTH  = BASE / "precision.pt"            # cut-off curves (dict[str|tuple] -> list[float])
TOPK_DIR  = BASE / "topk"                    # parquet files

# ---------- sampling ----------
QUERY_SZ  = 5_000
REF_SZ    = 50_000
QUERY_DB  = BLOB / "internal" / "processed" / "d4_138m" / "database.db"
TABLE     = "d4_138m"
BATCH     = 100_000              # DuckDB streaming chunk

# ---------- models ----------
EMBED_MODEL    = "entropy/roberta_zinc_480m"
COMPRESS_MODEL = "entropy/roberta_zinc_compression_encoder"

# ---------- eval ----------
SIZES   = (32, 64, 128, 256, 512, 768)
CUTS    = (1,2,3,4,5,6,7,8,9,10,15,25,50,75,100,200,300,400,500)
K_NN    = max(CUTS)                         # 500
BATCH_KNN = 512
