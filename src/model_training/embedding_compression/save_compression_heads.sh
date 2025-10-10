#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Paths & names
# ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # …/model_training/embedding_compression
PROJECT_ROOT="${SCRIPT_DIR}/../.."                              # top-level src/
BLOB_STORE="${PROJECT_ROOT}/blob_store"

IMAGE="embedding_compression"
DOCKERFILE="${SCRIPT_DIR}/embedding_compression.Dockerfile"

echo "🗃️  Blob-store bind-mount : ${BLOB_STORE}"
echo "🐳  Dockerfile            : ${DOCKERFILE}"
echo "📦  Image name            : ${IMAGE}"

# ────────────────────────────────────────────────────────────────
# Build / refresh image (uses Docker cache)
# ────────────────────────────────────────────────────────────────
docker build -f "${DOCKERFILE}" \
             -t "${IMAGE}" \
             "${SCRIPT_DIR}"

# ────────────────────────────────────────────────────────────────
# Run dataset embedding on **all** visible GPUs
# ────────────────────────────────────────────────────────────────
docker run --rm \
  --gpus all \
  -v "${BLOB_STORE}":/code/blob_store \
  "${IMAGE}" \
  python -m src.save_compression_heads
