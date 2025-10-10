#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Paths & names
# ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # .../model_training/erbb1_mlp
PROJECT_ROOT="${SCRIPT_DIR}/../.."                              # top-level src/
BLOB_STORE="${PROJECT_ROOT}/blob_store"

IMAGE="erbb1_mlp"
DOCKERFILE="${SCRIPT_DIR}/erbb1_mlp.Dockerfile"

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
# Run training on GPU 0
# ────────────────────────────────────────────────────────────────
docker run --rm \
  --gpus '"device=0"' \
  --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${BLOB_STORE}":/code/blob_store \
  "${IMAGE}" \
  python -m src.train
