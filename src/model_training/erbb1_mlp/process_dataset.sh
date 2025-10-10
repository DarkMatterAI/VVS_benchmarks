#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Locations
# ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # model_training/erbb1_mlp
PROJECT_ROOT="${SCRIPT_DIR}/../.."                              # back to top-level src/
BLOB_STORE="${PROJECT_ROOT}/blob_store"

IMAGE="erbb1_mlp"
DOCKERFILE="${SCRIPT_DIR}/erbb1_mlp.Dockerfile"

echo "🗃️  Blob-store bind-mount : ${BLOB_STORE}"
echo "🐳  Dockerfile            : ${DOCKERFILE}"
echo "📦  Image name            : ${IMAGE}"

# ────────────────────────────────────────────────────────────────
# Build (uses cache; quick if nothing changed)
# ────────────────────────────────────────────────────────────────
docker build -f "${DOCKERFILE}" \
             -t "${IMAGE}" \
             "${SCRIPT_DIR}"

# ────────────────────────────────────────────────────────────────
# Run dataset preprocessing on GPU 0
# ────────────────────────────────────────────────────────────────
docker run --rm \
  --gpus '"device=0"' \
  --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${BLOB_STORE}":/code/blob_store \
  "${IMAGE}" \
  python -m src.process_dataset
