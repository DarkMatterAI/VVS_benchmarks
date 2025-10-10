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
SWEEP_FILE="${SCRIPT_DIR}/sweep_space1.yaml"

echo "🗃️  Blob-store bind-mount : ${BLOB_STORE}"
echo "🐳  Dockerfile            : ${DOCKERFILE}"
echo "📦  Image name            : ${IMAGE}"
echo "📑  Sweep spec            : ${SWEEP_FILE}"

# ────────────────────────────────────────────────────────────────
# Build / refresh image (Docker cache keeps this quick)
# ────────────────────────────────────────────────────────────────
docker build -f "${DOCKERFILE}" \
             -t "${IMAGE}" \
             "${SCRIPT_DIR}"

# ────────────────────────────────────────────────────────────────
# Iterate over sweep_space.yaml and launch one container per run
# ────────────────────────────────────────────────────────────────
mapfile -t RUN_NAMES < <(yq '.runs.[].name' "${SWEEP_FILE}")
echo "🏃  Found ${#RUN_NAMES[@]} run(s): ${RUN_NAMES[*]}"

for RUN in "${RUN_NAMES[@]}"; do
  echo -e "\n╭──────────────────────────────────────────────────────────────╮"
  echo   "│  🚀  Starting run: ${RUN}"
  echo   "╰──────────────────────────────────────────────────────────────╯"

  # extract arg list for this run
  mapfile -t PARAMS < <(
    yq ".runs[] | select(.name == \"${RUN}\") | .args[]" "${SWEEP_FILE}"
  )

  docker run --rm \
    --gpus '"device=0"' \
    --shm-size 32g \
    --memory 48g --memory-swap 0 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${BLOB_STORE}":/code/blob_store \
    "${IMAGE}" \
    python -m src.train --run_name "${RUN}" "${PARAMS[@]}"

done