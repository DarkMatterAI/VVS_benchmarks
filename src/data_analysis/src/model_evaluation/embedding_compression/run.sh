#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── locate project tree ──────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"      # …/embedding_compression
DATA_ANALYSIS_DIR="$( realpath "${SCRIPT_DIR}/../../.." )"          # …/data_analysis
PROJECT_ROOT="$( dirname "${DATA_ANALYSIS_DIR}" )"                  # repo root
DOCKERFILE="${DATA_ANALYSIS_DIR}/data_analysis.Dockerfile"
BLOB_STORE="${PROJECT_ROOT}/blob_store"

echo "🗂  Project root   : ${PROJECT_ROOT}"
echo "📦  Dockerfile     : ${DOCKERFILE}"
echo "🗃  Blob-store dir : ${BLOB_STORE}"

# ───────────────────────── argument parsing ─────────────────────────────
MODE="generate"      # default action
if [[ $# -gt 0 && ( "$1" == "plot" || "$1" == "generate" ) ]]; then
  MODE="$1";         # first arg is the mode
  shift              # drop it so remaining flags pass to python module
fi

PY_MODULE="src.model_evaluation.embedding_compression.generate_data"
[[ "$MODE" == "plot" ]] && PY_MODULE="src.model_evaluation.embedding_compression.plot_results"

echo "🔧  Mode selected : ${MODE}  (module: ${PY_MODULE})"

# ───────────────────────── Docker build / run ───────────────────────────
docker build -t da_eval -f "${DOCKERFILE}" "${DATA_ANALYSIS_DIR}"

docker run --rm \
  --gpus '"device=0"' \
  --shm-size 8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${BLOB_STORE}":/code/blob_store \
  da_eval \
  python -m "${PY_MODULE}" "$@"

