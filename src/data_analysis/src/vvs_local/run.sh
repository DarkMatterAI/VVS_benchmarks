#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_ANALYSIS_DIR="$( realpath "${SCRIPT_DIR}/../.." )"
PROJECT_ROOT="$( dirname "${DATA_ANALYSIS_DIR}" )"
DOCKERFILE="${DATA_ANALYSIS_DIR}/data_analysis.Dockerfile"
BLOB_STORE="${PROJECT_ROOT}/blob_store"

echo "📦  Dockerfile : ${DOCKERFILE}"
echo "🗂  Blob store  : ${BLOB_STORE}"

docker build -t da_vvs -f "${DOCKERFILE}" "${DATA_ANALYSIS_DIR}"

# ── which plotting module? ────────────────────────────────────────────────
TARGET="${1:-sweep}"          # default is plot_sweep
shift || true

case "$TARGET" in
  sweep) PY_MOD="src.vvs_local.plot_sweep" ;;
  embed) PY_MOD="src.vvs_local.plot_embedding_size" ;;
  *)     echo "Unknown target '$TARGET' (use: sweep | embed)"; exit 1 ;;
esac

docker run --rm \
  -v "${BLOB_STORE}":/code/blob_store \
  da_vvs \
  python -m "${PY_MOD}" "$@"
