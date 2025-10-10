#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── locate project tree ──────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_ANALYSIS_DIR="$( realpath "${SCRIPT_DIR}/../.." )"
PROJECT_ROOT="$( dirname "${DATA_ANALYSIS_DIR}" )"
DOCKERFILE="${DATA_ANALYSIS_DIR}/data_analysis.Dockerfile"
BLOB_STORE="${PROJECT_ROOT}/blob_store"

echo "🗂  Blob-store  : ${BLOB_STORE}"
echo "📦  Dockerfile : ${DOCKERFILE}"

# ──────────────────────────── select module ─────────────────────────────
# Usage:
#   ./run.sh [cosine|sampling] [additional-python-args…]
#
#   cosine    →  src.bbknn.cosine_tanimoto   (default)
#   sampling  →  src.bbknn.sampling_analysis
#   egfr      →  src.bbknn.egfr_analysis
#
MODULE="cosine"   # default
if [[ $# -gt 0 && "$1" =~ ^(cosine|sampling|egfr)$ ]]; then
  MODULE="$1"
  shift            # drop the first CLI arg (module flag)
fi

case "$MODULE" in
  cosine)   PY_MOD="src.bbknn.cosine_tanimoto"   ;;
  sampling) PY_MOD="src.bbknn.sampling_analysis" ;;
  egfr)     PY_MOD="src.bbknn.egfr_analysis"     ;;
esac

echo "🚀  Running Python module  : ${PY_MOD}"

# ───────────────────────── Docker build / run ───────────────────────────
docker build -t da_bbknn -f "${DOCKERFILE}" "${DATA_ANALYSIS_DIR}"

docker run --rm \
  -v "${BLOB_STORE}":/code/blob_store \
  da_bbknn \
  python -m "${PY_MOD}" "$@"
