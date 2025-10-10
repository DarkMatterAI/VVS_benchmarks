#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_ANALYSIS_DIR="$( realpath "${SCRIPT_DIR}/../.." )"
PROJECT_ROOT="$( dirname "${DATA_ANALYSIS_DIR}" )"
DOCKERFILE="${DATA_ANALYSIS_DIR}/data_analysis.Dockerfile"
BLOB_STORE="${PROJECT_ROOT}/blob_store"

# ── choose analysis mode ─────────────────────────────────────────────
MODE="sweep"          # default = hyper-param sweep
if [[ $# -gt 0 && ( "$1" == "final" || "$1" == "sweep" ) ]]; then
  MODE="$1"; shift
fi

PY_MOD="src.benchmarks.best_hyperparams"
[[ "$MODE" == "final" ]] && PY_MOD="src.benchmarks.final"

echo "🔧  Mode : $MODE  →  ${PY_MOD}"
echo "🗂  Blob : ${BLOB_STORE}"

docker build -t da_bench -f "${DOCKERFILE}" "${DATA_ANALYSIS_DIR}"

docker run --rm \
  -v "${BLOB_STORE}":/code/blob_store \
  da_bench \
  python -m "${PY_MOD}" "$@"

