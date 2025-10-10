#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── path helpers ──────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"           # repo root
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"       # compose stack
BLOB_STORE="${PROJECT_ROOT}/blob_store"               # host data dir

export BLOB_STORE                                     # for compose env-sub

# ─────────────── which benchmark to run? ─────────────────────────────
TARGET="${1:-lr_sweep}"   # default if no arg
shift || true             # drop first arg (if present)

case "$TARGET" in
  lr_sweep)          PY_MOD="vvs_local.benchmarks.lr_sweep.run"              ;;
  hyperparam_sweep)  PY_MOD="vvs_local.benchmarks.hyperparam_sweeps.run"     ;;
  generate_indices)  PY_MOD="vvs_local.benchmarks.build_indices.run"         ;;
  build_d4_index)    PY_MOD="vvs_local.benchmarks.build_d4_index.run"        ;;
  compression_eval)  PY_MOD="vvs_local.benchmarks.compression_eval.run"      ;;
  *)                 echo "Unknown target '$TARGET'"; exit 1 ;;
esac


echo "🗂  Blob-store : ${BLOB_STORE}"
echo "📦  Compose   : ${COMPOSE_FILE}"
echo "🚀  Python mod: ${PY_MOD}"
echo "📝  Extra args: $*"

# ─────────────── build & run via docker-compose ──────────────────────
docker compose -f "$COMPOSE_FILE" build vvs_local_bench

docker compose \
  -f "$COMPOSE_FILE" \
  run --rm \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -v "${BLOB_STORE}:/code/blob_store" \
  vvs_local_bench \
  python -m "${PY_MOD}" "$@"

