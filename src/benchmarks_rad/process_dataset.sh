#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── paths & image ─────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BLOB_STORE="${SCRIPT_DIR}/../blob_store"
IMAGE="rad_benchmark"

# ───────────────────────── build image ───────────────────────────
docker build -f "${SCRIPT_DIR}/rad.Dockerfile" -t "${IMAGE}" "${SCRIPT_DIR}"

# ──────────────────────── debug flag parsing ─────────────────────
MODE="full"                      # default → full dataset
if [[ "${1-}" == "debug" ]]; then
  MODE="debug"
  shift                         # remove "debug" from $@ so it’s not forwarded
fi
echo "🗃️  Mode: ${MODE}"

# ───────────────────────── run processor ─────────────────────────
docker run --rm \
  -v "$BLOB_STORE":/code/blob_store \
  --cpus="32" \
  "${IMAGE}" \
  python -m src.process_dataset --cfg /code/rad_data_params.yaml --mode "${MODE}" "$@"

