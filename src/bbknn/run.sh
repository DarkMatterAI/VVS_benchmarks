#!/usr/bin/env bash
set -euo pipefail

# ─────────────── usage ───────────────
if [[ $# -ne 1 || ! "$1" =~ ^(bbknn|natprod|egfr)$ ]]; then
  echo "Usage: $0 {bbknn|natprod|egfr}"
  exit 1
fi
MODE="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

echo "🟦 Using compose file : ${COMPOSE_FILE}"
echo "🚀 Benchmark mode     : ${MODE}"

# ── build / pull image (idempotent) ───────────────────────────────
docker compose -f "${COMPOSE_FILE}" build

# ── run the chosen benchmark inside the composed service ─────────
docker compose -f "${COMPOSE_FILE}" run --rm \
  bbknn_bench \
  python -m "bbknn.${MODE}_eval"
