#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

TYPE=${1:-debug}            # debug | sweep | final
[[ $TYPE =~ ^(debug|sweep|final)$ ]] || { echo "usage: $0 [debug|sweep|final]"; exit 1; }
CFG_FILE="${SCRIPT_DIR}/${TYPE}_space.yaml"

echo "🗂  YAML config : $CFG_FILE"
docker compose -f "$COMPOSE_FILE" build rxnflow_bench
docker compose -f "$COMPOSE_FILE" run --rm \
  -e YAML_CFG=$(basename "$CFG_FILE") \
  -e RUN_TYPE=$TYPE \
  rxnflow_bench \
  python -m src.rxnflow_runner --cfg "/code/$(basename "$CFG_FILE")" --run_type "$TYPE"
