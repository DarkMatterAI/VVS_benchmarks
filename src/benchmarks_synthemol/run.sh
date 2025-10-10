#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Usage
#   ./run.sh debug     # takes names & params from DEBUG_SPACE.yaml
#   ./run.sh sweep     # takes names & params from SWEEP_SPACE.yaml
#   ./run.sh final     # takes names & params from FINAL_SPACE.yaml
# ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

TYPE=${1:-debug}
if [[ ! $TYPE =~ ^(debug|sweep|final)$ ]]; then
  echo "Usage: $0 {debug|sweep|final}"; exit 1
fi

# CFG_FILE="${SCRIPT_DIR^^}/${TYPE^^}_SPACE.yaml"    # DEBUG_SPACE.yaml etc.
CFG_FILE="${SCRIPT_DIR}/${TYPE}_space.yaml"

if [[ ! -f $CFG_FILE ]]; then
  echo "YAML file $CFG_FILE not found"; exit 1
fi

echo "🗂  using $CFG_FILE"

# build image (uses cache)
docker compose -f "$COMPOSE_FILE" build synthemol_bench

# run container & hand it the YAML path
docker compose -f "$COMPOSE_FILE" run --rm \
  -e RUN_TYPE=$TYPE \
  -e YAML_CFG=$(basename "$CFG_FILE") \
  synthemol_bench \
  python -m run_benchmark --cfg "/code/$(basename "$CFG_FILE")"
