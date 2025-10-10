#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Usage
#   ./run.sh debug     # uses debug_space.yaml
#   ./run.sh sweep     # uses sweep_space.yaml
#   ./run.sh final     # uses final_space.yaml
# ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

TYPE=${1:-debug}
if [[ ! $TYPE =~ ^(debug|sweep|final)$ ]]; then
  echo "Usage: $0 {debug|sweep|final}"
  exit 1
fi

CFG_FILE="${SCRIPT_DIR}/${TYPE}_space.yaml"
if [[ ! -f "$CFG_FILE" ]]; then
  echo "YAML file $CFG_FILE not found"
  exit 1
fi

echo "🗂  YAML config : $CFG_FILE"
echo "🐳  Compose file: $COMPOSE_FILE"

# ────────────────────────────────────────────────────────────────
# Build (leverages Docker-cache)
# ────────────────────────────────────────────────────────────────
docker compose -f "$COMPOSE_FILE" build rad

# ────────────────────────────────────────────────────────────────
# Run benchmark container
# ────────────────────────────────────────────────────────────────
docker compose -f "$COMPOSE_FILE" run --rm \
  -e RUN_TYPE="$TYPE" \
  -e YAML_CFG="$(basename "$CFG_FILE")" \
  rad \
  python -m src.rad_runner \
         --cfg "/code/$(basename "$CFG_FILE")" \
         --run_type "$TYPE"

