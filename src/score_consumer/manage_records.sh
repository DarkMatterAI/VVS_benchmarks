#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ACTION="${1:-create}"                           # create | delete

IMAGE="score_consumer_records"
PLUGIN_VOL="${SCRIPT_DIR}"                      # mount local dir
DOCKERFILE="${SCRIPT_DIR}/score_consumer.Dockerfile"

docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${SCRIPT_DIR}"

docker run --rm \
  --network vvs_v2_network \
  --env-file "${SCRIPT_DIR}/.env" \
  -e PLUGIN_MAP_PATH=/plugin_map/plugin_ids.json \
  -v "${PLUGIN_VOL}":/plugin_map \
  "${IMAGE}" \
  python create_records.py "${ACTION}"
