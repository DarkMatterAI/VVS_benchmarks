#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-demo.yml"
DEMO_BLOB="/code/blob_store/demo"

echo "🟦 Building image"
docker compose -f "${COMPOSE_FILE}" build

echo "🧪 Preparing demo artifacts (embeddings + reactions)"
docker compose -f "${COMPOSE_FILE}" run --rm \
  -e BLOB_STORE="${DEMO_BLOB}" \
  bbknn_demo \
  python /code/bbknn/demo/prepare_demo.py

echo "🚀 Running BBKNN demo"
docker compose -f "${COMPOSE_FILE}" run --rm \
  -e BLOB_STORE="${DEMO_BLOB}" \
  bbknn_demo \
  python -m bbknn.demo.run_demo

echo "✅ Done. Results in ./blob_store/demo/results/demo_results.csv"