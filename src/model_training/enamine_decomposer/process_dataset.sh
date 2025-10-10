# enamine_decomposer/run_dataset.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BLOB_STORE="${SCRIPT_DIR}/../../blob_store"

IMAGE="enamine_decomposer"
DOCKERFILE="${SCRIPT_DIR}/enamine_decomposer.Dockerfile"

docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${SCRIPT_DIR}"

docker run --rm \
  --gpus all \
  -v "${BLOB_STORE}":/code/blob_store \
  "${IMAGE}" \
  python -m src.process_dataset
