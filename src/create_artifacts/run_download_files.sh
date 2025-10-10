#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="artifact_downloader"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/.."        # one level up
BLOB_STORE_PATH="${PROJECT_ROOT}/blob_store"

build_image() {
  echo "🔨  Building image '${IMAGE_NAME}' …"
  docker build -f "${SCRIPT_DIR}/download_files.Dockerfile" \
               -t "${IMAGE_NAME}" \
               "${SCRIPT_DIR}"
}

run_container() {
  local entrypoint="$1"    # e.g. src_download_files  or  src_processing
  echo "🚀  Running ${IMAGE_NAME} - ${entrypoint}"
  docker run --rm \
    -v "${BLOB_STORE_PATH}":/code/blob_store \
    "${IMAGE_NAME}" \
    python -m "${entrypoint}"
}

# ────────────────────────────────────────────────────────────────
echo "🗃️  Blob-store mount: ${BLOB_STORE_PATH}"
build_image

# 1) download raw files
run_container "src_download_files"

# 2) light processing / re-organisation
run_container "src_processing"

# 3) building training datasets
run_container "src_create_datasets"

echo "✅  All stages finished"
