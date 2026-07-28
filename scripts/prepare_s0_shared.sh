#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S0_RUN_ROOT:?set S0_RUN_ROOT}"
: "${S0_READY_FILE:?set S0_READY_FILE}"
: "${S0_FAILED_FILE:?set S0_FAILED_FILE}"

LOG_PATH="${S0_RUN_ROOT}/prepare.log"
mkdir -p "${S0_RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_exit() {
  local code=$?
  if (( code != 0 )); then
    touch "${S0_FAILED_FILE}"
    printf >&2 'S0 shared preparation failed with code %d.\n' "${code}"
  fi
}
trap on_exit EXIT

export GPU_INDEX="${GPU_INDEX:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FE_ROOT}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${FE_ROOT}/.venv}"
printf 'Preparing the shared S0 environment from %s\n' "${FE_ROOT}"
"${FE_ROOT}/scripts/run_lpd_single_5090.sh" prepare

sha256sum \
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json" \
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/config.json" \
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors" \
  | tee "${S0_RUN_ROOT}/shared_artifact_sha256.txt"
touch "${S0_READY_FILE}"
printf 'S0 shared environment is ready. Candidates may start.\n'
