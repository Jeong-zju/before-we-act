#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S0_RUN_ROOT:?set S0_RUN_ROOT}"
: "${S0_READY_FILE:?set S0_READY_FILE}"
: "${S0_FAILED_FILE:?set S0_FAILED_FILE}"
: "${S0_HF_TOKEN_FIFO:?set S0_HF_TOKEN_FIFO}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"
: "${ROBOFACTORY_ROOT:?set ROBOFACTORY_ROOT}"
: "${RF_PYTHON:?set RF_PYTHON}"
S0_SHARED_PREPARE_LOCK="${S0_SHARED_PREPARE_LOCK:-${FE_ROOT}/outputs/s0_runs/.shared_prepare.lock}"

if [[ ! -p "${S0_HF_TOKEN_FIFO}" ]]; then
  printf >&2 'Missing protected Hugging Face token FIFO: %s\n' \
    "${S0_HF_TOKEN_FIFO}"
  exit 3
fi
HF_TOKEN_INPUT=""
IFS= read -r HF_TOKEN_INPUT <"${S0_HF_TOKEN_FIFO}"
unlink "${S0_HF_TOKEN_FIFO}" 2>/dev/null || true
if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
  printf >&2 'The protected Hugging Face token input was invalid.\n'
  exit 3
fi

LOG_PATH="${S0_RUN_ROOT}/prepare.log"
mkdir -p "${S0_RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_exit() {
  local code=$?
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
  unlink "${S0_HF_TOKEN_FIFO}" 2>/dev/null || true
  if (( code != 0 )); then
    touch "${S0_FAILED_FILE}"
    printf >&2 'S0 shared preparation failed with code %d.\n' "${code}"
  fi
}
trap on_exit EXIT

mkdir -p "$(dirname "${S0_SHARED_PREPARE_LOCK}")"
exec {S0_PREPARE_LOCK_FD}>"${S0_SHARED_PREPARE_LOCK}"
printf 'Waiting for the cross-run S0 shared preparation lock: %s\n' \
  "${S0_SHARED_PREPARE_LOCK}"
flock -x "${S0_PREPARE_LOCK_FD}"
printf 'Acquired the cross-run S0 shared preparation lock.\n'

printf 'Preparing the shared S0 environment from %s\n' "${FE_ROOT}"
unset \
  M2_DATA_ROOT \
  HF_M2_DATASET_REPO \
  HF_M2_DATASET_REVISION \
  LIFT_DATASET_REPO \
  LIFT_DATASET_REVISION \
  LPD_DATASET_REPO \
  LPD_DATASET_REVISION \
  ROBOFACTORY_REPO_URL \
  ROBOFACTORY_COMMIT_SHA \
  ROBOFACTORY_ASSET_REVISION
HF_TOKEN="${HF_TOKEN_INPUT}" \
GPU_INDEX="${GPU_INDEX:-0}" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}" \
RF_PYTHON="${RF_PYTHON}" \
  "${FE_ROOT}/scripts/run_lpd_single_5090.sh" prepare
HF_TOKEN_INPUT=""
unset HF_TOKEN_INPUT

sha256sum \
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json" \
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/config.json" \
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors" \
  "${ROBOFACTORY_ROOT}/robofactory/assets/scenes/table/table.glb" \
  | tee "${S0_RUN_ROOT}/shared_artifact_sha256.txt"
touch "${S0_READY_FILE}"
printf 'S0 shared environment is ready. Candidates may start.\n'
