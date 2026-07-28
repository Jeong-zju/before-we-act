#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S1_R1_RUN_ROOT:?set S1_R1_RUN_ROOT}"
: "${S1_R1_READY_FILE:?set S1_R1_READY_FILE}"
: "${S1_R1_FAILED_FILE:?set S1_R1_FAILED_FILE}"
: "${S1_R1_HF_TOKEN_FIFO:?set S1_R1_HF_TOKEN_FIFO}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"
: "${ROBOFACTORY_ROOT:?set ROBOFACTORY_ROOT}"
: "${RF_PYTHON:?set RF_PYTHON}"
SHARED_LOCK="${S1_R1_SHARED_PREPARE_LOCK:-${FE_ROOT}/outputs/s1_r1_runs/.shared_prepare.lock}"

if [[ ! -p "${S1_R1_HF_TOKEN_FIFO}" ]]; then
  printf >&2 'Missing protected Hugging Face token FIFO: %s\n' \
    "${S1_R1_HF_TOKEN_FIFO}"
  exit 3
fi
HF_TOKEN_INPUT=""
IFS= read -r HF_TOKEN_INPUT <"${S1_R1_HF_TOKEN_FIFO}"
unlink "${S1_R1_HF_TOKEN_FIFO}" 2>/dev/null || true
if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
  printf >&2 'The protected Hugging Face token input was invalid.\n'
  exit 3
fi

LOG_PATH="${S1_R1_RUN_ROOT}/prepare.log"
mkdir -p "${S1_R1_RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_exit() {
  local code=$?
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
  unlink "${S1_R1_HF_TOKEN_FIFO}" 2>/dev/null || true
  if (( code != 0 )); then
    touch "${S1_R1_FAILED_FILE}"
    printf >&2 'S1-R1 shared preparation failed with code %d.\n' "${code}"
  fi
}
trap on_exit EXIT

mkdir -p "$(dirname "${SHARED_LOCK}")"
exec {S1_R1_PREPARE_LOCK_FD}>"${SHARED_LOCK}"
printf 'Waiting for shared S1-R1 preparation lock: %s\n' "${SHARED_LOCK}"
flock -x "${S1_R1_PREPARE_LOCK_FD}"

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
  | tee "${S1_R1_RUN_ROOT}/shared_artifact_sha256.txt"
touch "${S1_R1_READY_FILE}"
printf 'S1-R1 shared environment is ready for F0 and F1.\n'
