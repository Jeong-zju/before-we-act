#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S2_R4_RUN_ROOT:?set S2_R4_RUN_ROOT}"
: "${S2_R4_READY_FILE:?set S2_R4_READY_FILE}"
: "${S2_R4_FAILED_FILE:?set S2_R4_FAILED_FILE}"
: "${S2_R4_HF_TOKEN_FIFO:?set S2_R4_HF_TOKEN_FIFO}"
: "${S2_R4_P0_CONFIG:?set S2_R4_P0_CONFIG}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"

STATUS_TOOL="${FE_ROOT}/scripts/s2_r4_runtime.py"
LOG_PATH="${S2_R4_RUN_ROOT}/prepare.log"
R3_READY_FILE="${S2_R4_RUN_ROOT}/r3_shared.ready"
R3_FAILED_FILE="${S2_R4_RUN_ROOT}/r3_shared.failed"
R3_PARENT_TARGET="${FE_ROOT}/artifacts/s2_r3_w1/predictor.pt"
R3_PARENT_SOURCE="${S2_R4_R3_W1_CHECKPOINT:-}"
R3_CONFIG="${FE_ROOT}/configs/wam_flow/s2_r3_local_future.yaml"
R3_RESUME="${FE_ROOT}/artifacts/s2_r3_w1/recovery/resume.pt"
R3_PROGRESS="${S2_R4_RUN_ROOT}/r3_recovery_progress.jsonl"
R3_STAGES="${S2_R4_RUN_ROOT}/r3_recovery_stages.jsonl"
HEARTBEAT_PID=""

mkdir -p "${S2_R4_RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

status() {
  python3 "${STATUS_TOOL}" shared-status \
    --run-root "${S2_R4_RUN_ROOT}" \
    --phase "$1" \
    --program "$2" \
    --detail "${3:-}"
}

heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat \
      --run-root "${S2_R4_RUN_ROOT}" \
      --shared || true
    sleep 20
  done
}

on_exit() {
  local code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  unlink "${S2_R4_HF_TOKEN_FIFO}" 2>/dev/null || true
  if (( code != 0 )); then
    touch "${S2_R4_FAILED_FILE}"
    status failed prepare_s2_r4_shared.sh \
      "shared preparation failed with code ${code}; inspect ${LOG_PATH}" || true
  fi
}
trap on_exit EXIT

heartbeat_loop &
HEARTBEAT_PID=$!

status r3_shared_prepare prepare_s2_r3_shared.sh \
  "S0-style five-task/DINO download, Flow recovery and R3 PCA preparation"
S2_R3_RUN_ROOT="${S2_R4_RUN_ROOT}" \
S2_R3_READY_FILE="${R3_READY_FILE}" \
S2_R3_FAILED_FILE="${R3_FAILED_FILE}" \
S2_R3_HF_TOKEN_FIFO="${S2_R4_HF_TOKEN_FIFO}" \
S2_R3_W0_CONFIG="${S2_R4_P0_CONFIG}" \
S2_R3_FLOW_CHECKPOINT="${S2_R4_FLOW_CHECKPOINT:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
  bash "${FE_ROOT}/scripts/prepare_s2_r3_shared.sh"

test -f "${R3_READY_FILE}"
test ! -f "${R3_FAILED_FILE}"

if [[ -n "${R3_PARENT_SOURCE}" && ! -f "${R3_PARENT_SOURCE}" ]]; then
  status r3_parent verify_s2_r3_w1_checkpoint.py \
    "configured R3-W1 checkpoint missing; searching shared outputs"
  R3_PARENT_SOURCE=""
fi
if [[ -z "${R3_PARENT_SOURCE}" && -f "${R3_PARENT_TARGET}" ]]; then
  R3_PARENT_SOURCE="${R3_PARENT_TARGET}"
fi
if [[ -z "${R3_PARENT_SOURCE}" ]]; then
  while IFS= read -r candidate; do
    R3_PARENT_SOURCE="${candidate}"
    break
  done < <(
    find "${FE_ROOT}/outputs/s2_r3_runs" \
      -path '*/candidates/w1/checkpoints/predictor.pt' \
      -type f 2>/dev/null | sort -r
  )
fi

if [[ -z "${R3_PARENT_SOURCE}" || ! -f "${R3_PARENT_SOURCE}" ]]; then
  status r3_parent_recovery train_s2_r3_future_predictor.py \
    "accepted R3-W1 checkpoint absent; rebuilding frozen W1 recipe 0/10000 on GPU0"
  mkdir -p "$(dirname "${R3_PARENT_TARGET}")" "$(dirname "${R3_RESUME}")"
  (
    cd "${FE_ROOT}"
    CUDA_VISIBLE_DEVICES=0 \
    LPD_STAGE_LOG="${R3_STAGES}" \
    PYTHONUNBUFFERED=1 \
      uv run --frozen python scripts/train_s2_r3_future_predictor.py \
        --config "${R3_CONFIG}" \
        --device cuda:0 \
        --updates 10000 \
        --output "${R3_PARENT_TARGET}" \
        --resume "${R3_RESUME}" \
        --progress-log "${R3_PROGRESS}"
  )
  R3_PARENT_SOURCE="${R3_PARENT_TARGET}"
fi

R3_PARENT_SOURCE="$(realpath "${R3_PARENT_SOURCE}")"
mkdir -p "$(dirname "${R3_PARENT_TARGET}")"
if [[ -L "${R3_PARENT_TARGET}" && ! -e "${R3_PARENT_TARGET}" ]]; then
  unlink "${R3_PARENT_TARGET}"
fi
if [[ ! -e "${R3_PARENT_TARGET}" ]]; then
  ln -s "${R3_PARENT_SOURCE}" "${R3_PARENT_TARGET}"
elif [[ "$(realpath "${R3_PARENT_TARGET}")" != "${R3_PARENT_SOURCE}" ]]; then
  printf >&2 \
    'Existing R3-W1 target resolves to a different file: %s\n' \
    "${R3_PARENT_TARGET}"
  exit 3
fi

status r3_parent verify_s2_r3_w1_checkpoint.py \
  "verifying R3-W1 method, 10k update, PCA and frozen Flow identities"
(
  cd "${FE_ROOT}"
  uv run --frozen python scripts/verify_s2_r3_w1_checkpoint.py \
    "${R3_PARENT_TARGET}" \
    --config "${R3_CONFIG}"
)

sha256sum "${R3_PARENT_TARGET}" \
  | tee -a "${S2_R4_RUN_ROOT}/shared_artifact_sha256.txt"
touch "${S2_R4_READY_FILE}"
status complete prepare_s2_r4_shared.sh \
  "shared data/DINO/PCA/Flow and verified R3-W1 parent are ready"
printf 'S2-R4 shared preparation complete.\n'
