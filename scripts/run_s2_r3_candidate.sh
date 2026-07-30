#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FE_ROOT}/experiments/wam_flow/s2_r3/candidate.env"
: "${S2_R3_RUN_ROOT:?set S2_R3_RUN_ROOT}"
: "${S2_R3_READY_FILE:?set S2_R3_READY_FILE}"
: "${S2_R3_FAILED_FILE:?set S2_R3_FAILED_FILE}"
: "${S2_R3_BASE_REPO:?set S2_R3_BASE_REPO}"
: "${S2_R3_RUN_ID:?set S2_R3_RUN_ID}"
: "${S2_R3_UV_CACHE_DIR:?set S2_R3_UV_CACHE_DIR}"
: "${S2_R3_UV_ENV:?set S2_R3_UV_ENV}"
: "${GPU_INDEX:?set GPU_INDEX}"
test -f "${ENV_FILE}"
# shellcheck source=/dev/null
source "${ENV_FILE}"
: "${S2_R3_CANDIDATE_ID:?candidate.env must set S2_R3_CANDIDATE_ID}"
: "${S2_R3_TOTAL_UPDATES:?candidate.env must set S2_R3_TOTAL_UPDATES}"
: "${S2_R3_CONFIG_REL:?candidate.env must set S2_R3_CONFIG_REL}"

SLUG="$(printf '%s' "${S2_R3_CANDIDATE_ID}" | tr '[:upper:]' '[:lower:]')"
CANDIDATE_ROOT="${S2_R3_RUN_ROOT}/candidates/${SLUG}"
STATUS_TOOL="${S2_R3_BASE_REPO}/scripts/s2_r3_runtime.py"
CONFIG="${FE_ROOT}/${S2_R3_CONFIG_REL}"
CHECKPOINT="${CANDIDATE_ROOT}/checkpoints/predictor.pt"
RESUME="${CANDIDATE_ROOT}/checkpoints/resume.pt"
EVALUATION="${CANDIDATE_ROOT}/validation/evaluation.json"
TRAIN_PROGRESS="${CANDIDATE_ROOT}/train/progress.jsonl"
VALIDATION_PROGRESS="${CANDIDATE_ROOT}/validation/progress.jsonl"
STAGE_LOG="${CANDIDATE_ROOT}/train/stages.jsonl"
LOG_PATH="${CANDIDATE_ROOT}/logs/candidate.log"
WAIT_HEARTBEAT_SECONDS="${S2_R3_WAIT_HEARTBEAT_SECONDS:-30}"
HEARTBEAT_PID=""
COMPLETED=0

mkdir -p \
  "${CANDIDATE_ROOT}/logs" \
  "${CANDIDATE_ROOT}/train" \
  "${CANDIDATE_ROOT}/validation" \
  "${CANDIDATE_ROOT}/checkpoints" \
  "${CANDIDATE_ROOT}/outputs" \
  "${CANDIDATE_ROOT}/workspace_logs"
exec > >(tee -a "${LOG_PATH}") 2>&1

status() {
  local arguments=(
    status
    --run-root "${S2_R3_RUN_ROOT}"
    --candidate "${S2_R3_CANDIDATE_ID}"
    --phase "$1"
    --program "$2"
    --detail "${3:-}"
    --gpu-index "${GPU_INDEX}"
    --total-updates "${S2_R3_TOTAL_UPDATES}"
  )
  if (( $# >= 4 )); then
    arguments+=(--exit-code "$4")
  fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}

heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat \
      --run-root "${S2_R3_RUN_ROOT}" \
      --candidate "${S2_R3_CANDIDATE_ID}" || true
    sleep 20
  done
}

on_exit() {
  local code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  if (( code != 0 )) && (( COMPLETED == 0 )); then
    status failed run_s2_r3_candidate.sh \
      "candidate exited with code ${code}; see ${LOG_PATH}" "${code}" || true
  fi
}
trap on_exit EXIT
heartbeat_loop &
HEARTBEAT_PID=$!

WAIT_STARTED="${SECONDS}"
NEXT_WAIT_NOTICE=0
while [[ ! -f "${S2_R3_READY_FILE}" ]]; do
  if [[ -f "${S2_R3_FAILED_FILE}" ]]; then
    printf >&2 'S2-R3 shared preparation failed; inspect prepare.log.\n'
    exit 4
  fi
  elapsed="$((SECONDS - WAIT_STARTED))"
  if (( elapsed >= NEXT_WAIT_NOTICE )); then
    status waiting run_s2_r3_candidate.sh \
      "waiting for five datasets, DINO, PCA and reused/recovered Flow; elapsed=${elapsed}s"
    NEXT_WAIT_NOTICE="$((elapsed + WAIT_HEARTBEAT_SECONDS))"
  fi
  sleep 5
done

link_shared() {
  local target="$1"
  local source="$2"
  if [[ -L "${target}" ]]; then
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      printf >&2 'Mismatched shared link: %s\n' "${target}"
      exit 3
    }
  elif [[ -e "${target}" ]]; then
    printf >&2 'Refusing existing candidate path: %s\n' "${target}"
    exit 3
  else
    ln -s "${source}" "${target}"
  fi
}

status setup run_s2_r3_candidate.sh \
  "linking read-only shared datasets/artifacts and isolated outputs"
link_shared "${FE_ROOT}/datasets" "${S2_R3_BASE_REPO}/datasets"
link_shared "${FE_ROOT}/artifacts" "${S2_R3_BASE_REPO}/artifacts"
test -f "${CONFIG}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export UV_CACHE_DIR="${S2_R3_UV_CACHE_DIR}"
export UV_PROJECT_ENVIRONMENT="${S2_R3_UV_ENV}"
export LPD_STAGE_LOG="${STAGE_LOG}"
unset HF_TOKEN || true

status training train_s2_r3_future_predictor.py \
  "five-task joint training; candidate=${S2_R3_CANDIDATE_ID}"
(
  cd "${FE_ROOT}"
  PYTHONUNBUFFERED=1 uv run --frozen python \
    scripts/train_s2_r3_future_predictor.py \
      --config "${CONFIG}" \
      --device cuda:0 \
      --updates "${S2_R3_TOTAL_UPDATES}" \
      --output "${CHECKPOINT}" \
      --resume "${RESUME}" \
      --progress-log "${TRAIN_PROGRESS}"
)
test -f "${CHECKPOINT}"

status validating evaluate_s2_r3_future_predictor.py \
  "five-task held-out loss plus paired own-action shuffle/bootstrap"
(
  cd "${FE_ROOT}"
  PYTHONUNBUFFERED=1 uv run --frozen python \
    scripts/evaluate_s2_r3_future_predictor.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --output "${EVALUATION}" \
      --progress-log "${VALIDATION_PROGRESS}" \
      --device cuda:0
)
test -f "${EVALUATION}"

exec {S2_R3_ACCEPT_LOCK_FD}>"${S2_R3_RUN_ROOT}/.acceptance.lock"
flock -x "${S2_R3_ACCEPT_LOCK_FD}"
W0_EVALUATION="${S2_R3_RUN_ROOT}/candidates/w0/validation/evaluation.json"
W1_EVALUATION="${S2_R3_RUN_ROOT}/candidates/w1/validation/evaluation.json"
if [[ -f "${W0_EVALUATION}" && -f "${W1_EVALUATION}" && \
      ! -f "${S2_R3_RUN_ROOT}/acceptance.json" ]]; then
  status accepting accept_s2_r3.py \
    "applying S2-R3 special five-task capability gate"
  (
    cd "${S2_R3_BASE_REPO}"
    uv run --frozen python scripts/accept_s2_r3.py \
      --w0 "${W0_EVALUATION}" \
      --w1 "${W1_EVALUATION}" \
      --output "${S2_R3_RUN_ROOT}/acceptance.json"
  )
fi

DECISION="pending peer evaluation"
if [[ -f "${S2_R3_RUN_ROOT}/acceptance.json" ]]; then
  DECISION="$(
    jq -r 'if .passed then "PASS: enter R4" else "FAIL: stop before R4" end' \
      "${S2_R3_RUN_ROOT}/acceptance.json"
  )"
fi
status complete run_s2_r3_candidate.sh \
  "training/evaluation complete; ${DECISION}" 0
COMPLETED=1
printf 'S2-R3 %s complete: %s\n' "${S2_R3_CANDIDATE_ID}" "${CANDIDATE_ROOT}"
