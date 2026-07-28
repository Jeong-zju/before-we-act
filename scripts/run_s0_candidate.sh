#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FE_ROOT}/experiments/wam_flow/s0/candidate.env"
: "${S0_RUN_ROOT:?set S0_RUN_ROOT}"
: "${S0_READY_FILE:?set S0_READY_FILE}"
: "${S0_FAILED_FILE:?set S0_FAILED_FILE}"
: "${S0_BASE_REPO:?set S0_BASE_REPO}"
: "${GPU_INDEX:?set GPU_INDEX}"
: "${S0_UV_CACHE_DIR:?set S0_UV_CACHE_DIR}"
: "${S0_UV_ENV:?set S0_UV_ENV}"
: "${S0_ROBOFACTORY_ROOT:?set S0_ROBOFACTORY_ROOT}"
: "${S0_RF_PYTHON:?set S0_RF_PYTHON}"
unset HF_TOKEN || true
test -f "${ENV_FILE}"
# Candidate env files contain only fixed, repository-owned scalar assignments.
# shellcheck source=/dev/null
source "${ENV_FILE}"
: "${S0_CANDIDATE_ID:?candidate.env must set S0_CANDIDATE_ID}"
: "${S0_TRAIN_MODE:?candidate.env must set S0_TRAIN_MODE}"
: "${S0_TOTAL_UPDATES:?candidate.env must set S0_TOTAL_UPDATES}"
: "${LPD_POLICY_KIND:?candidate.env must set LPD_POLICY_KIND}"
: "${LPD_CONFIG_REL:?candidate.env must set LPD_CONFIG_REL}"
: "${LPD_CHECKPOINT_REL:?candidate.env must set LPD_CHECKPOINT_REL}"

CANDIDATE_SLUG="$(printf '%s' "${S0_CANDIDATE_ID}" | tr '[:upper:]' '[:lower:]')"
CANDIDATE_ROOT="${S0_RUN_ROOT}/candidates/${CANDIDATE_SLUG}"
STATUS_TOOL="${FE_ROOT}/scripts/s0_runtime.py"
LOG_PATH="${CANDIDATE_ROOT}/logs/candidate.log"
mkdir -p "${CANDIDATE_ROOT}/logs" "${CANDIDATE_ROOT}/train"
exec > >(tee -a "${LOG_PATH}") 2>&1

COMPLETED=0
status() {
  local arguments=(
    status
    --run-root "${S0_RUN_ROOT}"
    --candidate "${S0_CANDIDATE_ID}"
    --phase "$1"
    --detail "${2:-}"
    --gpu-index "${GPU_INDEX}"
    --total-updates "${S0_TOTAL_UPDATES}"
  )
  if (( $# >= 3 )); then
    arguments+=(--exit-code "$3")
  fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}

on_exit() {
  local code=$?
  if (( code != 0 )) && (( COMPLETED == 0 )); then
    status failed "candidate exited with code ${code}; see ${LOG_PATH}" "${code}" || true
  fi
}
trap on_exit EXIT

link_shared() {
  local target="$1"
  local source="$2"
  if [[ -L "${target}" ]]; then
    if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then
      printf >&2 'Refusing mismatched shared link: %s -> %s\n' \
        "${target}" "$(readlink "${target}")"
      exit 3
    fi
    return
  fi
  if [[ -e "${target}" ]]; then
    printf >&2 'Refusing to replace existing candidate path: %s\n' "${target}"
    exit 3
  fi
  ln -s "${source}" "${target}"
}

link_isolated_run_path() {
  local target="$1"
  local source="$2"
  if [[ -L "${target}" ]]; then
    if [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]]; then
      return
    fi
    unlink "${target}"
  elif [[ -e "${target}" ]]; then
    printf >&2 'Refusing to replace existing non-link run path: %s\n' "${target}"
    exit 3
  fi
  ln -s "${source}" "${target}"
}

status waiting "waiting for shared environment, data and DINO artifacts"
while [[ ! -f "${S0_READY_FILE}" ]]; do
  if [[ -f "${S0_FAILED_FILE}" ]]; then
    printf >&2 'Shared S0 preparation failed; inspect the prepare tmux window.\n'
    exit 4
  fi
  sleep 5
done

test -d "${S0_BASE_REPO}/datasets"
test -d "${S0_BASE_REPO}/artifacts"
mkdir -p \
  "${CANDIDATE_ROOT}/checkpoints" \
  "${CANDIDATE_ROOT}/outputs" \
  "${CANDIDATE_ROOT}/workspace_logs"
link_shared "${FE_ROOT}/datasets" "${S0_BASE_REPO}/datasets"
link_shared "${FE_ROOT}/artifacts" "${S0_BASE_REPO}/artifacts"
link_isolated_run_path "${FE_ROOT}/checkpoints" "${CANDIDATE_ROOT}/checkpoints"
link_isolated_run_path "${FE_ROOT}/outputs" "${CANDIDATE_ROOT}/outputs"
link_isolated_run_path "${FE_ROOT}/logs" "${CANDIDATE_ROOT}/workspace_logs"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export UV_CACHE_DIR="${S0_UV_CACHE_DIR}"
export UV_PROJECT_ENVIRONMENT="${S0_UV_ENV}"
export ROBOFACTORY_ROOT="${S0_ROBOFACTORY_ROOT}"
export RF_PYTHON="${S0_RF_PYTHON}"
export LPD_EXPERIMENT_SLUG="s0_${CANDIDATE_SLUG}"
export LPD_CONFIG="${FE_ROOT}/${LPD_CONFIG_REL}"
export LPD_CHECKPOINT="${FE_ROOT}/${LPD_CHECKPOINT_REL}"
export LPD_PROGRESS_LOG="${CANDIDATE_ROOT}/train/progress.jsonl"
export LPD_PORT="$((8872 + GPU_INDEX))"
export LPD_RUN_ID="${S0_RUN_ID:-s0}"
export LPD_EPISODES="${S0_GATE_EPISODES:-20}"
export LPD_SEED_START="${S0_GATE_SEED_START:-900}"
test -f "${LPD_CONFIG}"

case "${S0_TRAIN_MODE}" in
  train)
    status training "training ${LPD_POLICY_KIND} for ${S0_TOTAL_UPDATES} updates"
    "${FE_ROOT}/scripts/run_lpd_single_5090.sh" train
    ;;
  reuse)
    : "${S0_REUSE_CANDIDATE:?reuse mode requires S0_REUSE_CANDIDATE}"
    : "${S0_REUSE_CHECKPOINT_REL:?reuse mode requires S0_REUSE_CHECKPOINT_REL}"
    reuse_slug="$(printf '%s' "${S0_REUSE_CANDIDATE}" | tr '[:upper:]' '[:lower:]')"
    LPD_CHECKPOINT="${S0_RUN_ROOT}/candidates/${reuse_slug}/${S0_REUSE_CHECKPOINT_REL}"
    export LPD_CHECKPOINT
    status waiting "waiting to reuse ${S0_REUSE_CANDIDATE} checkpoint"
    while [[ ! -e "${LPD_CHECKPOINT}" ]]; do
      reuse_status="${S0_RUN_ROOT}/candidates/${reuse_slug}/status.json"
      if [[ -f "${reuse_status}" ]] && \
        python3 -c \
          'import json,sys; raise SystemExit(json.load(open(sys.argv[1]))["phase"] != "failed")' \
          "${reuse_status}"; then
        printf >&2 '%s failed before publishing the shared checkpoint.\n' \
          "${S0_REUSE_CANDIDATE}"
        exit 5
      fi
      sleep 10
    done
    ;;
  *)
    printf >&2 'unknown S0_TRAIN_MODE=%q\n' "${S0_TRAIN_MODE}"
    exit 2
    ;;
esac

test -e "${LPD_CHECKPOINT}"
export LPD_OUTPUT_ROOT="${CANDIDATE_ROOT}/validation/gate_${LPD_RUN_ID}"
status validating "paired Gate${LPD_EPISODES} on LiftBarrier and LongPipelineDelivery"
"${FE_ROOT}/scripts/run_lpd_single_5090.sh" gate
test -f "${LPD_OUTPUT_ROOT}/gate_summary.json"

status complete "training/checkpoint and paired validation artifacts complete" 0
COMPLETED=1
printf 'S0 candidate %s complete. Artifacts: %s\n' \
  "${S0_CANDIDATE_ID}" "${CANDIDATE_ROOT}"
