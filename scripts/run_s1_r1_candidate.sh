#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FE_ROOT}/experiments/wam_flow/s1_r1/candidate.env"
: "${S1_R1_RUN_ROOT:?set S1_R1_RUN_ROOT}"
: "${S1_R1_READY_FILE:?set S1_R1_READY_FILE}"
: "${S1_R1_FAILED_FILE:?set S1_R1_FAILED_FILE}"
: "${S1_R1_BASE_REPO:?set S1_R1_BASE_REPO}"
: "${GPU_INDEX:?set GPU_INDEX}"
: "${S1_R1_UV_CACHE_DIR:?set S1_R1_UV_CACHE_DIR}"
: "${S1_R1_UV_ENV:?set S1_R1_UV_ENV}"
: "${S1_R1_ROBOFACTORY_ROOT:?set S1_R1_ROBOFACTORY_ROOT}"
: "${S1_R1_RF_PYTHON:?set S1_R1_RF_PYTHON}"
unset HF_TOKEN || true
test -f "${ENV_FILE}"
# Candidate env files contain only fixed, repository-owned scalar assignments.
# shellcheck source=/dev/null
source "${ENV_FILE}"
: "${S1_R1_CANDIDATE_ID:?candidate.env must set S1_R1_CANDIDATE_ID}"
: "${S1_R1_TOTAL_UPDATES:?candidate.env must set S1_R1_TOTAL_UPDATES}"
: "${LPD_POLICY_KIND:?candidate.env must set LPD_POLICY_KIND}"
: "${LPD_CONFIG_REL:?candidate.env must set LPD_CONFIG_REL}"
: "${LPD_CHECKPOINT_REL:?candidate.env must set LPD_CHECKPOINT_REL}"

CANDIDATE_SLUG="$(
  printf '%s' "${S1_R1_CANDIDATE_ID}" | tr '[:upper:]' '[:lower:]'
)"
CANDIDATE_ROOT="${S1_R1_RUN_ROOT}/candidates/${CANDIDATE_SLUG}"
STATUS_TOOL="${FE_ROOT}/scripts/s1_r1_runtime.py"
LOG_PATH="${CANDIDATE_ROOT}/logs/candidate.log"
mkdir -p "${CANDIDATE_ROOT}/logs" "${CANDIDATE_ROOT}/train"
exec > >(tee -a "${LOG_PATH}") 2>&1

COMPLETED=0
WAIT_HEARTBEAT_SECONDS="${S1_R1_WAIT_HEARTBEAT_SECONDS:-30}"
if [[ ! "${WAIT_HEARTBEAT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf >&2 'S1_R1_WAIT_HEARTBEAT_SECONDS must be a positive integer.\n'
  exit 2
fi
status() {
  local arguments=(
    status
    --run-root "${S1_R1_RUN_ROOT}"
    --candidate "${S1_R1_CANDIDATE_ID}"
    --phase "$1"
    --detail "${2:-}"
    --gpu-index "${GPU_INDEX}"
    --total-updates "${S1_R1_TOTAL_UPDATES}"
  )
  if (( $# >= 3 )); then
    arguments+=(--exit-code "$3")
  fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}

announce() {
  printf '[%s] %s %s\n' \
    "$(date -Is)" \
    "${S1_R1_CANDIDATE_ID}" \
    "$1"
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
    printf >&2 'Refusing existing non-link run path: %s\n' "${target}"
    exit 3
  fi
  ln -s "${source}" "${target}"
}

WAIT_STARTED="${SECONDS}"
NEXT_WAIT_NOTICE=0
while [[ ! -f "${S1_R1_READY_FILE}" ]]; do
  if [[ -f "${S1_R1_FAILED_FILE}" ]]; then
    printf >&2 'Shared S1-R1 preparation failed; inspect the prepare window.\n'
    exit 4
  fi
  WAIT_ELAPSED="$((SECONDS - WAIT_STARTED))"
  if (( WAIT_ELAPSED >= NEXT_WAIT_NOTICE )); then
    WAIT_DETAIL="waiting for shared dataset, DINO and RoboFactory; elapsed=${WAIT_ELAPSED}s"
    status waiting "${WAIT_DETAIL}"
    announce "${WAIT_DETAIL}"
    NEXT_WAIT_NOTICE="$((WAIT_ELAPSED + WAIT_HEARTBEAT_SECONDS))"
  fi
  sleep 5
done

status setup "shared environment ready; creating shared and isolated paths"
announce "shared environment ready; candidate setup started"
test -d "${S1_R1_BASE_REPO}/datasets"
test -d "${S1_R1_BASE_REPO}/artifacts"
mkdir -p \
  "${CANDIDATE_ROOT}/checkpoints" \
  "${CANDIDATE_ROOT}/outputs" \
  "${CANDIDATE_ROOT}/workspace_logs"
link_shared "${FE_ROOT}/datasets" "${S1_R1_BASE_REPO}/datasets"
link_shared "${FE_ROOT}/artifacts" "${S1_R1_BASE_REPO}/artifacts"
link_isolated_run_path "${FE_ROOT}/checkpoints" "${CANDIDATE_ROOT}/checkpoints"
link_isolated_run_path "${FE_ROOT}/outputs" "${CANDIDATE_ROOT}/outputs"
link_isolated_run_path "${FE_ROOT}/logs" "${CANDIDATE_ROOT}/workspace_logs"
announce "shared dataset/artifact links and isolated output paths are ready"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export UV_CACHE_DIR="${S1_R1_UV_CACHE_DIR}"
export UV_PROJECT_ENVIRONMENT="${S1_R1_UV_ENV}"
export ROBOFACTORY_ROOT="${S1_R1_ROBOFACTORY_ROOT}"
export RF_PYTHON="${S1_R1_RF_PYTHON}"
export LPD_POLICY_KIND
export LPD_EXPERIMENT_SLUG="s1_r1_${CANDIDATE_SLUG}"
export LPD_CONFIG="${FE_ROOT}/${LPD_CONFIG_REL}"
export LPD_CHECKPOINT="${FE_ROOT}/${LPD_CHECKPOINT_REL}"
export LPD_PROGRESS_LOG="${CANDIDATE_ROOT}/train/progress.jsonl"
export LPD_STAGE_LOG="${CANDIDATE_ROOT}/train/stages.jsonl"
export LPD_PORT="$((8872 + GPU_INDEX))"
export LPD_RUN_ID="${S1_R1_RUN_ID:-s1-r1}"
export LPD_EPISODES="${S1_R1_GATE_EPISODES:-20}"
export LPD_SEED_START="${S1_R1_GATE_SEED_START:-900}"
test -f "${LPD_CONFIG}"

status startup "loading data, DINO, model and DataLoader before the first update"
announce "trainer launch: ${LPD_POLICY_KIND}, ${S1_R1_TOTAL_UPDATES} updates on GPU ${GPU_INDEX}"
"${FE_ROOT}/scripts/run_lpd_single_5090.sh" train
test -e "${LPD_CHECKPOINT}"
announce "training complete; checkpoint is ready"

export LPD_OUTPUT_ROOT="${CANDIDATE_ROOT}/validation/gate_${LPD_RUN_ID}"
status validating "paired Gate${LPD_EPISODES} on LiftBarrier and LongPipelineDelivery"
announce "validation launch: paired Gate${LPD_EPISODES}"
"${FE_ROOT}/scripts/run_lpd_single_5090.sh" gate
test -f "${LPD_OUTPUT_ROOT}/gate_summary.json"

status complete "checkpoint and paired closed-loop validation complete" 0
COMPLETED=1
announce "training and validation complete"
printf 'S1-R1 candidate %s complete. Artifacts: %s\n' \
  "${S1_R1_CANDIDATE_ID}" "${CANDIDATE_ROOT}"
