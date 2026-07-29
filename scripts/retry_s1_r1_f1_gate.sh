#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
RETRY_ID="retry1"
DRY_RUN=0

usage() {
  printf 'usage: %s --run-id ID [--retry-id ID] [--dry-run]\n' "$0"
}

while (( $# )); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --retry-id)
      RETRY_ID="${2:?--retry-id requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

for value in "${RUN_ID}" "${RETRY_ID}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    printf >&2 'run and retry ids must be non-empty safe identifiers.\n'
    exit 2
  fi
done
if [[ "$(git -C "${FE_ROOT}" branch --show-current)" != \
  "s1/r1-f1-flow-cold" ]]; then
  printf >&2 'Switch to s1/r1-f1-flow-cold before retrying F1 Gate20.\n'
  exit 3
fi
if [[ -n "$(git -C "${FE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  printf >&2 'Refusing F1 Gate20 retry with tracked source changes.\n'
  exit 3
fi

COMMON_GIT_DIR="$(
  git -C "${FE_ROOT}" rev-parse --path-format=absolute --git-common-dir
)"
BASE_REPO="$(dirname "${COMMON_GIT_DIR}")"
WORKSPACE_ROOT="$(dirname "${BASE_REPO}")"
RUN_ROOT="${S1_R1_RUN_ROOT:-${BASE_REPO}/outputs/s1_r1_runs/${RUN_ID}}"
CANDIDATE_ROOT="${RUN_ROOT}/candidates/f1"
CHECKPOINT="$(
  realpath -m \
    "${CANDIDATE_ROOT}/checkpoints/s1_r1_f1_flow_cold/checkpoint_080000.pt"
)"
OUTPUT_ROOT="${CANDIDATE_ROOT}/validation/gate_${RUN_ID}_${RETRY_ID}"
CONFIG="${FE_ROOT}/configs/wam_flow/s1_r1_f1_flow_cold.yaml"
STATUS_TOOL="${FE_ROOT}/scripts/s1_r1_runtime.py"
ROBOFACTORY_ROOT="${S1_R1_ROBOFACTORY_ROOT:-${WORKSPACE_ROOT}/RoboFactory}"
RF_PYTHON="${S1_R1_RF_PYTHON:-${ROBOFACTORY_ROOT}/.venv/bin/python}"
UV_CACHE_DIR="${S1_R1_UV_CACHE_DIR:-${BASE_REPO}/.uv-cache}"
UV_PROJECT_ENVIRONMENT="${S1_R1_UV_ENV:-${BASE_REPO}/.venv}"
LOG_PATH="${CANDIDATE_ROOT}/logs/gate_${RUN_ID}_${RETRY_ID}.log"

printf 'S1-R1 F1 validation-only retry:\n'
printf '  branch: %s @ %s\n' \
  "$(git -C "${FE_ROOT}" branch --show-current)" \
  "$(git -C "${FE_ROOT}" rev-parse --short HEAD)"
printf '  checkpoint: %s\n  output: %s\n  log: %s\n' \
  "${CHECKPOINT}" "${OUTPUT_ROOT}" "${LOG_PATH}"
printf '  GPU: %s | port: %s | Gate episodes: %s\n' \
  "${GPU_INDEX:-1}" "${LPD_PORT:-8873}" "${S1_R1_GATE_EPISODES:-20}"

if (( DRY_RUN )); then
  printf 'Dry run: no process, status or output was changed.\n'
  exit 0
fi

for required in git jq python3 realpath uv; do
  command -v "${required}" >/dev/null || {
    printf >&2 'Missing required command: %s\n' "${required}"
    exit 3
  }
done
test -n "${TMUX:-}" || {
  printf >&2 'Run the F1 validation retry inside the permanent tmux session.\n'
  exit 3
}
test -f "${RUN_ROOT}/run_manifest.json"
jq -e --arg run_id "${RUN_ID}" '
  .round_id == "s1-r1" and .run_id == $run_id
' "${RUN_ROOT}/run_manifest.json" >/dev/null
test -f "${CHECKPOINT}"
test -f "${CONFIG}"
test -x "${RF_PYTHON}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  printf >&2 'Retry output already exists; choose a new --retry-id: %s\n' \
    "${OUTPUT_ROOT}"
  exit 3
fi

status() {
  local phase="$1"
  local detail="$2"
  shift 2
  python3 "${STATUS_TOOL}" status \
    --run-root "${RUN_ROOT}" \
    --candidate F1 \
    --phase "${phase}" \
    --detail "${detail}" \
    --gpu-index "${GPU_INDEX:-1}" \
    --total-updates 80000 \
    "$@"
}

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
COMPLETED=0
on_exit() {
  local code=$?
  if (( code != 0 )) && (( COMPLETED == 0 )); then
    status failed \
      "F1 Gate20 retry ${RETRY_ID} exited with code ${code}; see ${LOG_PATH}" \
      --exit-code "${code}" || true
  fi
}
trap on_exit EXIT

status validating \
  "F1 Gate20 retry ${RETRY_ID}; reusing completed 80k checkpoint"
S1_R1_RUN_ROOT="${RUN_ROOT}" \
GPU_INDEX="${GPU_INDEX:-1}" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}" \
RF_PYTHON="${RF_PYTHON}" \
LPD_POLICY_KIND=agent_flow \
LPD_CONFIG="${CONFIG}" \
LPD_CHECKPOINT="${CHECKPOINT}" \
LPD_EXPERIMENT_SLUG=s1_r1_f1 \
LPD_OUTPUT_ROOT="${OUTPUT_ROOT}" \
LPD_RUN_ID="${RUN_ID}-${RETRY_ID}" \
LPD_EPISODES="${S1_R1_GATE_EPISODES:-20}" \
LPD_SEED_START="${S1_R1_GATE_SEED_START:-900}" \
LPD_PORT="${LPD_PORT:-8873}" \
PYTHONUNBUFFERED=1 \
  "${FE_ROOT}/scripts/run_lpd_single_5090.sh" gate

status complete \
  "F1 checkpoint and Gate20 retry ${RETRY_ID} complete" \
  --exit-code 0
COMPLETED=1
printf 'S1-R1 F1 validation retry complete: %s\n' "${OUTPUT_ROOT}"
