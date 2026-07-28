#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S0_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S0_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ATTACH=auto
DRY_RUN=0

usage() {
  printf 'usage: %s [--run-id ID] [--attach|--no-attach] [--dry-run]\n' "$0"
}

while (( $# )); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --attach)
      ATTACH=yes
      shift
      ;;
    --no-attach)
      ATTACH=no
      shift
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

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid run id: %q\n' "${RUN_ID}"
  exit 2
fi
SESSION="${S0_TMUX_SESSION:-wam-s0-${RUN_ID}}"
RUN_ROOT="${S0_RUN_ROOT:-${FE_ROOT}/outputs/s0_runs/${RUN_ID}}"
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"

CANDIDATES=(B0 B1 B2 B3)
BRANCHES=(
  round/s0-b0-legacy-moe-ensemble
  round/s0-b1-legacy-dense-ensemble
  round/s0-b2-flow-reference
  round/s0-b3-legacy-moe-latest
)
WORKTREE_NAMES=(
  s0-b0-legacy-moe-ensemble
  s0-b1-legacy-dense-ensemble
  s0-b2-flow-reference
  s0-b3-legacy-moe-latest
)

shell_join() {
  local result=""
  local value
  for value in "$@"; do
    printf -v result '%s%q ' "${result}" "${value}"
  done
  printf '%s' "${result% }"
}

candidate_command() {
  local gpu="$1"
  local command
  command="$(shell_join \
    env \
    "GPU_INDEX=${gpu}" \
    "S0_RUN_ID=${RUN_ID}" \
    "S0_RUN_ROOT=${RUN_ROOT}" \
    "S0_READY_FILE=${READY_FILE}" \
    "S0_FAILED_FILE=${FAILED_FILE}" \
    "S0_BASE_REPO=${FE_ROOT}" \
    "S0_GATE_EPISODES=${S0_GATE_EPISODES:-20}" \
    "S0_GATE_SEED_START=${S0_GATE_SEED_START:-900}" \
    bash scripts/run_s0_candidate.sh
  )"
  printf '%s' "${command}"
}

printf 'S0 parent: %s @ %s\n' \
  "$(git -C "${FE_ROOT}" branch --show-current)" \
  "$(git -C "${FE_ROOT}" rev-parse HEAD)"
printf 'Run root: %s\nTmux session: %s\n' "${RUN_ROOT}" "${SESSION}"
for index in "${!CANDIDATES[@]}"; do
  printf '%s GPU%s  %-43s  %s/%s\n' \
    "${CANDIDATES[index]}" "${index}" "${BRANCHES[index]}" \
    "${WORKTREE_ROOT}" "${WORKTREE_NAMES[index]}"
done

if (( DRY_RUN )); then
  printf '\nDry run: no worktrees, files, tmux sessions or GPU jobs were changed.\n'
  printf 'prepare: %s\n' "$(shell_join \
    env \
    GPU_INDEX=0 \
    "S0_RUN_ROOT=${RUN_ROOT}" \
    "S0_READY_FILE=${READY_FILE}" \
    "S0_FAILED_FILE=${FAILED_FILE}" \
    bash scripts/prepare_s0_shared.sh
  )"
  for index in "${!CANDIDATES[@]}"; do
    printf '%s: %s\n' "${CANDIDATES[index]}" "$(candidate_command "${index}")"
  done
  exit 0
fi

command -v git >/dev/null
command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
command -v python3 >/dev/null
if [[ "$(git -C "${FE_ROOT}" branch --show-current)" != "feat/model-improvements" ]]; then
  printf >&2 'Launch S0 from the feat/model-improvements worktree.\n'
  exit 3
fi
if [[ -n "$(git -C "${FE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  printf >&2 'Refusing S0 launch with tracked changes in the parent worktree.\n'
  exit 3
fi
GPU_COUNT="$(nvidia-smi -L | sed -n '$=')"
if (( GPU_COUNT < 4 )); then
  printf >&2 'S0 requires at least four visible GPUs; found %d.\n' "${GPU_COUNT}"
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf >&2 'Tmux session already exists: %s\n' "${SESSION}"
  exit 3
fi
if [[ -e "${RUN_ROOT}" ]]; then
  printf >&2 'Run root already exists: %s\n' "${RUN_ROOT}"
  exit 3
fi

mkdir -p "${WORKTREE_ROOT}"
WORKTREES=()
for index in "${!CANDIDATES[@]}"; do
  branch="${BRANCHES[index]}"
  path="${WORKTREE_ROOT}/${WORKTREE_NAMES[index]}"
  git -C "${FE_ROOT}" show-ref --verify --quiet "refs/heads/${branch}"
  if [[ -d "${path}" ]]; then
    actual="$(git -C "${path}" branch --show-current)"
    if [[ "${actual}" != "${branch}" ]]; then
      printf >&2 'Existing worktree has wrong branch: %s (%s)\n' \
        "${path}" "${actual}"
      exit 3
    fi
  else
    git -C "${FE_ROOT}" worktree add "${path}" "${branch}"
  fi
  test -f "${path}/experiments/wam_flow/s0/candidate.env"
  test -f "${path}/experiments/wam_flow/s0/candidate_card.yaml"
  WORKTREES+=("${path}")
done

init_arguments=(
  "${FE_ROOT}/scripts/s0_runtime.py" init
  --run-root "${RUN_ROOT}"
  --run-id "${RUN_ID}"
  --session "${SESSION}"
  --base-repo "${FE_ROOT}"
)
for index in "${!CANDIDATES[@]}"; do
  init_arguments+=(--worktree "${CANDIDATES[index]}=${WORKTREES[index]}")
done
python3 "${init_arguments[@]}"

prepare_command="$(shell_join \
  env \
  GPU_INDEX=0 \
  "S0_RUN_ROOT=${RUN_ROOT}" \
  "S0_READY_FILE=${READY_FILE}" \
  "S0_FAILED_FILE=${FAILED_FILE}" \
  bash scripts/prepare_s0_shared.sh
)"
tmux new-session -d -s "${SESSION}" -n prepare -c "${FE_ROOT}" "${prepare_command}"
tmux set-option -t "${SESSION}" remain-on-exit on
tmux set-option -t "${SESSION}" history-limit 200000

for index in "${!CANDIDATES[@]}"; do
  tmux new-window -d \
    -t "${SESSION}" \
    -n "$(printf '%s' "${CANDIDATES[index]}" | tr '[:upper:]' '[:lower:]')" \
    -c "${WORKTREES[index]}" \
    "$(candidate_command "${index}")"
done
monitor_command="$(shell_join \
  python3 scripts/s0_runtime.py monitor \
  --run-root "${RUN_ROOT}" \
  --interval 5
)"
tmux new-window -d -t "${SESSION}" -n monitor -c "${FE_ROOT}" "${monitor_command}"
tmux select-window -t "${SESSION}:monitor"

printf '\nS0 is running persistently in tmux.\n'
printf 'Attach: tmux attach -t %s\n' "${SESSION}"
printf 'Monitor once: python3 %s/scripts/s0_runtime.py monitor --once --run-root %s\n' \
  "${FE_ROOT}" "${RUN_ROOT}"
if [[ "${ATTACH}" == "yes" || ( "${ATTACH}" == "auto" && -t 0 && -t 1 ) ]]; then
  exec tmux attach -t "${SESSION}"
fi
