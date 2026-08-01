#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S3_R6_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S3_R6_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT=""
SESSION="${S3_R6_TMUX_SESSION:-}"
WINDOW_PREFIX=""
PREPARE_FROM_S0=0
DRY_RUN=0
FOCUS_MONITOR=auto
HF_TOKEN_INPUT=""
FLOW_SOURCE="${S3_R6_FLOW_SOURCE:-}"
PROTECTED_SOURCE="${S3_R6_PROTECTED_OWN_SOURCE:-}"
TEAM_SOURCE="${S3_R6_PROTECTED_TEAM_SOURCE:-}"

CANDIDATES=(R6L-P0 R6L-P1 R6J-P0 R6J-P1)
BRANCHES=(
  s3/r6l-p0-protected-local-aux
  s3/r6l-p1-protected-local-gated
  s3/r6j-p0-protected-team-offpath
  s3/r6j-p1-protected-team-gated
)
WORKTREE_NAMES=(
  s3-r6l-p0-protected-local-aux
  s3-r6l-p1-protected-local-gated
  s3-r6j-p0-protected-team-offpath
  s3-r6j-p1-protected-team-gated
)
CONFIG_REL=configs/wam_flow/s3_r6.yaml

usage() {
  printf 'usage: %s [--run-id ID] [--flow PATH] [--protected-own PATH] [--protected-team PATH] [--prepare-from-s0] [--dry-run]\n' "$0"
}
while (( $# )); do
  case "$1" in
    --run-id) RUN_ID="${2:?}"; shift 2 ;;
    --flow) FLOW_SOURCE="${2:?}"; shift 2 ;;
    --protected-own) PROTECTED_SOURCE="${2:?}"; shift 2 ;;
    --protected-team) TEAM_SOURCE="${2:?}"; shift 2 ;;
    --prepare-from-s0) PREPARE_FROM_S0=1; shift ;;
    --focus-monitor) FOCUS_MONITOR=yes; shift ;;
    --no-focus-monitor) FOCUS_MONITOR=no; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid run id: %s\n' "${RUN_ID}"; exit 2
fi
RUN_ROOT="${S3_R6_RUN_ROOT:-${FE_ROOT}/outputs/s3_r6_runs/${RUN_ID}}"
WINDOW_PREFIX="${S3_R6_WINDOW_PREFIX:-${RUN_ID}}"
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"
CANDIDATE_WINDOWS=(
  "${WINDOW_PREFIX}-r6l-p0" "${WINDOW_PREFIX}-r6l-p1"
  "${WINDOW_PREFIX}-r6j-p0" "${WINDOW_PREFIX}-r6j-p1"
)
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
source "${FE_ROOT}/scripts/s0_hf_token_fifo.sh"

shell_join() {
  result=""
  for value in "$@"; do printf -v result '%s%q ' "${result}" "${value}"; done
  printf '%s' "${result% }"
}
resolve_session() {
  if [[ -n "${TMUX:-}" ]]; then SESSION="$(tmux display-message -p '#S')"; return; fi
  if [[ -n "${SESSION}" ]] && tmux has-session -t "${SESSION}" 2>/dev/null; then return; fi
  mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null)
  if (( ${#sessions[@]} != 1 )); then
    printf >&2 'Need one existing permanent tmux session; found %d.\n' "${#sessions[@]}"
    exit 3
  fi
  SESSION="${sessions[0]}"
}
window_exists() {
  tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null | grep -Fxq "$1"
}
create_window() {
  window_id="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION}:" \
    -n "$1" -c "$2" "$3")" || return $?
  tmux set-option -w -t "${window_id}" remain-on-exit on
  tmux set-option -w -t "${window_id}" history-limit 200000
  printf '%s' "${window_id}"
}
start_or_repair_window() {
  name="$1"; directory="$2"; command="$3"
  if window_exists "${name}"; then
    dead="$(tmux display-message -p -t "${SESSION}:${name}" '#{pane_dead}')"
    if [[ "${dead}" == "1" ]]; then
      tmux respawn-pane -k -t "${SESSION}:${name}" -c "${directory}" "${command}" || return $?
      printf 'repaired:%s' "${name}"
    else
      printf 'reused:%s' "${name}"
    fi
    return
  fi
  create_window "${name}" "${directory}" "${command}"
}
prompt_hf_token() {
  if [[ ! -r /dev/tty ]]; then printf >&2 'S0 mode needs an interactive terminal.\n'; exit 3; fi
  printf 'Hugging Face read token (hidden): ' >/dev/tty
  IFS= read -r -s HF_TOKEN_INPUT </dev/tty
  printf '\n' >/dev/tty
  if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
    printf >&2 'Invalid Hugging Face token.\n'; exit 3
  fi
}
print_plan() {
  printf 'S3-R6 run=%s root=%s\n' "${RUN_ID}" "${RUN_ROOT}"
  printf 'tmux=%s (existing permanent session; never killed)\n' "${SESSION:-<auto>}"
  printf 'phase1 GPU0=%s GPU1=%s\n' "${BRANCHES[0]}" "${BRANCHES[1]}"
  printf 'phase2 GPU0=%s GPU1=%s (auto-starts after R6L pair result)\n' \
    "${BRANCHES[2]}" "${BRANCHES[3]}"
  printf 'shared data=%s/datasets/robofactory_multitask; artifacts=%s/artifacts\n' \
    "${FE_ROOT}" "${FE_ROOT}"
  printf 'monitor=program/status/phase/update/loss/gate/rollout progress/heartbeat/GPU/PID/S3 gates\n'
  if (( PREPARE_FROM_S0 )); then
    printf 'HF=S0 FIFO method; dataset Xet/default concurrency; DINO no-Xet/one worker\n'
  else
    printf 'HF=existing asset reuse; no token requested\n'
  fi
}

if (( DRY_RUN )); then print_plan; exit 0; fi
for command in git tmux jq nvidia-smi python3 uv flock realpath sha256sum; do
  if ! command -v "${command}" >/dev/null; then printf >&2 'Missing %s\n' "${command}"; exit 3; fi
done
resolve_session
if (( $(nvidia-smi -L | sed -n '$=') != 2 )); then
  printf >&2 'S3-R6 requires exactly two visible GPUs.\n'; exit 3
fi
if [[ -z "$(git -C "${FE_ROOT}" remote get-url origin 2>/dev/null)" ]]; then
  printf >&2 'Repository origin is missing.\n'; exit 3
fi
if [[ "$(git -C "${FE_ROOT}" branch --show-current)" != "feat/model-improvements" ]]; then
  printf >&2 'Launch S3-R6 from feat/model-improvements.\n'; exit 3
fi
if [[ -n "$(git -C "${FE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  printf >&2 'Refusing launch with tracked parent-worktree changes.\n'; exit 3
fi
available_kib="$(df -Pk "${FE_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -z "${available_kib}" || "${available_kib}" -lt 52428800 ]]; then
  printf >&2 'At least 50 GiB free workspace disk is required.\n'; exit 3
fi
print_plan

EXISTING_RUN=0
if [[ -e "${RUN_ROOT}" ]]; then
  if [[ ! -f "${RUN_ROOT}/run_manifest.json" ]]; then
    printf >&2 'Run root exists without S3 manifest: %s\n' "${RUN_ROOT}"; exit 3
  fi
  EXISTING_RUN=1
fi
mkdir -p "${WORKTREE_ROOT}" || exit $?
WORKTREES=()
for index in "${!CANDIDATES[@]}"; do
  branch="${BRANCHES[index]}"; path="${WORKTREE_ROOT}/${WORKTREE_NAMES[index]}"
  git -C "${FE_ROOT}" fetch --no-tags origin \
    "+refs/heads/${branch}:refs/remotes/origin/${branch}" || exit $?
  if [[ -d "${path}/.git" || -f "${path}/.git" ]]; then
    if [[ "$(git -C "${path}" branch --show-current)" != "${branch}" ]]; then
      printf >&2 'Existing worktree has wrong branch: %s\n' "${path}"; exit 3
    fi
  else
    if ! git -C "${FE_ROOT}" show-ref --verify --quiet "refs/heads/${branch}"; then
      git -C "${FE_ROOT}" branch "${branch}" "refs/remotes/origin/${branch}" || exit $?
    fi
    git -C "${FE_ROOT}" worktree add "${path}" "${branch}" || exit $?
  fi
  if [[ -n "$(git -C "${path}" status --porcelain --untracked-files=no)" ]]; then
    printf >&2 'Dirty S3 candidate worktree: %s\n' "${path}"; exit 3
  fi
  git -C "${path}" merge --ff-only "origin/${branch}" || exit $?
  WORKTREES+=("${path}")
done
( cd "${FE_ROOT}" && uv run --frozen python scripts/validate_s3_r6_branch_matrix.py \
  --r6l-p0 "${WORKTREES[0]}/${CONFIG_REL}" \
  --r6l-p1 "${WORKTREES[1]}/${CONFIG_REL}" \
  --r6j-p0 "${WORKTREES[2]}/${CONFIG_REL}" \
  --r6j-p1 "${WORKTREES[3]}/${CONFIG_REL}" ) || exit $?

if (( EXISTING_RUN == 0 )); then
  init_args=(init --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" \
    --session "${SESSION}" --window-prefix "${WINDOW_PREFIX}" \
    --monitor-window "${MONITOR_WINDOW}" --base-repo "${FE_ROOT}")
  for index in "${!CANDIDATES[@]}"; do
    init_args+=(--worktree "${CANDIDATES[index]}=${WORKTREES[index]}")
  done
  python3 "${FE_ROOT}/scripts/s3_r6_runtime.py" "${init_args[@]}" || exit $?
fi
UV_CACHE="${FE_ROOT}/.uv-cache"; UV_ENV="${FE_ROOT}/.venv"
ROBOFACTORY_ROOT="${WORKSPACE_ROOT}/RoboFactory"
RF_PYTHON="${ROBOFACTORY_ROOT}/.venv/bin/python"
prepare_command="$(shell_join env -u HF_TOKEN \
  "S3_R6_RUN_ROOT=${RUN_ROOT}" "S3_R6_READY_FILE=${READY_FILE}" \
  "S3_R6_FAILED_FILE=${FAILED_FILE}" "S3_R6_USE_S0_PREP=${PREPARE_FROM_S0}" \
  "S3_R6_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" "S3_R6_FLOW_SOURCE=${FLOW_SOURCE}" \
  "S3_R6_PROTECTED_OWN_SOURCE=${PROTECTED_SOURCE}" \
  "S3_R6_PROTECTED_TEAM_SOURCE=${TEAM_SOURCE}" \
  "UV_CACHE_DIR=${UV_CACHE}" "UV_PROJECT_ENVIRONMENT=${UV_ENV}" \
  bash scripts/prepare_s3_r6_shared.sh)"
if [[ ! -f "${READY_FILE}" ]]; then
  if (( PREPARE_FROM_S0 )) && ! window_exists "${PREPARE_WINDOW}"; then
    prompt_hf_token
    s0_prepare_hf_token_fifo
    trap s0_cleanup_hf_secret EXIT
  fi
  start_or_repair_window "${PREPARE_WINDOW}" "${FE_ROOT}" \
    "${prepare_command}" >/dev/null || exit $?
  if (( PREPARE_FROM_S0 )) && [[ -n "${HF_TOKEN_INPUT}" ]]; then
    s0_deliver_hf_token
    trap - EXIT
  fi
fi

for index in "${!CANDIDATES[@]}"; do
  candidate_command="$(shell_join env -u HF_TOKEN \
    "GPU_INDEX=$((index % 2))" "S3_R6_RUN_ID=${RUN_ID}" \
    "S3_R6_RUN_ROOT=${RUN_ROOT}" "S3_R6_READY_FILE=${READY_FILE}" \
    "S3_R6_FAILED_FILE=${FAILED_FILE}" "S3_R6_BASE_REPO=${FE_ROOT}" \
    "S3_R6_UV_CACHE_DIR=${UV_CACHE}" "S3_R6_UV_ENV=${UV_ENV}" \
    "S3_R6_ROBOFACTORY_ROOT=${ROBOFACTORY_ROOT}" "S3_R6_RF_PYTHON=${RF_PYTHON}" \
    "S3_R6_GATE_EPISODES=${S3_R6_GATE_EPISODES:-20}" \
    "S3_R6_GATE_SEED_START=${S3_R6_GATE_SEED_START:-900}" \
    bash scripts/run_s3_r6_candidate.sh)"
  start_or_repair_window "${CANDIDATE_WINDOWS[index]}" "${WORKTREES[index]}" \
    "${candidate_command}" >/dev/null || exit $?
done
monitor_command="$(shell_join python3 scripts/s3_r6_runtime.py monitor \
  --run-root "${RUN_ROOT}" --interval 5)"
start_or_repair_window "${MONITOR_WINDOW}" "${FE_ROOT}" \
  "${monitor_command}" >/dev/null || exit $?
printf 'S3-R6 running two-by-two in permanent tmux %s.\n' "${SESSION}"
printf 'Monitor: tmux select-window -t %s:%s\n' "${SESSION}" "${MONITOR_WINDOW}"
printf 'One-shot: python3 scripts/s3_r6_runtime.py monitor --once --run-root %s\n' "${RUN_ROOT}"
if [[ "${FOCUS_MONITOR}" == yes || ( "${FOCUS_MONITOR}" == auto && -n "${TMUX:-}" ) ]]; then
  tmux select-window -t "${SESSION}:${MONITOR_WINDOW}"
fi
