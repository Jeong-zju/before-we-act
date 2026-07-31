#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S2_R5_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S2_R5_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${S2_R5_RUN_ROOT:-${FE_ROOT}/outputs/s2_r5_runs/${RUN_ID}}"
SESSION="${S2_R5_TMUX_SESSION:-}"
WINDOW_PREFIX="${S2_R5_WINDOW_PREFIX:-${RUN_ID}}"
PREPARE_FROM_S0=0
DRY_RUN=0
FOCUS_MONITOR=auto
PROTECTED_SOURCE="${S2_R5_PROTECTED_P0_SOURCE:-}"
HF_TOKEN_INPUT=""
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
source "${FE_ROOT}/scripts/s0_hf_token_fifo.sh"

CANDIDATES=(P0 P1)
BRANCHES=(s2/r5-p0-protected-shared s2/r5-p1-protected-role-mot)
WORKTREE_NAMES=(s2-r5-p0-protected-shared s2-r5-p1-protected-role-mot)
CONFIG_REL=configs/wam_flow/s2_r5_protected_team.yaml

usage() {
  printf 'usage: %s [--run-id ID] [--protected-p0 PATH] [--prepare-from-s0] [--dry-run]\n' "$0"
}
while (( $# )); do
  case "$1" in
    --run-id) RUN_ID="${2:?}"; shift 2 ;;
    --protected-p0) PROTECTED_SOURCE="${2:?}"; shift 2 ;;
    --prepare-from-s0) PREPARE_FROM_S0=1; shift ;;
    --focus-monitor) FOCUS_MONITOR=yes; shift ;;
    --no-focus-monitor) FOCUS_MONITOR=no; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
RUN_ROOT="${S2_R5_RUN_ROOT:-${FE_ROOT}/outputs/s2_r5_runs/${RUN_ID}}"
WINDOW_PREFIX="${S2_R5_WINDOW_PREFIX:-${RUN_ID}}"
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
CANDIDATE_WINDOWS=("${WINDOW_PREFIX}-p0" "${WINDOW_PREFIX}-p1")
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"

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
  tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "$1"
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
      tmux respawn-pane -k -t "${SESSION}:${name}" -c "${directory}" \
        "${command}" || return $?
      printf 'repaired:%s' "${name}"
    else
      printf 'reused:%s' "${name}"
    fi
    return
  fi
  create_window "${name}" "${directory}" "${command}"
}
find_protected_source() {
  if [[ -n "${PROTECTED_SOURCE}" && -f "${PROTECTED_SOURCE}" ]]; then
    PROTECTED_SOURCE="$(realpath "${PROTECTED_SOURCE}")"; return
  fi
  while IFS= read -r candidate; do PROTECTED_SOURCE="${candidate}"; break; done < <(
    find "${FE_ROOT}/outputs/s2_r4_runs" -type f \
      -path '*/candidates/p0/checkpoints/predictor.pt' 2>/dev/null \
      -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-
  )
  if [[ -z "${PROTECTED_SOURCE}" || ! -f "${PROTECTED_SOURCE}" ]]; then
    printf >&2 'No R4-P0 checkpoint found; pass --protected-p0 PATH.\n'
    exit 3
  fi
  PROTECTED_SOURCE="$(realpath "${PROTECTED_SOURCE}")"
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
  printf 'S2-R5 run=%s root=%s\n' "${RUN_ID}" "${RUN_ROOT}"
  printf 'tmux=%s (reused; never killed)\n' "${SESSION:-<auto>}"
  printf 'GPU0=%s GPU1=%s\n' "${BRANCHES[0]}" "${BRANCHES[1]}"
  printf 'shared data=%s/datasets/robofactory_multitask\n' "${FE_ROOT}"
  printf 'shared artifacts=%s/artifacts; isolated candidate outputs\n' "${FE_ROOT}"
  printf 'monitor=current program/status/update/task-batch/heartbeat/GPU/PID/R5 gates\n'
  if (( PREPARE_FROM_S0 )); then
    printf 'HF=S0 method (dataset Xet/default workers; DINO no-Xet/one worker; FIFO token)\n'
  else
    printf 'HF=existing-server reuse; use --prepare-from-s0 only for missing assets\n'
  fi
}

if (( DRY_RUN )); then print_plan; exit 0; fi
for command in git tmux jq nvidia-smi python3 uv flock realpath sha256sum; do
  if ! command -v "${command}" >/dev/null; then printf >&2 'Missing %s\n' "${command}"; exit 3; fi
done
resolve_session
find_protected_source
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then exit 2; fi
EXISTING_RUN=0
if [[ -e "${RUN_ROOT}" ]]; then
  if [[ ! -f "${RUN_ROOT}/run_manifest.json" ]]; then
    printf >&2 'Run root exists without an R5 manifest: %s\n' "${RUN_ROOT}"; exit 3
  fi
  EXISTING_RUN=1
fi
if (( $(nvidia-smi -L | sed -n '$=') != 2 )); then printf >&2 'Exactly two GPUs required.\n'; exit 3; fi
origin_url="$(git -C "${FE_ROOT}" remote get-url origin 2>/dev/null)"
if [[ -z "${origin_url}" ]]; then printf >&2 'Repository origin is missing.\n'; exit 3; fi
available_kib="$(df -Pk "${FE_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -z "${available_kib}" || "${available_kib}" -lt 52428800 ]]; then
  printf >&2 'At least 50 GiB free workspace disk is required; available KiB=%s.\n' "${available_kib:-?}"
  exit 3
fi
print_plan

mkdir -p "${WORKTREE_ROOT}"
WORKTREES=()
for index in "${!CANDIDATES[@]}"; do
  branch="${BRANCHES[index]}"; path="${WORKTREE_ROOT}/${WORKTREE_NAMES[index]}"
  git -C "${FE_ROOT}" fetch --no-tags origin \
    "+refs/heads/${branch}:refs/remotes/origin/${branch}" || exit $?
  if [[ -d "${path}/.git" || -f "${path}/.git" ]]; then
    if [[ "$(git -C "${path}" branch --show-current)" != "${branch}" ]]; then exit 3; fi
  else
    if ! git -C "${FE_ROOT}" show-ref --verify --quiet "refs/heads/${branch}"; then
      git -C "${FE_ROOT}" branch --track "${branch}" "origin/${branch}" || exit $?
    fi
    git -C "${FE_ROOT}" worktree add "${path}" "${branch}" || exit $?
  fi
  if [[ -n "$(git -C "${path}" status --porcelain --untracked-files=no)" ]]; then
    printf >&2 'Dirty candidate worktree: %s\n' "${path}"; exit 3
  fi
  git -C "${path}" merge --ff-only "origin/${branch}" || exit $?
  WORKTREES+=("${path}")
done
UV_CACHE="${FE_ROOT}/.uv-cache"; UV_ENV="${FE_ROOT}/.venv"
( cd "${FE_ROOT}" && uv run --frozen python scripts/validate_s2_r5_branch_pair.py \
  --p0-config "${WORKTREES[0]}/${CONFIG_REL}" \
  --p1-config "${WORKTREES[1]}/${CONFIG_REL}" ) || exit $?

if (( EXISTING_RUN == 0 )); then
  python3 "${FE_ROOT}/scripts/s2_r5_runtime.py" init \
    --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" --session "${SESSION}" \
    --window-prefix "${WINDOW_PREFIX}" --monitor-window "${MONITOR_WINDOW}" \
    --base-repo "${FE_ROOT}" --worktree "P0=${WORKTREES[0]}" \
    --worktree "P1=${WORKTREES[1]}" || exit $?
fi
prepare_command="$(shell_join env -u HF_TOKEN \
  "S2_R5_RUN_ROOT=${RUN_ROOT}" "S2_R5_READY_FILE=${READY_FILE}" \
  "S2_R5_FAILED_FILE=${FAILED_FILE}" \
  "S2_R5_PROTECTED_P0_SOURCE=${PROTECTED_SOURCE}" \
  "S2_R5_P0_CONFIG=${WORKTREES[0]}/${CONFIG_REL}" \
  "S2_R5_USE_S0_PREP=${PREPARE_FROM_S0}" \
  "S2_R5_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" \
  "UV_CACHE_DIR=${UV_CACHE}" "UV_PROJECT_ENVIRONMENT=${UV_ENV}" \
  bash scripts/prepare_s2_r5_shared.sh)"
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
    "GPU_INDEX=${index}" "S2_R5_RUN_ID=${RUN_ID}" \
    "S2_R5_RUN_ROOT=${RUN_ROOT}" "S2_R5_READY_FILE=${READY_FILE}" \
    "S2_R5_FAILED_FILE=${FAILED_FILE}" "S2_R5_BASE_REPO=${FE_ROOT}" \
    "S2_R5_UV_CACHE_DIR=${UV_CACHE}" "S2_R5_UV_ENV=${UV_ENV}" \
    bash scripts/run_s2_r5_candidate.sh)"
  evaluation="${RUN_ROOT}/candidates/${CANDIDATES[index],,}/validation/evaluation.json"
  if [[ ! -f "${evaluation}" ]]; then
    start_or_repair_window "${CANDIDATE_WINDOWS[index]}" "${WORKTREES[index]}" \
      "${candidate_command}" >/dev/null || exit $?
  fi
done
monitor_command="$(shell_join python3 scripts/s2_r5_runtime.py monitor \
  --run-root "${RUN_ROOT}" --interval 5)"
monitor_id="$(start_or_repair_window "${MONITOR_WINDOW}" "${FE_ROOT}" "${monitor_command}")" || exit $?
printf 'S2-R5 running in permanent tmux %s.\n' "${SESSION}"
printf 'Monitor: tmux select-window -t %s:%s\n' "${SESSION}" "${MONITOR_WINDOW}"
printf 'One-shot: python3 scripts/s2_r5_runtime.py monitor --once --run-root %s\n' "${RUN_ROOT}"
if [[ "${FOCUS_MONITOR}" == yes || ( "${FOCUS_MONITOR}" == auto && -n "${TMUX:-}" ) ]]; then
  tmux select-window -t "${SESSION}:${MONITOR_WINDOW}"
fi
