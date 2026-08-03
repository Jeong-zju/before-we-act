#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S4_R7_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S4_R7_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT=""
SESSION="ssh_tmux"
WINDOW_PREFIX=""
PREPARE_FROM_S0=0
DRY_RUN=0
FOCUS_MONITOR=0
HF_TOKEN_INPUT=""

CANDIDATES=(P0 P1)
BRANCHES=(
  s4/r7-p0-token-preserving-evidence
  s4/r7-p1-world-utility-coupling
)
WORKTREE_NAMES=(
  s4-r7-p0-token-preserving-evidence
  s4-r7-p1-world-utility-coupling
)
CONFIG_RELS=(
  configs/wam_flow/s4_r7.yaml
  configs/wam_flow/s4_r7.yaml
)
PREPARE_REL=scripts/prepare_s4_r7_shared.sh
CANDIDATE_REL=scripts/run_s4_r7_candidate.sh
RUNTIME_REL=scripts/s4_r7_runtime.py
PAIR_VALIDATOR_REL=scripts/validate_s4_r7_branch_pair.py

usage() {
  printf 'usage: %s [--run-id ID] [--prepare-from-s0] [--focus-monitor] [--dry-run]\n' "$0"
}

fail() {
  printf >&2 'S4-R7 launcher error: %s\n' "$*"
  exit 3
}

while (( $# )); do
  case "$1" in
    --run-id)
      if (( $# < 2 )); then printf >&2 '%s\n' '--run-id requires a value'; exit 2; fi
      RUN_ID="$2"; shift 2
      ;;
    --prepare-from-s0) PREPARE_FROM_S0=1; shift ;;
    --focus-monitor) FOCUS_MONITOR=1; shift ;;
    --no-focus-monitor) FOCUS_MONITOR=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid run id: %s\n' "${RUN_ID}"
  exit 2
fi
RUN_ROOT="${S4_R7_RUN_ROOT:-${FE_ROOT}/outputs/s4_r7_runs/${RUN_ID}}"
WINDOW_PREFIX="${S4_R7_WINDOW_PREFIX:-${RUN_ID}}"
if [[ ! "${WINDOW_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid tmux window prefix: %s\n' "${WINDOW_PREFIX}"
  exit 2
fi
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
CANDIDATE_WINDOWS=("${WINDOW_PREFIX}-p0" "${WINDOW_PREFIX}-p1")
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"

TOKEN_HELPER="${FE_ROOT}/scripts/s0_hf_token_fifo.sh"
if [[ ! -r "${TOKEN_HELPER}" ]]; then
  fail "missing protected Hugging Face FIFO helper: ${TOKEN_HELPER}"
fi
# shellcheck source=scripts/s0_hf_token_fifo.sh
source "${TOKEN_HELPER}" || fail "could not load ${TOKEN_HELPER}"

shell_join() {
  local result=""
  local value
  for value in "$@"; do
    printf -v result '%s%q ' "${result}" "${value}"
  done
  printf '%s' "${result% }"
}

require_commands() {
  local command
  for command in git tmux jq nvidia-smi python3 uv flock realpath sha256sum df awk grep sed; do
    if ! command -v "${command}" >/dev/null 2>&1; then
      fail "missing required command: ${command}"
    fi
  done
}

resolve_permanent_session() {
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    fail "required permanent tmux session does not exist: ${SESSION}"
  fi
  if [[ -n "${TMUX:-}" ]]; then
    local current
    current="$(tmux display-message -p '#S')" || fail "cannot identify current tmux session"
    if [[ "${current}" != "${SESSION}" ]]; then
      fail "current tmux session is ${current}; S4-R7 must reuse ${SESSION}"
    fi
  fi
}

window_exists() {
  tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null | grep -Fxq "$1"
}

create_window() {
  local name="$1"
  local directory="$2"
  local command="$3"
  local window_id
  window_id="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION}:" \
    -n "${name}" -c "${directory}" "${command}")" || return $?
  tmux set-option -w -t "${window_id}" remain-on-exit on || return $?
  tmux set-option -w -t "${window_id}" history-limit 200000 || return $?
  printf '%s' "${window_id}"
}

start_or_repair_window() {
  local name="$1"
  local directory="$2"
  local command="$3"
  local dead
  if window_exists "${name}"; then
    dead="$(tmux display-message -p -t "${SESSION}:${name}" '#{pane_dead}')" || return $?
    if [[ "${dead}" == "1" ]]; then
      tmux respawn-pane -k -t "${SESSION}:${name}" -c "${directory}" "${command}" || return $?
      printf 'repaired:%s' "${name}"
    else
      printf 'reused:%s' "${name}"
    fi
    return 0
  fi
  create_window "${name}" "${directory}" "${command}"
}

prompt_hf_token() {
  if [[ ! -r /dev/tty ]]; then
    fail "--prepare-from-s0 requires an interactive terminal"
  fi
  printf 'Hugging Face read token (hidden): ' >/dev/tty
  IFS= read -r -s HF_TOKEN_INPUT </dev/tty
  printf '\n' >/dev/tty
  if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
    fail "invalid Hugging Face token"
  fi
}

remote_head() {
  local branch="$1"
  git -C "${FE_ROOT}" ls-remote --exit-code --heads origin \
    "refs/heads/${branch}" 2>/dev/null | awk 'NR==1 {print $1}'
}

validate_existing_run() {
  if [[ ! -e "${RUN_ROOT}" ]]; then
    local target
    for target in "${PREPARE_WINDOW}" "${CANDIDATE_WINDOWS[@]}" "${MONITOR_WINDOW}"; do
      if window_exists "${target}"; then
        fail "tmux window exists without a matching run manifest: ${target}"
      fi
    done
    return 0
  fi
  local manifest="${RUN_ROOT}/run_manifest.json"
  if [[ ! -f "${manifest}" ]]; then
    fail "run root exists without run_manifest.json: ${RUN_ROOT}"
  fi
  local manifest_round manifest_run manifest_session manifest_prefix
  manifest_round="$(jq -er '.round_id | strings' "${manifest}")" || fail "invalid round_id in ${manifest}"
  manifest_run="$(jq -er '.run_id | strings' "${manifest}")" || fail "invalid run_id in ${manifest}"
  manifest_session="$(jq -er '.tmux_session | strings' "${manifest}")" || fail "invalid tmux_session in ${manifest}"
  manifest_prefix="$(jq -er '.tmux_window_prefix | strings' "${manifest}")" || fail "invalid window prefix in ${manifest}"
  if [[ "${manifest_round}" != s4-r7 || "${manifest_run}" != "${RUN_ID}" || \
        "${manifest_session}" != "${SESSION}" || "${manifest_prefix}" != "${WINDOW_PREFIX}" ]]; then
    fail "existing run manifest identity does not match requested S4-R7 run"
  fi
}

validate_launch_read_only() {
  require_commands
  resolve_permanent_session

  local gpu_count
  gpu_count="$(nvidia-smi -L | sed -n '$=')" || fail "nvidia-smi -L failed"
  if [[ "${gpu_count}" != 2 ]]; then
    fail "S4-R7 requires exactly two visible GPUs; found ${gpu_count:-0}"
  fi
  if [[ -z "$(git -C "${FE_ROOT}" remote get-url origin 2>/dev/null)" ]]; then
    fail "repository origin is missing"
  fi
  if [[ "$(git -C "${FE_ROOT}" branch --show-current)" != feat/model-improvements ]]; then
    fail "launch from feat/model-improvements"
  fi
  # Deliberately includes untracked files.  Candidate identity and launcher
  # provenance are unsafe if a dry run silently ignores an untracked override.
  local dirty
  dirty="$(git -C "${FE_ROOT}" status --porcelain)" || fail "git status failed"
  if [[ -n "${dirty}" ]]; then
    printf >&2 '%s\n' "${dirty}"
    fail "parent worktree is dirty (tracked or untracked files are both rejected)"
  fi

  local parent_commit remote_parent
  parent_commit="$(git -C "${FE_ROOT}" rev-parse HEAD)" || fail "cannot resolve parent HEAD"
  remote_parent="$(remote_head feat/model-improvements)" || fail "cannot read origin/feat/model-improvements"
  if [[ "${remote_parent}" != "${parent_commit}" ]]; then
    fail "local feat/model-improvements is not the pushed remote head"
  fi

  local script
  for script in "${PREPARE_REL}" "${CANDIDATE_REL}" "${RUNTIME_REL}" \
    "${PAIR_VALIDATOR_REL}" scripts/prepare_shared_hdf5_receipt.py \
    scripts/prepare_s4_future_feature_cache.py; do
    if [[ ! -f "${FE_ROOT}/${script}" ]]; then
      fail "missing S4-R7 public-slice script: ${script}"
    fi
  done
  local branch
  for branch in "${BRANCHES[@]}"; do
    if [[ -z "$(remote_head "${branch}")" ]]; then
      fail "missing pushed candidate branch: ${branch}"
    fi
  done

  local available_kib
  available_kib="$(df -Pk "${FE_ROOT}" | awk 'NR==2 {print $4}')" || fail "disk query failed"
  if [[ -z "${available_kib}" || "${available_kib}" -lt 52428800 ]]; then
    fail "at least 50 GiB free workspace disk is required; available KiB=${available_kib:-?}"
  fi
  validate_existing_run
}

print_plan() {
  printf 'S4-R7 run=%s root=%s\n' "${RUN_ID}" "${RUN_ROOT}"
  printf 'tmux=%s (existing permanent session; no session lifecycle operation)\n' "${SESSION}"
  printf 'windows=%s {%s,%s,%s,%s}; remain-on-exit=on\n' \
    "${WINDOW_PREFIX}" "${PREPARE_WINDOW}" "${CANDIDATE_WINDOWS[0]}" \
    "${CANDIDATE_WINDOWS[1]}" "${MONITOR_WINDOW}"
  printf 'GPU0=%s GPU1=%s; independent candidates, no DDP\n' "${BRANCHES[0]}" "${BRANCHES[1]}"
  printf 'training=30k fast-selection; micro4/accum3/effective12; target 1152000 agent windows; Flow unfreezes at 6400\n'
  printf 'speed=two-GPU shared future DINO-PCA cache + current-view-only online DINO + fused AdamW; paired preflight requires >=0.75 update/s\n'
  printf 'validation=normal first; then legacy/gate-zero/shuffle-all core; four diagnostic conditions last\n'
  printf 'shared=%s/datasets/robofactory_multitask and %s/artifacts; candidate outputs remain isolated\n' \
    "${FE_ROOT}" "${FE_ROOT}"
  printf 'heartbeat=20s; STALE after 75s; monitor includes program/phase/PIDs/GPU/rollout/batch/exposure/optimizer/preflight/special gates\n'
  if (( PREPARE_FROM_S0 )); then
    printf 'HF=hidden mode-0600 FIFO; no token in export/argv/tmux/manifest/log\n'
  else
    printf 'HF=reuse existing data/cache/artifacts; no token requested\n'
  fi
  if (( DRY_RUN )); then
    printf 'dry-run=all read-only dependency/session/GPU/git-dirty/remote-branch/disk/run-identity checks passed; no files or windows created\n'
  fi
}

validate_launch_read_only
print_plan
if (( DRY_RUN )); then
  exit 0
fi

PARENT_COMMIT="$(git -C "${FE_ROOT}" rev-parse HEAD)" || fail "cannot resolve parent commit"
if ! git -C "${FE_ROOT}" fetch --no-tags origin \
  "+refs/heads/feat/model-improvements:refs/remotes/origin/feat/model-improvements" \
  "+refs/heads/${BRANCHES[0]}:refs/remotes/origin/${BRANCHES[0]}" \
  "+refs/heads/${BRANCHES[1]}:refs/remotes/origin/${BRANCHES[1]}"; then
  fail "candidate branch fetch failed"
fi

if ! mkdir -p "${WORKTREE_ROOT}"; then
  fail "cannot create worktree root: ${WORKTREE_ROOT}"
fi
WORKTREES=()
for index in "${!CANDIDATES[@]}"; do
  branch="${BRANCHES[index]}"
  path="${WORKTREE_ROOT}/${WORKTREE_NAMES[index]}"
  if [[ -d "${path}/.git" || -f "${path}/.git" ]]; then
    if [[ "$(git -C "${path}" branch --show-current)" != "${branch}" ]]; then
      fail "existing worktree has wrong branch: ${path}"
    fi
  else
    if ! git -C "${FE_ROOT}" show-ref --verify --quiet "refs/heads/${branch}"; then
      git -C "${FE_ROOT}" branch "${branch}" "refs/remotes/origin/${branch}" \
        || fail "cannot create local branch ${branch}"
    fi
    git -C "${FE_ROOT}" worktree add "${path}" "${branch}" \
      || fail "cannot create worktree ${path}"
  fi
  candidate_dirty="$(git -C "${path}" status --porcelain)" || fail "git status failed for ${path}"
  if [[ -n "${candidate_dirty}" ]]; then
    printf >&2 '%s\n' "${candidate_dirty}"
    fail "candidate worktree is dirty (including untracked): ${path}"
  fi
  git -C "${path}" merge --ff-only "origin/${branch}" \
    || fail "cannot fast-forward ${branch}"
  if ! git -C "${path}" merge-base --is-ancestor "${PARENT_COMMIT}" HEAD; then
    fail "candidate branch does not descend from public parent ${PARENT_COMMIT}: ${branch}"
  fi
  if [[ ! -f "${path}/${CONFIG_RELS[index]}" ]]; then
    fail "candidate config is missing: ${path}/${CONFIG_RELS[index]}"
  fi
  if [[ ! -f "${path}/${CANDIDATE_REL}" ]]; then
    fail "candidate runner is missing from worktree: ${path}/${CANDIDATE_REL}"
  fi
  WORKTREES+=("${path}")
done

if ! ( cd "${FE_ROOT}" && uv run --frozen python "${PAIR_VALIDATOR_REL}" \
  --p0-config "${WORKTREES[0]}/${CONFIG_RELS[0]}" \
  --p1-config "${WORKTREES[1]}/${CONFIG_RELS[1]}" --config-only ); then
  fail "candidate-axis/config pair validation failed"
fi

EXISTING_RUN=0
if [[ -f "${RUN_ROOT}/run_manifest.json" ]]; then
  EXISTING_RUN=1
fi
if (( EXISTING_RUN == 0 )); then
  python3 "${FE_ROOT}/${RUNTIME_REL}" init \
    --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" --session "${SESSION}" \
    --window-prefix "${WINDOW_PREFIX}" --monitor-window "${MONITOR_WINDOW}" \
    --base-repo "${FE_ROOT}" --parent-commit "${PARENT_COMMIT}" \
    --worktree "P0=${WORKTREES[0]}" --worktree "P1=${WORKTREES[1]}" \
    || fail "runtime manifest initialization failed"
fi

UV_CACHE="${FE_ROOT}/.uv-cache"
UV_ENV="${FE_ROOT}/.venv"
ROBOFACTORY_ROOT="${WORKSPACE_ROOT}/RoboFactory"
RF_PYTHON="${ROBOFACTORY_ROOT}/.venv/bin/python"
prepare_args=(
  bash "${PREPARE_REL}"
  --run-id "${RUN_ID}"
  --run-root "${RUN_ROOT}"
  --ready-file "${READY_FILE}"
  --failed-file "${FAILED_FILE}"
  --base-repo "${FE_ROOT}"
  --p0-worktree "${WORKTREES[0]}"
  --p1-worktree "${WORKTREES[1]}"
  --heartbeat-seconds 20
)
if (( PREPARE_FROM_S0 )); then
  prepare_args+=(--prepare-from-s0 --hf-token-fifo "${HF_TOKEN_FIFO}")
fi
prepare_command="$(shell_join env -u HF_TOKEN \
  "S4_R7_RUN_ROOT=${RUN_ROOT}" "S4_R7_READY_FILE=${READY_FILE}" \
  "S4_R7_FAILED_FILE=${FAILED_FILE}" "S4_R7_HEARTBEAT_SECONDS=20" \
  "S4_R7_USE_S0_PREP=${PREPARE_FROM_S0}" \
  "S4_R7_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" \
  "S4_R7_ROBOFACTORY_ROOT=${ROBOFACTORY_ROOT}" \
  "S4_R7_RF_PYTHON=${RF_PYTHON}" \
  "UV_CACHE_DIR=${UV_CACHE}" "UV_PROJECT_ENVIRONMENT=${UV_ENV}" \
  "${prepare_args[@]}")"

PREPARE_NEEDS_START=0
if ! window_exists "${PREPARE_WINDOW}"; then
  PREPARE_NEEDS_START=1
elif [[ "$(tmux display-message -p -t "${SESSION}:${PREPARE_WINDOW}" '#{pane_dead}')" == 1 ]]; then
  PREPARE_NEEDS_START=1
fi
if (( PREPARE_FROM_S0 && PREPARE_NEEDS_START )); then
  prompt_hf_token
  s0_prepare_hf_token_fifo || fail "could not create protected token FIFO"
  trap s0_cleanup_hf_secret EXIT
fi
start_or_repair_window "${PREPARE_WINDOW}" "${FE_ROOT}" "${prepare_command}" >/dev/null \
  || fail "could not start prepare window"
if (( PREPARE_FROM_S0 && PREPARE_NEEDS_START )); then
  s0_deliver_hf_token || fail "could not deliver token to prepare FIFO"
  trap - EXIT
fi

for index in "${!CANDIDATES[@]}"; do
  candidate_command="$(shell_join env -u HF_TOKEN \
    "S4_R7_RUN_ROOT=${RUN_ROOT}" "S4_R7_HEARTBEAT_SECONDS=20" \
    "UV_CACHE_DIR=${UV_CACHE}" "UV_PROJECT_ENVIRONMENT=${UV_ENV}" \
    bash "${CANDIDATE_REL}" \
    --candidate "${CANDIDATES[index]}" \
    --run-id "${RUN_ID}" \
    --run-root "${RUN_ROOT}" \
    --ready-file "${READY_FILE}" \
    --failed-file "${FAILED_FILE}" \
    --config "${WORKTREES[index]}/${CONFIG_RELS[index]}" \
    --gpu-index "${index}" \
    --heartbeat-seconds 20)"
  start_or_repair_window "${CANDIDATE_WINDOWS[index]}" "${WORKTREES[index]}" \
    "${candidate_command}" >/dev/null || fail "could not start ${CANDIDATES[index]} window"
done

monitor_command="$(shell_join python3 "${RUNTIME_REL}" monitor \
  --run-root "${RUN_ROOT}" --interval 60)"
start_or_repair_window "${MONITOR_WINDOW}" "${FE_ROOT}" "${monitor_command}" >/dev/null \
  || fail "could not start monitor window"

printf 'S4-R7 is running in permanent tmux %s.\n' "${SESSION}"
printf 'Monitor: tmux select-window -t %s:%s\n' "${SESSION}" "${MONITOR_WINDOW}"
printf 'One-shot: python3 %s monitor --once --run-root %s\n' \
  "${RUNTIME_REL}" "${RUN_ROOT}"
if (( FOCUS_MONITOR )); then
  tmux select-window -t "${SESSION}:${MONITOR_WINDOW}" \
    || fail "could not focus monitor window"
fi
