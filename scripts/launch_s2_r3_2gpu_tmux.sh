#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S2_R3_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S2_R3_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${S2_R3_RUN_ROOT:-${FE_ROOT}/outputs/s2_r3_runs/${RUN_ID}}"
FOCUS_MONITOR=auto
DRY_RUN=0
HF_TOKEN_INPUT=""

SHARED_UV_CACHE="${FE_ROOT}/.uv-cache"
SHARED_UV_ENV="${FE_ROOT}/.venv"
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
source "${FE_ROOT}/scripts/s0_hf_token_fifo.sh"

CANDIDATES=(W0 W1)
BRANCHES=(
  s2/r3-w0-action-independent
  s2/r3-w1-action-conditioned
)
WORKTREE_NAMES=(
  s2-r3-w0-action-independent
  s2-r3-w1-action-conditioned
)
CONFIG_REL="configs/wam_flow/s2_r3_local_future.yaml"
SESSION="${S2_R3_TMUX_SESSION:-<current-or-only-existing-session>}"
WINDOW_PREFIX="${S2_R3_WINDOW_PREFIX:-${RUN_ID}}"
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
CANDIDATE_WINDOWS=("${WINDOW_PREFIX}-w0" "${WINDOW_PREFIX}-w1")
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"

usage() {
  printf \
    'usage: %s [--run-id ID] [--focus-monitor|--no-focus-monitor] [--dry-run]\n' \
    "$0"
}

while (( $# )); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      RUN_ROOT="${S2_R3_RUN_ROOT:-${FE_ROOT}/outputs/s2_r3_runs/${RUN_ID}}"
      READY_FILE="${RUN_ROOT}/shared.ready"
      FAILED_FILE="${RUN_ROOT}/shared.failed"
      HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
      WINDOW_PREFIX="${S2_R3_WINDOW_PREFIX:-${RUN_ID}}"
      PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
      CANDIDATE_WINDOWS=("${WINDOW_PREFIX}-w0" "${WINDOW_PREFIX}-w1")
      MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"
      shift 2
      ;;
    --focus-monitor)
      FOCUS_MONITOR=yes
      shift
      ;;
    --no-focus-monitor)
      FOCUS_MONITOR=no
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

shell_join() {
  local result=""
  local value
  for value in "$@"; do
    printf -v result '%s%q ' "${result}" "${value}"
  done
  printf '%s' "${result% }"
}

require_commands() {
  local name
  for name in git tmux jq nvidia-smi python3 df sha256sum flock find sort grep; do
    command -v "${name}" >/dev/null || {
      printf >&2 'Missing required command: %s\n' "${name}"
      exit 3
    }
  done
}

resolve_existing_tmux_session() {
  local requested="${S2_R3_TMUX_SESSION:-}"
  local sessions=()
  if [[ -n "${TMUX:-}" ]]; then
    SESSION="$(tmux display-message -p '#S')"
    return
  fi
  if [[ -n "${requested}" ]]; then
    tmux has-session -t "${requested}" 2>/dev/null || {
      printf >&2 'Requested tmux session does not exist: %s\n' "${requested}"
      exit 3
    }
    SESSION="${requested}"
    return
  fi
  mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null || true)
  if (( ${#sessions[@]} != 1 )); then
    printf >&2 \
      'Run inside the permanent tmux session, or leave exactly one session; found %d.\n' \
      "${#sessions[@]}"
    exit 3
  fi
  SESSION="${sessions[0]}"
}

tmux_window_id() {
  local desired="$1"
  local observed
  local window_id
  while IFS='|' read -r observed window_id; do
    if [[ "${observed}" == "${desired}" ]]; then
      printf '%s' "${window_id}"
      return 0
    fi
  done < <(
    tmux list-windows -t "${SESSION}" -F '#{window_name}|#{window_id}'
  )
  return 1
}

assert_windows_absent() {
  local desired
  for desired in \
    "${PREPARE_WINDOW}" \
    "${CANDIDATE_WINDOWS[@]}" \
    "${MONITOR_WINDOW}"; do
    if tmux_window_id "${desired}" >/dev/null; then
      printf >&2 'Tmux window already exists: %s\n' "${desired}"
      exit 3
    fi
  done
}

create_persistent_window() {
  local name="$1"
  local directory="$2"
  local command="$3"
  local window_id
  window_id="$(
    tmux new-window \
      -d \
      -P \
      -F '#{window_id}' \
      -t "${SESSION}:" \
      -n "${name}" \
      -c "${directory}" \
      "${command}"
  )"
  tmux set-option -w -t "${window_id}" remain-on-exit on
  tmux set-option -w -t "${window_id}" history-limit 200000
  printf '%s' "${window_id}"
}

ensure_uv() {
  if command -v uv >/dev/null; then
    return
  fi
  local bootstrap="${WORKSPACE_ROOT}/.s2-r3-tools/uv"
  printf 'uv not found; installing pinned uv 0.9.26 into %s\n' "${bootstrap}"
  if [[ ! -x "${bootstrap}/bin/uv" ]]; then
    python3 -m venv "${bootstrap}"
    "${bootstrap}/bin/python" -m pip install \
      --disable-pip-version-check \
      'uv==0.9.26'
  fi
  PATH="${bootstrap}/bin:${PATH}"
}

prompt_hf_token() {
  if [[ ! -r /dev/tty ]]; then
    printf >&2 'An interactive terminal is required for the Hugging Face token.\n'
    exit 3
  fi
  printf 'Hugging Face read token (input hidden): ' >/dev/tty
  IFS= read -r -s HF_TOKEN_INPUT </dev/tty
  printf '\n' >/dev/tty
  if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
    printf >&2 'The token must start with hf_ and contain no spaces.\n'
    exit 3
  fi
}

fetch_candidate_branches() {
  local configured=()
  local desired
  local existing
  local branch
  local all_branches=0
  mapfile -t configured < <(
    git -C "${FE_ROOT}" config --get-all remote.origin.fetch || true
  )
  for existing in "${configured[@]}"; do
    if [[ "${existing#+}" == 'refs/heads/*:refs/remotes/origin/*' ]]; then
      all_branches=1
      break
    fi
  done
  if (( all_branches == 0 )); then
    for branch in "${BRANCHES[@]}"; do
      desired="+refs/heads/${branch}:refs/remotes/origin/${branch}"
      if ! printf '%s\n' "${configured[@]}" | grep -Fxq "${desired}"; then
        git -C "${FE_ROOT}" config --add remote.origin.fetch "${desired}"
        configured+=("${desired}")
      fi
    done
  fi
  git -C "${FE_ROOT}" fetch --no-tags origin
  for branch in "${BRANCHES[@]}"; do
    git -C "${FE_ROOT}" show-ref \
      --verify --quiet "refs/remotes/origin/${branch}" || {
      printf >&2 'Missing remote S2-R3 branch: origin/%s\n' "${branch}"
      exit 3
    }
    if ! git -C "${FE_ROOT}" show-ref --verify --quiet "refs/heads/${branch}"; then
      git -C "${FE_ROOT}" branch --track "${branch}" "origin/${branch}"
    fi
  done
}

candidate_command() {
  local gpu="$1"
  shell_join \
    env \
    -u HF_TOKEN \
    "PATH=${PATH}" \
    "GPU_INDEX=${gpu}" \
    "S2_R3_RUN_ID=${RUN_ID}" \
    "S2_R3_RUN_ROOT=${RUN_ROOT}" \
    "S2_R3_READY_FILE=${READY_FILE}" \
    "S2_R3_FAILED_FILE=${FAILED_FILE}" \
    "S2_R3_BASE_REPO=${FE_ROOT}" \
    "S2_R3_UV_CACHE_DIR=${SHARED_UV_CACHE}" \
    "S2_R3_UV_ENV=${SHARED_UV_ENV}" \
    bash scripts/run_s2_r3_candidate.sh
}

print_plan() {
  printf 'S2-R3 parent: %s @ %s\n' \
    "$(git -C "${FE_ROOT}" branch --show-current 2>/dev/null || printf '?')" \
    "$(git -C "${FE_ROOT}" rev-parse HEAD 2>/dev/null || printf '?')"
  printf 'Run root: %s\nExisting tmux session: %s\n' "${RUN_ROOT}" "${SESSION}"
  printf 'Shared five-task dataset: %s\n' \
    "${FE_ROOT}/datasets/robofactory_multitask"
  printf 'Shared DINO/PCA/Flow artifacts: %s\n' "${FE_ROOT}/artifacts"
  local index
  for index in "${!CANDIDATES[@]}"; do
    printf '%s GPU%s  %-35s worktree=%s/%s window=%s\n' \
      "${CANDIDATES[index]}" "${index}" "${BRANCHES[index]}" \
      "${WORKTREE_ROOT}" "${WORKTREE_NAMES[index]}" \
      "${CANDIDATE_WINDOWS[index]}"
  done
}

if (( DRY_RUN )); then
  print_plan
  printf '\nDry run: no files, worktrees, tmux windows or GPU jobs were changed.\n'
  printf 'Tmux: reuses the permanent session and never kills/exits it.\n'
  printf 'Data: one five-task copy; both branches link the same datasets/artifacts.\n'
  printf 'HF datasets: S0 mode (Xet enabled, default 8 workers, five bounded attempts).\n'
  printf 'HF DINO: Xet disabled, one worker; all downloads reuse the final local-dir.\n'
  printf 'HF token: hidden input delivered through a mode-0600 FIFO.\n'
  printf 'Monitor: program, status, heartbeat, train/validation progress, GPU, R3 gate.\n'
  exit 0
fi

require_commands
resolve_existing_tmux_session
assert_windows_absent
if [[ -e "${RUN_ROOT}" ]]; then
  printf >&2 'Run root already exists; choose a new --run-id: %s\n' "${RUN_ROOT}"
  exit 3
fi
if [[ "$(git -C "${FE_ROOT}" branch --show-current)" != "feat/model-improvements" ]]; then
  printf >&2 'Launch S2-R3 from feat/model-improvements.\n'
  exit 3
fi
if [[ -n "$(git -C "${FE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  printf >&2 'Refusing launch with tracked changes in the parent worktree.\n'
  exit 3
fi
GPU_COUNT="$(nvidia-smi -L | sed -n '$=')"
if (( GPU_COUNT != 2 )); then
  printf >&2 'S2-R3 requires exactly two visible GPUs; found %d.\n' "${GPU_COUNT}"
  exit 3
fi
print_plan
fetch_candidate_branches
ensure_uv
mkdir -p "${WORKTREE_ROOT}"
WORKTREES=()
for index in "${!CANDIDATES[@]}"; do
  branch="${BRANCHES[index]}"
  path="${WORKTREE_ROOT}/${WORKTREE_NAMES[index]}"
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
  if [[ "$(git -C "${path}" rev-parse HEAD)" != \
        "$(git -C "${FE_ROOT}" rev-parse "origin/${branch}")" ]]; then
    if [[ -n "$(git -C "${path}" status --porcelain --untracked-files=no)" ]]; then
      printf >&2 'Refusing to update dirty candidate worktree: %s\n' "${path}"
      exit 3
    fi
    git -C "${path}" merge --ff-only "origin/${branch}"
  fi
  test -f "${path}/experiments/wam_flow/s2_r3/candidate.env"
  test -f "${path}/experiments/wam_flow/s2_r3/candidate_card.yaml"
  test -f "${path}/${CONFIG_REL}"
  WORKTREES+=("${path}")
done
(
  cd "${FE_ROOT}"
  uv run --frozen python scripts/validate_s2_r3_branch_pair.py \
    --w0-config "${WORKTREES[0]}/${CONFIG_REL}" \
    --w1-config "${WORKTREES[1]}/${CONFIG_REL}"
)

prompt_hf_token
python3 "${FE_ROOT}/scripts/s2_r3_runtime.py" init \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --session "${SESSION}" \
  --window-prefix "${WINDOW_PREFIX}" \
  --monitor-window "${MONITOR_WINDOW}" \
  --base-repo "${FE_ROOT}" \
  --worktree "W0=${WORKTREES[0]}" \
  --worktree "W1=${WORKTREES[1]}"

s0_prepare_hf_token_fifo
trap s0_cleanup_hf_secret EXIT
prepare_command="$(shell_join \
  env \
  -u HF_TOKEN \
  "PATH=${PATH}" \
  "S2_R3_RUN_ROOT=${RUN_ROOT}" \
  "S2_R3_READY_FILE=${READY_FILE}" \
  "S2_R3_FAILED_FILE=${FAILED_FILE}" \
  "S2_R3_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" \
  "S2_R3_W0_CONFIG=${WORKTREES[0]}/${CONFIG_REL}" \
  "S2_R3_FLOW_CHECKPOINT=${S2_R3_FLOW_CHECKPOINT:-}" \
  "UV_CACHE_DIR=${SHARED_UV_CACHE}" \
  "UV_PROJECT_ENVIRONMENT=${SHARED_UV_ENV}" \
  bash scripts/prepare_s2_r3_shared.sh
)"
create_persistent_window \
  "${PREPARE_WINDOW}" "${FE_ROOT}" "${prepare_command}" >/dev/null
s0_deliver_hf_token
trap - EXIT

for index in "${!CANDIDATES[@]}"; do
  create_persistent_window \
    "${CANDIDATE_WINDOWS[index]}" \
    "${WORKTREES[index]}" \
    "$(candidate_command "${index}")" \
    >/dev/null
done
monitor_command="$(shell_join \
  python3 scripts/s2_r3_runtime.py monitor \
  --run-root "${RUN_ROOT}" \
  --interval 5
)"
MONITOR_WINDOW_ID="$(
  create_persistent_window \
    "${MONITOR_WINDOW}" "${FE_ROOT}" "${monitor_command}"
)"

printf '\nS2-R3 is running inside permanent tmux session %s.\n' "${SESSION}"
printf 'Monitor: tmux select-window -t %s:%s\n' "${SESSION}" "${MONITOR_WINDOW}"
printf 'One-shot: python3 scripts/s2_r3_runtime.py monitor --once --run-root %s\n' \
  "${RUN_ROOT}"
if [[ "${FOCUS_MONITOR}" == "yes" || \
      ( "${FOCUS_MONITOR}" == "auto" && -n "${TMUX:-}" ) ]]; then
  tmux select-window -t "${MONITOR_WINDOW_ID}"
fi
