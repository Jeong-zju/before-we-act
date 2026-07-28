#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${FE_ROOT}/.." && pwd)"
WORKTREE_ROOT="${S0_WORKTREE_ROOT:-${WORKSPACE_ROOT}/worktrees}"
RUN_ID="${S0_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FOCUS_MONITOR=auto
DRY_RUN=0
HF_TOKEN_INPUT=""
RESUME_LAUNCH=0

# S0 owns these defaults. A fresh clone therefore needs no path exports.
SHARED_DATA_ROOT="${FE_ROOT}/datasets/robofactory_multitask"
SHARED_ARTIFACT_ROOT="${FE_ROOT}/artifacts"
SHARED_UV_CACHE="${FE_ROOT}/.uv-cache"
SHARED_UV_ENV="${FE_ROOT}/.venv"
ROBOFACTORY_ROOT="${WORKSPACE_ROOT}/RoboFactory"
RF_PYTHON="${ROBOFACTORY_ROOT}/.venv/bin/python"

usage() {
  printf \
    'usage: %s [--run-id ID] [--focus-monitor|--no-focus-monitor] [--dry-run]\n' \
    "$0"
}

while (( $# )); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
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
SESSION="${S0_TMUX_SESSION:-<current-or-only-existing-session>}"
WINDOW_PREFIX="${S0_WINDOW_PREFIX:-${RUN_ID}}"
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"
RUN_ROOT="${S0_RUN_ROOT:-${FE_ROOT}/outputs/s0_runs/${RUN_ID}}"
READY_FILE="${RUN_ROOT}/shared.ready"
FAILED_FILE="${RUN_ROOT}/shared.failed"
HF_TOKEN_FIFO="${RUN_ROOT}/.hf_token.fifo"
source "${FE_ROOT}/scripts/s0_hf_token_fifo.sh"

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
CANDIDATE_WINDOWS=(
  "${WINDOW_PREFIX}-b0"
  "${WINDOW_PREFIX}-b1"
  "${WINDOW_PREFIX}-b2"
  "${WINDOW_PREFIX}-b3"
)

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
  for name in git tmux jq nvidia-smi python3 df sha256sum flock; do
    if ! command -v "${name}" >/dev/null; then
      printf >&2 'Missing required system command: %s\n' "${name}"
      exit 3
    fi
  done
}

resolve_existing_tmux_session() {
  local requested="${S0_TMUX_SESSION:-}"
  local sessions=()
  if [[ -n "${TMUX:-}" ]]; then
    SESSION="$(tmux display-message -p '#S')"
    return
  fi
  if [[ -n "${requested}" ]]; then
    if ! tmux has-session -t "${requested}" 2>/dev/null; then
      printf >&2 'Requested tmux session does not exist: %s\n' "${requested}"
      exit 3
    fi
    SESSION="${requested}"
    return
  fi
  mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null || true)
  if (( ${#sessions[@]} != 1 )); then
    printf >&2 \
      'Run inside the permanent tmux session, or ensure exactly one session exists; found %d.\n' \
      "${#sessions[@]}"
    exit 3
  fi
  SESSION="${sessions[0]}"
}

assert_s0_windows_absent() {
  local desired
  for desired in \
    "${PREPARE_WINDOW}" \
    "${CANDIDATE_WINDOWS[@]}" \
    "${MONITOR_WINDOW}"; do
    if tmux_window_id "${desired}" >/dev/null; then
      printf >&2 \
        'Tmux window already exists in session %s: %s\n' \
        "${SESSION}" "${desired}"
      exit 3
    fi
  done
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
    tmux list-windows \
      -t "${SESSION}" \
      -F '#{window_name}|#{window_id}'
  )
  return 1
}

validate_resume_manifest() {
  local manifest="${RUN_ROOT}/run_manifest.json"
  if [[ ! -f "${manifest}" ]]; then
    printf >&2 \
      'Run root exists without a resumable manifest: %s\n' \
      "${RUN_ROOT}"
    exit 3
  fi
  if ! jq -e \
    --arg run_id "${RUN_ID}" \
    --arg session "${SESSION}" \
    --arg prefix "${WINDOW_PREFIX}" \
    '
      .run_id == $run_id and
      .tmux_session == $session and
      .tmux_window_prefix == $prefix
    ' \
    "${manifest}" \
    >/dev/null; then
    printf >&2 \
      'Existing run manifest does not match this run/session/window prefix: %s\n' \
      "${manifest}"
    exit 3
  fi
  RESUME_LAUNCH=1
  printf 'Resuming partially created S0 windows for run %s.\n' "${RUN_ID}"
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
  local bootstrap="${WORKSPACE_ROOT}/.s0-tools/uv"
  printf 'uv was not found; installing pinned uv 0.9.26 into %s\n' "${bootstrap}"
  if [[ ! -x "${bootstrap}/bin/uv" ]]; then
    python3 -m venv "${bootstrap}"
    "${bootstrap}/bin/python" -m pip install \
      --disable-pip-version-check \
      'uv==0.9.26'
  fi
  PATH="${bootstrap}/bin:${PATH}"
  command -v uv >/dev/null
}

prompt_hf_token() {
  if [[ ! -r /dev/tty ]]; then
    printf >&2 \
      'An interactive terminal is required to read the Hugging Face token.\n'
    exit 3
  fi
  printf 'Hugging Face read token (input hidden): ' >/dev/tty
  IFS= read -r -s HF_TOKEN_INPUT </dev/tty
  printf '\n' >/dev/tty
  if [[ "${HF_TOKEN_INPUT}" != hf_* || \
    "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
    printf >&2 \
      'The Hugging Face token must start with hf_ and contain no spaces.\n'
    exit 3
  fi
}

fetch_candidate_branches() {
  local configured=()
  local desired
  local existing
  local found
  local fetches_all_branches=0
  local branch
  mapfile -t configured < <(
    git -C "${FE_ROOT}" config --get-all remote.origin.fetch || true
  )
  for existing in "${configured[@]}"; do
    if [[ "${existing#+}" == \
      'refs/heads/*:refs/remotes/origin/*' ]]; then
      fetches_all_branches=1
      break
    fi
  done
  if (( fetches_all_branches == 0 )); then
    for branch in "${BRANCHES[@]}"; do
      desired="+refs/heads/${branch}:refs/remotes/origin/${branch}"
      found=0
      for existing in "${configured[@]}"; do
        if [[ "${existing}" == "${desired}" || \
          "${existing}" == "${desired#+}" ]]; then
          found=1
          break
        fi
      done
      if (( found == 0 )); then
        git -C "${FE_ROOT}" config --add remote.origin.fetch "${desired}"
        configured+=("${desired}")
      fi
    done
  fi
  printf 'Fetching the four immutable S0 candidate branches from origin.\n'
  git -C "${FE_ROOT}" fetch --no-tags origin
  for branch in "${BRANCHES[@]}"; do
    git -C "${FE_ROOT}" show-ref \
      --verify --quiet "refs/remotes/origin/${branch}"
    if ! git -C "${FE_ROOT}" show-ref \
      --verify --quiet "refs/heads/${branch}"; then
      git -C "${FE_ROOT}" branch --track "${branch}" "origin/${branch}"
    fi
    if [[ "$(git -C "${FE_ROOT}" rev-parse "${branch}")" != \
      "$(git -C "${FE_ROOT}" rev-parse "origin/${branch}")" ]]; then
      if ! git -C "${FE_ROOT}" merge-base \
        --is-ancestor "${branch}" "origin/${branch}"; then
        printf >&2 \
          'Local branch %s has diverged from origin/%s; refusing an ambiguous run.\n' \
          "${branch}" "${branch}"
        exit 3
      fi
      printf 'Candidate branch %s will be fast-forwarded to origin/%s.\n' \
        "${branch}" "${branch}"
    fi
  done
}

candidate_command() {
  local gpu="$1"
  local command
  command="$(shell_join \
    env \
    -u HF_TOKEN \
    "PATH=${PATH}" \
    "GPU_INDEX=${gpu}" \
    "S0_RUN_ID=${RUN_ID}" \
    "S0_RUN_ROOT=${RUN_ROOT}" \
    "S0_READY_FILE=${READY_FILE}" \
    "S0_FAILED_FILE=${FAILED_FILE}" \
    "S0_BASE_REPO=${FE_ROOT}" \
    "S0_GATE_EPISODES=${S0_GATE_EPISODES:-20}" \
    "S0_GATE_SEED_START=${S0_GATE_SEED_START:-900}" \
    "S0_UV_CACHE_DIR=${SHARED_UV_CACHE}" \
    "S0_UV_ENV=${SHARED_UV_ENV}" \
    "S0_ROBOFACTORY_ROOT=${ROBOFACTORY_ROOT}" \
    "S0_RF_PYTHON=${RF_PYTHON}" \
    bash scripts/run_s0_candidate.sh
  )"
  printf '%s' "${command}"
}

print_plan() {
  printf 'S0 parent: %s @ %s\n' \
    "$(git -C "${FE_ROOT}" branch --show-current)" \
    "$(git -C "${FE_ROOT}" rev-parse HEAD)"
  printf 'Run root: %s\nExisting tmux session: %s\n' "${RUN_ROOT}" "${SESSION}"
  printf 'Window prefix: %s\n' "${WINDOW_PREFIX}"
  printf 'Shared dataset: %s\nShared artifacts: %s\n' \
    "${SHARED_DATA_ROOT}" "${SHARED_ARTIFACT_ROOT}"
  printf 'Shared uv env/cache: %s | %s\nRoboFactory: %s\n' \
    "${SHARED_UV_ENV}" "${SHARED_UV_CACHE}" "${ROBOFACTORY_ROOT}"
  for index in "${!CANDIDATES[@]}"; do
    printf '%s GPU%s  %-43s  %s/%s  window=%s\n' \
      "${CANDIDATES[index]}" "${index}" "${BRANCHES[index]}" \
      "${WORKTREE_ROOT}" "${WORKTREE_NAMES[index]}" \
      "${CANDIDATE_WINDOWS[index]}"
  done
}

if (( DRY_RUN )); then
  print_plan
  printf \
    '\nDry run: no worktrees, files, tmux windows or GPU jobs were changed.\n'
  printf 'Tmux: will reuse the current or only existing permanent session.\n'
  printf 'HF token: will be requested with hidden interactive input.\n'
  printf 'Branches: will be fetched from origin and tracked locally when absent.\n'
  printf 'prepare: %s\n' "$(shell_join \
    env \
    -u HF_TOKEN \
    "PATH=${PATH}" \
    GPU_INDEX=0 \
    "S0_RUN_ROOT=${RUN_ROOT}" \
    "S0_READY_FILE=${READY_FILE}" \
    "S0_FAILED_FILE=${FAILED_FILE}" \
    "S0_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" \
    "UV_CACHE_DIR=${SHARED_UV_CACHE}" \
    "UV_PROJECT_ENVIRONMENT=${SHARED_UV_ENV}" \
    "ROBOFACTORY_ROOT=${ROBOFACTORY_ROOT}" \
    "RF_PYTHON=${RF_PYTHON}" \
    bash scripts/prepare_s0_shared.sh
  )"
  for index in "${!CANDIDATES[@]}"; do
    printf '%s: %s\n' "${CANDIDATES[index]}" "$(candidate_command "${index}")"
  done
  exit 0
fi

require_commands
resolve_existing_tmux_session
if [[ -e "${RUN_ROOT}" ]]; then
  validate_resume_manifest
else
  assert_s0_windows_absent
fi
print_plan
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
fetch_candidate_branches
ensure_uv

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
  if [[ "$(git -C "${path}" rev-parse HEAD)" != \
    "$(git -C "${FE_ROOT}" rev-parse "origin/${branch}")" ]]; then
    if [[ -n "$(git -C "${path}" status --porcelain --untracked-files=no)" ]]; then
      printf >&2 \
        'Refusing to fast-forward candidate worktree with tracked changes: %s\n' \
        "${path}"
      exit 3
    fi
    git -C "${path}" merge --ff-only "origin/${branch}"
  fi
  test -f "${path}/experiments/wam_flow/s0/candidate.env"
  test -f "${path}/experiments/wam_flow/s0/candidate_card.yaml"
  WORKTREES+=("${path}")
done

START_PREPARE=0
if (( RESUME_LAUNCH )); then
  if [[ -f "${FAILED_FILE}" ]]; then
    printf >&2 \
      'Shared preparation already failed; inspect window %s and use a new run id after fixing it.\n' \
      "${PREPARE_WINDOW}"
    exit 4
  elif [[ -f "${READY_FILE}" ]]; then
    printf 'Shared preparation is already complete; reusing it.\n'
  elif PREPARE_WINDOW_ID="$(tmux_window_id "${PREPARE_WINDOW}")"; then
    if [[ "$(tmux display-message -p -t "${PREPARE_WINDOW_ID}" '#{pane_dead}')" \
      == "1" ]]; then
      printf >&2 \
        'Prepare window exited without ready/failed evidence: %s\n' \
        "${PREPARE_WINDOW}"
      exit 4
    fi
    printf 'Shared preparation is still running in window %s; preserving it.\n' \
      "${PREPARE_WINDOW}"
  else
    printf 'Prepare window is missing; recreating it for this run.\n'
    START_PREPARE=1
  fi
else
  prompt_hf_token
  init_arguments=(
    "${FE_ROOT}/scripts/s0_runtime.py" init
    --run-root "${RUN_ROOT}"
    --run-id "${RUN_ID}"
    --session "${SESSION}"
    --window-prefix "${WINDOW_PREFIX}"
    --monitor-window "${MONITOR_WINDOW}"
    --base-repo "${FE_ROOT}"
  )
  for index in "${!CANDIDATES[@]}"; do
    init_arguments+=(--worktree "${CANDIDATES[index]}=${WORKTREES[index]}")
  done
  python3 "${init_arguments[@]}"
  START_PREPARE=1
fi

if (( START_PREPARE )); then
  if (( RESUME_LAUNCH )); then
    prompt_hf_token
  fi
  s0_prepare_hf_token_fifo
  trap s0_cleanup_hf_secret EXIT

  prepare_command="$(shell_join \
    env \
    -u HF_TOKEN \
    "PATH=${PATH}" \
    GPU_INDEX=0 \
    "S0_RUN_ROOT=${RUN_ROOT}" \
    "S0_READY_FILE=${READY_FILE}" \
    "S0_FAILED_FILE=${FAILED_FILE}" \
    "S0_HF_TOKEN_FIFO=${HF_TOKEN_FIFO}" \
    "UV_CACHE_DIR=${SHARED_UV_CACHE}" \
    "UV_PROJECT_ENVIRONMENT=${SHARED_UV_ENV}" \
    "ROBOFACTORY_ROOT=${ROBOFACTORY_ROOT}" \
    "RF_PYTHON=${RF_PYTHON}" \
    bash scripts/prepare_s0_shared.sh
  )"
  create_persistent_window \
    "${PREPARE_WINDOW}" \
    "${FE_ROOT}" \
    "${prepare_command}" \
    >/dev/null

  # The secret crosses into the prepare window through a mode-0600 FIFO. It is
  # never exported by the launcher and never appears in a tmux command or argv.
  s0_deliver_hf_token
  trap - EXIT
fi

for index in "${!CANDIDATES[@]}"; do
  if tmux_window_id "${CANDIDATE_WINDOWS[index]}" >/dev/null; then
    printf 'Preserving existing candidate window %s.\n' \
      "${CANDIDATE_WINDOWS[index]}"
  else
    create_persistent_window \
      "${CANDIDATE_WINDOWS[index]}" \
      "${WORKTREES[index]}" \
      "$(candidate_command "${index}")" \
      >/dev/null
  fi
done
monitor_command="$(shell_join \
  python3 scripts/s0_runtime.py monitor \
  --run-root "${RUN_ROOT}" \
  --interval 5
)"
if MONITOR_WINDOW_ID="$(tmux_window_id "${MONITOR_WINDOW}")"; then
  printf 'Preserving existing monitor window %s.\n' "${MONITOR_WINDOW}"
else
  MONITOR_WINDOW_ID="$(
    create_persistent_window "${MONITOR_WINDOW}" "${FE_ROOT}" "${monitor_command}"
  )"
fi

printf '\nS0 windows are running in the existing permanent tmux session %s.\n' \
  "${SESSION}"
printf 'Monitor window: %s (target %s)\n' \
  "${MONITOR_WINDOW}" "${MONITOR_WINDOW_ID}"
printf 'Switch monitor: tmux select-window -t %s:%s\n' \
  "${SESSION}" "${MONITOR_WINDOW}"
printf 'Monitor once: python3 %s/scripts/s0_runtime.py monitor --once --run-root %s\n' \
  "${FE_ROOT}" "${RUN_ROOT}"
if [[ "${FOCUS_MONITOR}" == "yes" || \
  ( "${FOCUS_MONITOR}" == "auto" && -n "${TMUX:-}" ) ]]; then
  tmux select-window -t "${MONITOR_WINDOW_ID}"
fi
