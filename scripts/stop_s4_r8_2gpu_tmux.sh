#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S4_R8_RUN_ID:-}"
RUN_ROOT="${S4_R8_RUN_ROOT:-}"
GRACE_SECONDS=10
DRY_RUN=0

usage() {
  printf 'usage: %s --run-id ID [--run-root PATH] [--grace-seconds N] [--dry-run]\n' "$0"
}

fail() {
  printf >&2 'S4-R8 stop error: %s\n' "$*"
  exit 3
}

while (( $# )); do
  case "$1" in
    --run-id)
      if (( $# < 2 )); then printf >&2 '%s\n' '--run-id requires a value'; exit 2; fi
      RUN_ID="$2"; shift 2
      ;;
    --run-root)
      if (( $# < 2 )); then printf >&2 '%s\n' '--run-root requires a value'; exit 2; fi
      RUN_ROOT="$2"; shift 2
      ;;
    --grace-seconds)
      if (( $# < 2 )); then printf >&2 '%s\n' '--grace-seconds requires a value'; exit 2; fi
      GRACE_SECONDS="$2"; shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ -z "${RUN_ID}" && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        RUN_ID="$1"; shift
      else
        usage >&2; exit 2
      fi
      ;;
  esac
done

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'A valid --run-id is required.\n'
  exit 2
fi
if [[ ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  printf >&2 '%s\n' '--grace-seconds must be a non-negative integer'
  exit 2
fi

for required in tmux jq grep nvidia-smi realpath sleep kill; do
  if ! command -v "${required}" >/dev/null 2>&1; then
    fail "missing required command: ${required}"
  fi
done

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${FE_ROOT}/outputs/s4_r8_runs/${RUN_ID}"
fi
RUN_ROOT="$(realpath -m "${RUN_ROOT}")" || fail "cannot resolve run root"
MANIFEST="${RUN_ROOT}/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  fail "missing S4-R8 run manifest: ${MANIFEST}"
fi

MANIFEST_FORMAT="$(jq -er '.format_version | strings' "${MANIFEST}")" \
  || fail "invalid format_version in manifest"
MANIFEST_ROUND="$(jq -er '.round_id | strings' "${MANIFEST}")" \
  || fail "invalid round_id in manifest"
MANIFEST_RUN_ID="$(jq -er '.run_id | strings' "${MANIFEST}")" \
  || fail "invalid run_id in manifest"
MANIFEST_RUN_ROOT="$(jq -er '.run_root | strings' "${MANIFEST}")" \
  || fail "invalid run_root in manifest"
MANIFEST_SESSION="$(jq -er '.tmux_session | strings' "${MANIFEST}")" \
  || fail "invalid tmux_session in manifest"
WINDOW_PREFIX="$(jq -er '.tmux_window_prefix | strings' "${MANIFEST}")" \
  || fail "invalid tmux_window_prefix in manifest"
MONITOR_WINDOW="$(jq -er '.tmux_monitor_window | strings' "${MANIFEST}")" \
  || fail "invalid tmux_monitor_window in manifest"

if [[ "${MANIFEST_FORMAT}" != wam.robofactory.s4_r8.runtime/1 || \
      "${MANIFEST_ROUND}" != s4-r8 || "${MANIFEST_RUN_ID}" != "${RUN_ID}" ]]; then
  fail "manifest identity does not match requested S4-R8 run ${RUN_ID}"
fi
MANIFEST_RUN_ROOT="$(realpath -m "${MANIFEST_RUN_ROOT}")" \
  || fail "cannot resolve manifest run root"
if [[ "${MANIFEST_RUN_ROOT}" != "${RUN_ROOT}" ]]; then
  fail "manifest run root ${MANIFEST_RUN_ROOT} does not exactly match ${RUN_ROOT}"
fi
if [[ "${MANIFEST_SESSION}" != ssh_tmux ]]; then
  fail "manifest does not belong to the permanent ssh_tmux session"
fi
if [[ ! "${WINDOW_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  fail "unsafe tmux window prefix in manifest: ${WINDOW_PREFIX}"
fi
if [[ "${MONITOR_WINDOW}" != "${WINDOW_PREFIX}-monitor" ]]; then
  fail "unexpected monitor window in manifest: ${MONITOR_WINDOW}"
fi
if ! tmux has-session -t "${MANIFEST_SESSION}" 2>/dev/null; then
  fail "permanent tmux session is absent: ${MANIFEST_SESSION}"
fi

if [[ -n "${TMUX:-}" ]]; then
  SESSION="$(tmux display-message -p '#S')" || fail "cannot identify current session"
  CURRENT_WINDOW="$(tmux display-message -p '#W')" || fail "cannot identify current window"
else
  SESSION="${MANIFEST_SESSION}"
  CURRENT_WINDOW=""
fi
if [[ "${SESSION}" != "${MANIFEST_SESSION}" ]]; then
  fail "run belongs to ${MANIFEST_SESSION}, not current session ${SESSION}"
fi

TARGET_WINDOWS=(
  "${WINDOW_PREFIX}-prepare"
  "${WINDOW_PREFIX}-p0"
  "${WINDOW_PREFIX}-p1"
  "${MONITOR_WINDOW}"
)
for name in "${TARGET_WINDOWS[@]}"; do
  if [[ "${CURRENT_WINDOW}" == "${name}" ]]; then
    fail "run stop from a non-S4-R8 window; current target window is ${name}"
  fi
done

tmux_window_id() {
  local desired="$1"
  local observed
  local window_id
  while IFS='|' read -r observed window_id; do
    if [[ "${observed}" == "${desired}" ]]; then
      printf '%s' "${window_id}"
      return 0
    fi
  done < <(tmux list-windows -t "${SESSION}" -F '#{window_name}|#{window_id}')
  return 1
}

run_process_ids() {
  local environment
  local pid
  local paths=(/proc/[0-9]*/environ)
  for environment in "${paths[@]}"; do
    [[ -r "${environment}" ]] || continue
    pid="${environment#/proc/}"
    pid="${pid%/environ}"
    [[ "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
    if grep -Fzqx "S4_R8_RUN_ROOT=${RUN_ROOT}" "${environment}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  done
}

printf 'S4-R8 stop target (exact run identity):\n'
printf '  run id: %s\n  run root: %s\n  tmux session: %s\n' \
  "${RUN_ID}" "${RUN_ROOT}" "${SESSION}"
for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    printf '  window: %s (%s)\n' "${name}" "${window_id}"
  else
    printf '  window: %s (already absent)\n' "${name}"
  fi
done
mapfile -t INITIAL_PIDS < <(run_process_ids)
printf '  tagged process PIDs: %s\n' "${INITIAL_PIDS[*]:-none}"

if (( DRY_RUN )); then
  printf 'Dry run: no process was signaled and no tmux window was closed.\n'
  exit 0
fi

for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    tmux send-keys -t "${window_id}" C-c 2>/dev/null || true
  fi
done

for (( second=0; second<GRACE_SECONDS; second++ )); do
  mapfile -t remaining < <(run_process_ids)
  if (( ${#remaining[@]} == 0 )); then break; fi
  sleep 1
done

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf 'Sending SIGTERM only to tagged S4-R8 processes: %s\n' "${remaining[*]}"
  kill -TERM "${remaining[@]}" 2>/dev/null || true
  for (( second=0; second<5; second++ )); do
    mapfile -t remaining < <(run_process_ids)
    if (( ${#remaining[@]} == 0 )); then break; fi
    sleep 1
  done
fi

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf 'Sending SIGKILL only to remaining tagged processes: %s\n' "${remaining[*]}"
  kill -KILL "${remaining[@]}" 2>/dev/null || true
  for (( second=0; second<5; second++ )); do
    mapfile -t remaining < <(run_process_ids)
    if (( ${#remaining[@]} == 0 )); then break; fi
    sleep 1
  done
fi

for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    tmux kill-window -t "${window_id}" 2>/dev/null || true
  fi
done

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf >&2 'Failed to terminate tagged S4-R8 processes: %s\n' "${remaining[*]}"
  exit 5
fi
for name in "${TARGET_WINDOWS[@]}"; do
  if tmux_window_id "${name}" >/dev/null; then
    printf >&2 'Failed to close S4-R8 tmux window: %s\n' "${name}"
    exit 5
  fi
done

printf 'Stopped only S4-R8 run %s and closed its four windows.\n' "${RUN_ID}"
printf 'Preserved the permanent tmux session, shared datasets/cache/artifacts, worktrees, checkpoints, resumes, logs, videos and all acceptance reports.\n'
printf 'Remaining GPU processes, if any, are outside this exact run identity:\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader 2>/dev/null || true
