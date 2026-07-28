#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
RUN_ROOT=""
GRACE_SECONDS=10
DRY_RUN=0

usage() {
  printf \
    'usage: %s --run-id ID [--run-root PATH] [--grace-seconds N] [--dry-run]\n' \
    "$0"
}

while (( $# )); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:?--run-root requires a value}"
      shift 2
      ;;
    --grace-seconds)
      GRACE_SECONDS="${2:?--grace-seconds requires a value}"
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

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'A valid --run-id is required.\n'
  exit 2
fi
if [[ ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  printf >&2 '--grace-seconds must be a non-negative integer.\n'
  exit 2
fi
for required in tmux jq grep nvidia-smi realpath sleep; do
  if ! command -v "${required}" >/dev/null; then
    printf >&2 'Missing required command: %s\n' "${required}"
    exit 3
  fi
done

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${FE_ROOT}/outputs/s1_r1_runs/${RUN_ID}"
fi
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
MANIFEST="${RUN_ROOT}/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  printf >&2 'Missing S1-R1 run manifest: %s\n' "${MANIFEST}"
  exit 3
fi

MANIFEST_RUN_ID="$(jq -er '.run_id | strings' "${MANIFEST}")"
MANIFEST_ROUND_ID="$(jq -er '.round_id | strings' "${MANIFEST}")"
MANIFEST_SESSION="$(jq -er '.tmux_session | strings' "${MANIFEST}")"
WINDOW_PREFIX="$(jq -er '.tmux_window_prefix | strings' "${MANIFEST}")"
MONITOR_WINDOW="$(jq -er '.tmux_monitor_window | strings' "${MANIFEST}")"
if [[ "${MANIFEST_RUN_ID}" != "${RUN_ID}" || "${MANIFEST_ROUND_ID}" != "s1-r1" ]]; then
  printf >&2 'Run manifest does not match requested S1-R1 run %s.\n' "${RUN_ID}"
  exit 3
fi
if [[ ! "${WINDOW_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Unsafe tmux window prefix in manifest: %q\n' "${WINDOW_PREFIX}"
  exit 3
fi
if [[ "${MONITOR_WINDOW}" != "${WINDOW_PREFIX}-monitor" ]]; then
  printf >&2 'Unexpected monitor window in manifest: %q\n' "${MONITOR_WINDOW}"
  exit 3
fi

if [[ -n "${TMUX:-}" ]]; then
  SESSION="$(tmux display-message -p '#S')"
  CURRENT_WINDOW="$(tmux display-message -p '#W')"
else
  mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null || true)
  if (( ${#sessions[@]} != 1 )); then
    printf >&2 \
      'Run inside the permanent tmux session, or leave exactly one session; found %d.\n' \
      "${#sessions[@]}"
    exit 3
  fi
  SESSION="${sessions[0]}"
  CURRENT_WINDOW=""
fi
if [[ "${SESSION}" != "${MANIFEST_SESSION}" ]]; then
  printf >&2 'Run belongs to tmux session %s, not %s.\n' \
    "${MANIFEST_SESSION}" "${SESSION}"
  exit 3
fi

TARGET_WINDOWS=(
  "${WINDOW_PREFIX}-prepare"
  "${WINDOW_PREFIX}-f0"
  "${WINDOW_PREFIX}-f1"
  "${MONITOR_WINDOW}"
)
for name in "${TARGET_WINDOWS[@]}"; do
  if [[ "${CURRENT_WINDOW}" == "${name}" ]]; then
    printf >&2 \
      'Run stop from a non-S1-R1 window; current target window is %s.\n' \
      "${CURRENT_WINDOW}"
    exit 3
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
  done < <(
    tmux list-windows -t "${SESSION}" -F '#{window_name}|#{window_id}'
  )
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
    if grep -Fzqx "S1_R1_RUN_ROOT=${RUN_ROOT}" "${environment}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  done
}

printf 'S1-R1 stop target:\n'
printf '  run id: %s\n  run root: %s\n  tmux session: %s\n' \
  "${RUN_ID}" "${RUN_ROOT}" "${SESSION}"
for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    printf '  window: %s (%s)\n' "${name}" "${window_id}"
  else
    printf '  window: %s (already absent)\n' "${name}"
  fi
done

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
  (( ${#remaining[@]} == 0 )) && break
  sleep 1
done

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf 'Sending SIGTERM to run processes: %s\n' "${remaining[*]}"
  kill -TERM "${remaining[@]}" 2>/dev/null || true
  for (( second=0; second<5; second++ )); do
    mapfile -t remaining < <(run_process_ids)
    (( ${#remaining[@]} == 0 )) && break
    sleep 1
  done
fi

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf 'Sending SIGKILL to remaining run processes: %s\n' "${remaining[*]}"
  kill -KILL "${remaining[@]}" 2>/dev/null || true
  for (( second=0; second<5; second++ )); do
    mapfile -t remaining < <(run_process_ids)
    (( ${#remaining[@]} == 0 )) && break
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
  printf >&2 'Failed to terminate S1-R1 run processes: %s\n' "${remaining[*]}"
  exit 5
fi
for name in "${TARGET_WINDOWS[@]}"; do
  if tmux_window_id "${name}" >/dev/null; then
    printf >&2 'Failed to close S1-R1 tmux window: %s\n' "${name}"
    exit 5
  fi
done

printf 'S1-R1 run %s stopped; its four windows are closed.\n' "${RUN_ID}"
printf 'The permanent tmux session, shared dataset and all run artifacts remain.\n'
printf 'Remaining GPU compute processes, if any, are outside this run:\n'
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader \
  2>/dev/null || true
