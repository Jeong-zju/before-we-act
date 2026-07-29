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

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || \
      ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${FE_ROOT}/outputs/s2_r3_runs/${RUN_ID}"
fi
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
MANIFEST="${RUN_ROOT}/run_manifest.json"
test -f "${MANIFEST}" || {
  printf >&2 'Missing S2-R3 run manifest: %s\n' "${MANIFEST}"
  exit 3
}

MANIFEST_RUN_ID="$(jq -er '.run_id | strings' "${MANIFEST}")"
ROUND_ID="$(jq -er '.round_id | strings' "${MANIFEST}")"
SESSION="$(jq -er '.tmux_session | strings' "${MANIFEST}")"
WINDOW_PREFIX="$(jq -er '.tmux_window_prefix | strings' "${MANIFEST}")"
MONITOR_WINDOW="$(jq -er '.tmux_monitor_window | strings' "${MANIFEST}")"
if [[ "${MANIFEST_RUN_ID}" != "${RUN_ID}" || "${ROUND_ID}" != "s2-r3" ]]; then
  printf >&2 'Run manifest does not match S2-R3 run %s.\n' "${RUN_ID}"
  exit 3
fi
TARGET_WINDOWS=(
  "${WINDOW_PREFIX}-prepare"
  "${WINDOW_PREFIX}-w0"
  "${WINDOW_PREFIX}-w1"
  "${MONITOR_WINDOW}"
)

if [[ -n "${TMUX:-}" ]]; then
  CURRENT_SESSION="$(tmux display-message -p '#S')"
  CURRENT_WINDOW="$(tmux display-message -p '#W')"
else
  mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null || true)
  if (( ${#sessions[@]} != 1 )); then
    printf >&2 'Run inside the permanent tmux or leave exactly one session.\n'
    exit 3
  fi
  CURRENT_SESSION="${sessions[0]}"
  CURRENT_WINDOW=""
fi
if [[ "${CURRENT_SESSION}" != "${SESSION}" ]]; then
  printf >&2 'Run belongs to tmux session %s, not %s.\n' \
    "${SESSION}" "${CURRENT_SESSION}"
  exit 3
fi
for name in "${TARGET_WINDOWS[@]}"; do
  if [[ "${CURRENT_WINDOW}" == "${name}" ]]; then
    printf >&2 'Run stop from a non-S2-R3 base window; current=%s.\n' \
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
  for environment in /proc/[0-9]*/environ; do
    [[ -r "${environment}" ]] || continue
    pid="${environment#/proc/}"
    pid="${pid%/environ}"
    [[ "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
    if grep -Fzqx "S2_R3_RUN_ROOT=${RUN_ROOT}" "${environment}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  done
}

printf 'S2-R3 stop target: run=%s root=%s tmux=%s\n' \
  "${RUN_ID}" "${RUN_ROOT}" "${SESSION}"
for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    printf '  window: %s (%s)\n' "${name}" "${window_id}"
  else
    printf '  window: %s (already absent)\n' "${name}"
  fi
done
if (( DRY_RUN )); then
  printf 'Dry run: no process or window changed; permanent session untouched.\n'
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
  kill -TERM "${remaining[@]}" 2>/dev/null || true
  sleep 2
fi
mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  kill -KILL "${remaining[@]}" 2>/dev/null || true
fi
for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    tmux kill-window -t "${window_id}" 2>/dev/null || true
  fi
done

mapfile -t remaining < <(run_process_ids)
if (( ${#remaining[@]} > 0 )); then
  printf >&2 'Failed to terminate S2-R3 processes: %s\n' "${remaining[*]}"
  exit 5
fi
printf 'S2-R3 stopped. Permanent tmux, shared data, artifacts, checkpoints, resumes, logs and evaluations are preserved.\n'
