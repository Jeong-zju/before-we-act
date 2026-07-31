#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
RUN_ROOT=""
GRACE_SECONDS=10
DRY_RUN=0

usage() {
  printf 'usage: %s --run-id ID [--run-root PATH] [--grace-seconds N] [--dry-run]\n' "$0"
}
while (( $# )); do
  case "$1" in
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?--run-root requires a value}"; shift 2 ;;
    --grace-seconds) GRACE_SECONDS="${2:?--grace-seconds requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || \
      ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${S2_R4_SHARED_ROOT:-/workspace/fe-pc-wam}/outputs/s2_r4_hybrid/${RUN_ID}"
fi
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
MANIFEST="${RUN_ROOT}/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  printf >&2 'Missing S2-R4 hybrid manifest: %s\n' "${MANIFEST}"
  exit 3
fi
SESSION="$(jq -er '.tmux_session' "${MANIFEST}")" || exit 3
WINDOW_PREFIX="$(jq -er '.tmux_window_prefix' "${MANIFEST}")" || exit 3
MONITOR_WINDOW="$(jq -er '.tmux_monitor_window' "${MANIFEST}")" || exit 3
MANIFEST_RUN_ID="$(jq -er '.run_id' "${MANIFEST}")" || exit 3
if [[ "${MANIFEST_RUN_ID}" != "${RUN_ID}" ]]; then
  printf >&2 'Run id mismatch in manifest.\n'
  exit 3
fi
TARGET_WINDOWS=("${WINDOW_PREFIX}-prepare" "${WINDOW_PREFIX}-evaluate" "${MONITOR_WINDOW}")
if [[ -n "${TMUX:-}" ]]; then
  CURRENT_SESSION="$(tmux display-message -p '#S')"
  CURRENT_WINDOW="$(tmux display-message -p '#W')"
else
  mapfile -t SESSIONS < <(tmux list-sessions -F '#S' 2>/dev/null)
  if (( ${#SESSIONS[@]} != 1 )); then
    printf >&2 'Expected one permanent tmux session.\n'
    exit 3
  fi
  CURRENT_SESSION="${SESSIONS[0]}"
  CURRENT_WINDOW=""
fi
if [[ "${CURRENT_SESSION}" != "${SESSION}" ]]; then
  printf >&2 'Run belongs to tmux %s, current is %s.\n' "${SESSION}" "${CURRENT_SESSION}"
  exit 3
fi
for window_name in "${TARGET_WINDOWS[@]}"; do
  if [[ "${CURRENT_WINDOW}" == "${window_name}" ]]; then
    printf >&2 'Stop from a base window, not %s.\n' "${CURRENT_WINDOW}"
    exit 3
  fi
done

run_process_ids() {
  local environment pid
  for environment in /proc/[0-9]*/environ; do
    [[ -r "${environment}" ]] || continue
    pid="${environment#/proc/}"
    pid="${pid%/environ}"
    [[ "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
    if grep -Fzqx "S2_R4_HYBRID_RUN_ROOT=${RUN_ROOT}" "${environment}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  done
}

printf 'S2-R4 hybrid stop target: run=%s root=%s tmux=%s\n' \
  "${RUN_ID}" "${RUN_ROOT}" "${SESSION}"
printf 'Preserve: shared data/cache/artifacts, source checkpoints, logs, manifest, diagnostic.\n'
if (( DRY_RUN )); then
  printf 'Dry run complete; no process/window/session changed.\n'
  exit 0
fi
for window_name in "${TARGET_WINDOWS[@]}"; do
  tmux send-keys -t "${SESSION}:${window_name}" C-c 2>/dev/null
done
for (( second=0; second<GRACE_SECONDS; second++ )); do
  mapfile -t REMAINING < <(run_process_ids)
  (( ${#REMAINING[@]} == 0 )) && break
  sleep 1
done
mapfile -t REMAINING < <(run_process_ids)
if (( ${#REMAINING[@]} > 0 )); then
  kill -TERM "${REMAINING[@]}" 2>/dev/null
  sleep 2
fi
mapfile -t REMAINING < <(run_process_ids)
if (( ${#REMAINING[@]} > 0 )); then
  kill -KILL "${REMAINING[@]}" 2>/dev/null
fi
for window_name in "${TARGET_WINDOWS[@]}"; do
  tmux kill-window -t "${SESSION}:${window_name}" 2>/dev/null
done
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf >&2 'Permanent tmux session was lost unexpectedly.\n'
  exit 5
fi
printf 'S2-R4 hybrid stopped; permanent tmux and all data/results are preserved.\n'
