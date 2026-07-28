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
  RUN_ROOT="${FE_ROOT}/outputs/s0_runs/${RUN_ID}"
fi
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
MANIFEST="${RUN_ROOT}/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  printf >&2 'Missing S0 run manifest: %s\n' "${MANIFEST}"
  exit 3
fi

MANIFEST_RUN_ID="$(jq -er '.run_id | strings' "${MANIFEST}")"
MANIFEST_SESSION="$(jq -er '.tmux_session | strings' "${MANIFEST}")"
WINDOW_PREFIX="$(jq -er '.tmux_window_prefix | strings' "${MANIFEST}")"
MONITOR_WINDOW="$(jq -er '.tmux_monitor_window | strings' "${MANIFEST}")"
if [[ "${MANIFEST_RUN_ID}" != "${RUN_ID}" ]]; then
  printf >&2 \
    'Run manifest ID mismatch: requested=%s manifest=%s\n' \
    "${RUN_ID}" \
    "${MANIFEST_RUN_ID}"
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
  printf >&2 \
    'Run manifest belongs to tmux session %s, not %s.\n' \
    "${MANIFEST_SESSION}" \
    "${SESSION}"
  exit 3
fi

TARGET_WINDOWS=(
  "${WINDOW_PREFIX}-prepare"
  "${WINDOW_PREFIX}-b0"
  "${WINDOW_PREFIX}-b1"
  "${WINDOW_PREFIX}-b2"
  "${WINDOW_PREFIX}-b3"
  "${MONITOR_WINDOW}"
)
for name in "${TARGET_WINDOWS[@]}"; do
  if [[ "${CURRENT_WINDOW}" == "${name}" ]]; then
    printf >&2 \
      'Run the stop command from a non-S0 window; current target window is %s.\n' \
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
    tmux list-windows \
      -t "${SESSION}" \
      -F '#{window_name}|#{window_id}'
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
    if grep -Fzqx "S0_RUN_ROOT=${RUN_ROOT}" "${environment}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  done
}

printf 'S0 stop target:\n'
printf '  run id: %s\n  run root: %s\n  tmux session: %s\n' \
  "${RUN_ID}" \
  "${RUN_ROOT}" \
  "${SESSION}"
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

# Give training, dataloader, rollout-server and inference processes a chance to
# run their normal SIGINT cleanup before escalating.
for name in "${TARGET_WINDOWS[@]}"; do
  if window_id="$(tmux_window_id "${name}")"; then
    # A completed candidate can leave a dead pane, or a pane can disappear
    # between lookup and signaling. Final process/window verification below is
    # authoritative, so these expected races must not abort the stop sequence.
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
  printf >&2 'Failed to terminate S0 run processes: %s\n' "${remaining[*]}"
  exit 5
fi
for name in "${TARGET_WINDOWS[@]}"; do
  if tmux_window_id "${name}" >/dev/null; then
    printf >&2 'Failed to close S0 tmux window: %s\n' "${name}"
    exit 5
  fi
done

printf 'S0 run %s stopped; its six windows are closed.\n' "${RUN_ID}"
printf 'The permanent tmux session and all run artifacts were preserved.\n'
printf 'Remaining GPU compute processes, if any, are outside this run:\n'
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader \
  2>/dev/null || true
