#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r11-four-way-v1
PYTHON=/venv/robofactory-act/bin/python
SELECTION=""
GRACE=300
GRACEFUL=0
DRY_RUN=0
while (($#)); do
  case "$1" in
    --all) SELECTION=all; shift ;;
    --candidate) SELECTION="${2^^}"; shift 2 ;;
    --graceful) GRACEFUL=1; shift ;;
    --grace-seconds) GRACE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$SELECTION" == all || "$SELECTION" =~ ^[A-D]$ ]] || {
  printf 'use --all or --candidate A|B|C|D\n' >&2; exit 2;
}
((GRACEFUL)) || { printf 'explicit --graceful is required\n' >&2; exit 2; }
[[ "$GRACE" =~ ^[0-9]+$ ]] || { printf 'grace seconds must be an integer\n' >&2; exit 2; }
if [[ "$SELECTION" == all ]]; then SELECTED=(A B C D); else SELECTED=("$SELECTION"); fi
MANIFEST="$RUN_ROOT/run_manifest.json"
[[ -f "$MANIFEST" && "$(jq -r '.stage' "$MANIFEST")" == R11 ]] || {
  printf 'immutable R11 run manifest is missing or differs\n' >&2; exit 3;
}
[[ "$(realpath -m "$(jq -r '.remote.run_root' "$MANIFEST")")" == "$(realpath -m "$RUN_ROOT")" ]] || {
  printf 'run root differs from immutable manifest\n' >&2; exit 3;
}

identity_alive() {
  local pid="$1" expected="$2" observed
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$expected" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/stat" ]] || return 1
  observed="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$observed" == "$expected" ]]
}

declare -A WRAPPER_PIDS WRAPPER_STARTS CHILD_PIDS CHILD_STARTS SESSIONS
printf 'Exact R11 graceful-stop targets (artifacts are preserved):\n'
for candidate in "${SELECTED[@]}"; do
  status="$RUN_ROOT/$candidate/status/runtime.json"
  session="$(jq -r --arg candidate "$candidate" '.candidates[$candidate].tmux' "$MANIFEST")"
  SESSIONS[$candidate]="$session"
  if [[ -f "$status" ]]; then
    WRAPPER_PIDS[$candidate]="$(jq -r '.pid // 0' "$status")"
    WRAPPER_STARTS[$candidate]="$(jq -r '.pid_start_time_ticks // 0' "$status")"
    CHILD_PIDS[$candidate]="$(jq -r '.child_pid // 0' "$status")"
    CHILD_STARTS[$candidate]="$(jq -r '.child_pid_start_time_ticks // 0' "$status")"
  else
    WRAPPER_PIDS[$candidate]=0; WRAPPER_STARTS[$candidate]=0
    CHILD_PIDS[$candidate]=0; CHILD_STARTS[$candidate]=0
  fi
  printf '  %s tmux=%s wrapper=%s@%s child=%s@%s\n' "$candidate" "$session" \
    "${WRAPPER_PIDS[$candidate]}" "${WRAPPER_STARTS[$candidate]}" \
    "${CHILD_PIDS[$candidate]}" "${CHILD_STARTS[$candidate]}"
done
if ((DRY_RUN)); then
  printf 'dry-run: no signal sent and no session changed\n'
  exit 0
fi

for candidate in "${SELECTED[@]}"; do
  if identity_alive "${WRAPPER_PIDS[$candidate]}" "${WRAPPER_STARTS[$candidate]}"; then
    environment="/proc/${WRAPPER_PIDS[$candidate]}/environ"
    grep -Fzqx "BWA_R11_RUN_ROOT=$RUN_ROOT" "$environment" 2>/dev/null && \
      grep -Fzqx "BWA_R11_CANDIDATE=$candidate" "$environment" 2>/dev/null || {
        printf 'wrapper environment identity differs for %s; refusing signal\n' "$candidate" >&2
        exit 4
      }
    kill -USR1 "${WRAPPER_PIDS[$candidate]}"
  elif tmux has-session -t "${SESSIONS[$candidate]}" 2>/dev/null; then
    printf 'session exists but exact wrapper PID identity is unavailable for %s\n' "$candidate" >&2
    exit 4
  fi
done

for ((second=0; second<GRACE; second++)); do
  remaining=0
  for candidate in "${SELECTED[@]}"; do
    identity_alive "${WRAPPER_PIDS[$candidate]}" "${WRAPPER_STARTS[$candidate]}" && remaining=$((remaining + 1))
    identity_alive "${CHILD_PIDS[$candidate]}" "${CHILD_STARTS[$candidate]}" && remaining=$((remaining + 1))
  done
  ((remaining == 0)) && break
  sleep 1
done

for signal_name in TERM KILL; do
  found=0
  for candidate in "${SELECTED[@]}"; do
    targets=()
    identity_alive "${CHILD_PIDS[$candidate]}" "${CHILD_STARTS[$candidate]}" && targets+=("${CHILD_PIDS[$candidate]}")
    identity_alive "${WRAPPER_PIDS[$candidate]}" "${WRAPPER_STARTS[$candidate]}" && targets+=("${WRAPPER_PIDS[$candidate]}")
    if ((${#targets[@]})); then
      found=1
      printf 'SIG%s only to exact %s PID/start-time targets: %s\n' "$signal_name" "$candidate" "${targets[*]}"
      kill "-$signal_name" "${targets[@]}" 2>/dev/null || true
    fi
  done
  ((found == 0)) && break
  for _ in 1 2 3 4 5; do sleep 1; done
done

for candidate in "${SELECTED[@]}"; do
  identity_alive "${WRAPPER_PIDS[$candidate]}" "${WRAPPER_STARTS[$candidate]}" && {
    printf 'exact wrapper PID remained alive for %s\n' "$candidate" >&2; exit 5;
  }
  identity_alive "${CHILD_PIDS[$candidate]}" "${CHILD_STARTS[$candidate]}" && {
    printf 'exact child PID remained alive for %s\n' "$candidate" >&2; exit 5;
  }
  tmux has-session -t "${SESSIONS[$candidate]}" 2>/dev/null && tmux kill-session -t "${SESSIONS[$candidate]}"
done
printf 'Stopped only the selected R11 candidates; data, cache, logs, receipts and checkpoints remain.\n'
