#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r12e1-current
SELECTION=all
GRACE=30
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --grace-seconds) GRACE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$GRACE" =~ ^[0-9]+$ ]] || { printf 'grace must be a nonnegative integer\n' >&2; exit 2; }
MANIFEST="$RUN_ROOT/run_manifest.json"
[[ -f "$MANIFEST" ]] || { printf 'missing run manifest: %s\n' "$MANIFEST" >&2; exit 3; }
[[ "$(jq -r '.round' "$MANIFEST")" == R12-E1 && "$(realpath -m "$(jq -r '.run_root' "$MANIFEST")")" == "$(realpath -m "$RUN_ROOT")" ]] || { printf 'R12-E1 run manifest identity mismatch\n' >&2; exit 3; }
declare -A ALIAS=( [A]=p0 [B]=p1 [C]=p2 [D]=p3 [a]=p0 [b]=p1 [c]=p2 [d]=p3 )
if [[ "$SELECTION" == all ]]; then
  SELECTED=(p0 p1 p2 p3)
else
  IFS=',' read -r -a RAW <<<"$SELECTION"; SELECTED=()
  for item in "${RAW[@]}"; do
    item="${ALIAS[$item]:-$item}"
    [[ "$item" =~ ^p[0-3]$ ]] || { printf 'invalid candidate: %s\n' "$item" >&2; exit 2; }
    [[ " ${SELECTED[*]} " != *" $item "* ]] && SELECTED+=("$item")
  done
fi
tagged_pids() {
  local candidate="$1" environment pid
  for environment in /proc/[0-9]*/environ; do
    [[ -r "$environment" ]] || continue
    pid="${environment#/proc/}"; pid="${pid%/environ}"
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    if grep -Fzqx "BWA_R12_RUN_ROOT=$RUN_ROOT" "$environment" 2>/dev/null && grep -Fzqx "BWA_R12_CANDIDATE=$candidate" "$environment" 2>/dev/null; then
      printf '%s\n' "$pid"
    fi
  done
}
printf 'Exact R12-E1 stop targets (cache/log/checkpoints are preserved):\n'
for candidate in "${SELECTED[@]}"; do
  session="$(jq -r --arg candidate "$candidate" '.tmux_sessions[$candidate]' "$MANIFEST")"
  mapfile -t pids < <(tagged_pids "$candidate")
  printf '  %s session=%s pids=%s\n' "$candidate" "$session" "${pids[*]:-none}"
done
if ((DRY_RUN)); then
  printf 'dry-run: no signal sent and no tmux session closed\n'; exit 0
fi
for candidate in "${SELECTED[@]}"; do
  session="$(jq -r --arg candidate "$candidate" '.tmux_sessions[$candidate]' "$MANIFEST")"
  tmux has-session -t "$session" 2>/dev/null && tmux send-keys -t "$session" C-c || true
done
for ((second=0; second<GRACE; second++)); do
  remaining=0
  for candidate in "${SELECTED[@]}"; do
    mapfile -t pids < <(tagged_pids "$candidate")
    remaining=$((remaining + ${#pids[@]}))
  done
  ((remaining == 0)) && break
  sleep 1
done
for signal in TERM KILL; do
  found=0
  for candidate in "${SELECTED[@]}"; do
    mapfile -t pids < <(tagged_pids "$candidate")
    if ((${#pids[@]})); then
      found=1
      printf 'sending SIG%s only to %s tagged PIDs: %s\n' "$signal" "$candidate" "${pids[*]}"
      kill "-$signal" "${pids[@]}" 2>/dev/null || true
    fi
  done
  ((found == 0)) && break
  sleep 5
done
for candidate in "${SELECTED[@]}"; do
  session="$(jq -r --arg candidate "$candidate" '.tmux_sessions[$candidate]' "$MANIFEST")"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
  state="$(jq -r '.state // "UNKNOWN"' "$RUN_ROOT/candidates/$candidate/status.json" 2>/dev/null || true)"
  if [[ "$state" != PASSED && "$state" != FAILED ]]; then
    "$PYTHON" "$ROOT/scripts/before_we_act/r12_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state STOPPED --stage stopped --program stop_r12_evolution_4gpu_tmux.sh --detail "graceful stop; latest checkpoint and all artifacts preserved" --pid 0 --child-pid 0 --exit-code 130 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  mapfile -t pids < <(tagged_pids "$candidate")
  ((${#pids[@]} == 0)) || { printf 'failed to stop exact %s PIDs: %s\n' "$candidate" "${pids[*]}" >&2; exit 5; }
done
printf 'Stopped only selected R12-E1 targets; datasets, HF cache, feature cache, logs and checkpoints remain intact.\n'
