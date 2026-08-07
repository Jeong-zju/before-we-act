#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=""; SELECTION=all; GRACE=30; DRY_RUN=0; PYTHON=/venv/robofactory-act/bin/python
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
[[ -n "$RUN_ROOT" && -f "$RUN_ROOT/run_manifest.json" ]] || { printf 'valid --run-root is required\n' >&2; exit 2; }
[[ "$(jq -r '.round' "$RUN_ROOT/run_manifest.json")" == R14 ]] || { printf 'R14 manifest identity mismatch\n' >&2; exit 3; }
declare -A ALIAS=( [A]=p0 [B]=p1 [C]=p2 [D]=p3 [a]=p0 [b]=p1 [c]=p2 [d]=p3 )
if [[ "$SELECTION" == all ]]; then SELECTED=(p0 p1 p2 p3); else IFS=',' read -r -a RAW <<<"$SELECTION"; SELECTED=(); for item in "${RAW[@]}"; do item="${ALIAS[$item]:-$item}"; [[ "$item" =~ ^p[0-3]$ ]] || exit 2; SELECTED+=("$item"); done; fi
tagged_pids() {
  local candidate="$1" environment pid
  for environment in /proc/[0-9]*/environ; do
    [[ -r "$environment" ]] || continue; pid="${environment#/proc/}"; pid="${pid%/environ}"
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    grep -Fzqx "BWA_R14_RUN_ROOT=$RUN_ROOT" "$environment" 2>/dev/null && grep -Fzqx "BWA_R14_CANDIDATE=$candidate" "$environment" 2>/dev/null && printf '%s\n' "$pid"
  done
}
printf 'Exact R14 stop targets (all data/logs remain intact):\n'
for candidate in "${SELECTED[@]}"; do mapfile -t pids < <(tagged_pids "$candidate"); printf '  %s session=bwa-r14-%s pids=%s\n' "$candidate" "$candidate" "${pids[*]:-none}"; done
if ((DRY_RUN)); then printf 'dry-run: no signal/session change\n'; exit 0; fi
for candidate in "${SELECTED[@]}"; do tmux has-session -t "bwa-r14-$candidate" 2>/dev/null && tmux send-keys -t "bwa-r14-$candidate" C-c || true; done
for ((second=0; second<GRACE; second++)); do remaining=0; for candidate in "${SELECTED[@]}"; do mapfile -t pids < <(tagged_pids "$candidate"); remaining=$((remaining+${#pids[@]})); done; ((remaining==0)) && break; sleep 1; done
for signal in TERM KILL; do targets=(); for candidate in "${SELECTED[@]}"; do mapfile -t pids < <(tagged_pids "$candidate"); targets+=("${pids[@]}"); done; ((${#targets[@]})) || break; printf 'sending SIG%s only to tagged R14 PIDs: %s\n' "$signal" "${targets[*]}"; kill "-$signal" "${targets[@]}" 2>/dev/null || true; sleep 5; done
for candidate in "${SELECTED[@]}"; do
  tmux has-session -t "bwa-r14-$candidate" 2>/dev/null && tmux kill-session -t "bwa-r14-$candidate" || true
  state="$(jq -r '.state // "UNKNOWN"' "$RUN_ROOT/candidates/$candidate/status.json" 2>/dev/null || true)"
  if [[ "$state" != PASSED && "$state" != FAILED ]]; then "$PYTHON" "$ROOT/scripts/before_we_act/r14_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state STOPPED --stage stopped --program stop_r14_4gpu_tmux.sh --detail "graceful exact-run stop; artifacts preserved" --pid 0 --child-pid 0 --exit-code 130 --total-steps 100 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"; fi
done
for candidate in "${SELECTED[@]}"; do mapfile -t pids < <(tagged_pids "$candidate"); ((${#pids[@]}==0)) || { printf 'failed to stop %s exact PIDs\n' "$candidate" >&2; exit 5; }; done
printf 'Stopped only selected R14 tasks; checkpoints, cache, dataset and logs preserved.\n'
