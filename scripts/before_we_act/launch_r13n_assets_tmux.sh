#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION=bwa-r13n-assets
RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1/assets
DRY_RUN=0
while (($#)); do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid tmux session\n' >&2; exit 2; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'already active: %s\n' "$SESSION" >&2
  exit 3
fi
command=("$FE_ROOT/scripts/before_we_act/download_r13n_hf_assets.sh" --run-root "$RUN_ROOT")
((DRY_RUN)) && command+=(--dry-run)
printf -v command_text '%q ' "${command[@]}"
if ((DRY_RUN)); then
  printf '%s\n' "$command_text"
  exit 0
fi
tmux new-session -d -s "$SESSION" "cd '$FE_ROOT' && exec $command_text"
printf 'started session=%s run_root=%s\n' "$SESSION" "$RUN_ROOT"
printf 'monitor: %s --run-root %s --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r13n_assets.sh" "$RUN_ROOT"
