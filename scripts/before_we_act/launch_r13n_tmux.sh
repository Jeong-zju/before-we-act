#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION=bwa-r13n
RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1
DRY_RUN=0
while (($#)); do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid session\n' >&2; exit 2; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == feat/model-improvements ]] || { printf 'wrong R13N branch\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" rev-parse HEAD)" == "$(git -C "$FE_ROOT" rev-parse origin/feat/model-improvements)" && -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'R13N code is not clean/synced\n' >&2; exit 3; }
if tmux has-session -t "$SESSION" 2>/dev/null; then printf 'R13N session already exists: %s\n' "$SESSION" >&2; exit 3; fi
command=("$FE_ROOT/scripts/before_we_act/run_r13n_pipeline.sh" --run-root "$RUN_ROOT" --session "$SESSION")
printf -v command_text '%q ' "${command[@]}"
printf 'branch=%s commit=%s session=%s run_root=%s\n' "$(git -C "$FE_ROOT" branch --show-current)" "$(git -C "$FE_ROOT" rev-parse HEAD)" "$SESSION" "$RUN_ROOT"
printf 'command=%s\n' "$command_text"
((DRY_RUN)) && exit 0
mkdir -p "$RUN_ROOT/logs"
tmux new-session -d -s "$SESSION" "cd '$FE_ROOT' && exec env BWA_R13N_RUN_ROOT='$RUN_ROOT' BWA_R13N_SESSION='$SESSION' $command_text >>'$RUN_ROOT/logs/pipeline.log' 2>&1"
printf 'monitor once: %s --run-root %s --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r13n.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r13n.sh" "$RUN_ROOT"
printf 'safe stop: %s --run-root %s --session %s\n' "$FE_ROOT/scripts/before_we_act/stop_r13n_tmux.sh" "$RUN_ROOT" "$SESSION"
