#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1
SESSION=bwa-r13n
DRY_RUN=0
while (($#)); do
  case "$1" in --run-root) RUN_ROOT="$2"; shift 2;; --session) SESSION="$2"; shift 2;; --dry-run) DRY_RUN=1; shift;; *) printf 'unknown argument: %s\n' "$1" >&2; exit 2;; esac
done
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid session\n' >&2; exit 2; }
[[ -f "$RUN_ROOT/run_manifest.json" ]] || { printf 'missing R13N run manifest\n' >&2; exit 3; }
[[ "$(jq -r '.round' "$RUN_ROOT/run_manifest.json")" == R13N && "$(jq -r '.tmux_session' "$RUN_ROOT/run_manifest.json")" == "$SESSION" && "$(realpath -m "$(jq -r '.run_root' "$RUN_ROOT/run_manifest.json")")" == "$(realpath -m "$RUN_ROOT")" ]] || { printf 'R13N stop identity differs\n' >&2; exit 3; }
printf 'target session=%s run_root=%s\n' "$SESSION" "$RUN_ROOT"
[[ -f "$RUN_ROOT/status.json" ]] && jq -r '"pid=\(.pid) child=\(.child_pid) status=\(.status) stage=\(.stage)"' "$RUN_ROOT/status.json"
((DRY_RUN)) && exit 0
if tmux has-session -t "$SESSION" 2>/dev/null; then tmux send-keys -t "$SESSION" C-c; fi
for _ in $(seq 1 60); do tmux has-session -t "$SESSION" 2>/dev/null || break; sleep 1; done
if tmux has-session -t "$SESSION" 2>/dev/null; then printf 'grace period expired; terminating only %s\n' "$SESSION"; tmux kill-session -t "$SESSION"; fi
printf 'R13N stop complete; datasets, caches, logs and checkpoints preserved\n'
