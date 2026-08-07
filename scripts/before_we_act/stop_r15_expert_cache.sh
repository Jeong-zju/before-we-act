#!/usr/bin/env bash
set -Eeuo pipefail
CACHE_ROOT=""; GRACE=30; DRY_RUN=0; SESSION=bwa-r15-expert-cache
while (($#)); do case "$1" in --cache-root) CACHE_ROOT="$2"; shift 2 ;; --grace-seconds) GRACE="$2"; shift 2 ;; --dry-run) DRY_RUN=1; shift ;; *) exit 2 ;; esac; done
[[ -n "$CACHE_ROOT" ]] || { printf '%s\n' '--cache-root required' >&2; exit 2; }
tagged_pids() {
  local environment pid
  for environment in /proc/[0-9]*/environ; do
    [[ -r "$environment" ]] || continue; pid="${environment#/proc/}"; pid="${pid%/environ}"
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    grep -Fzqx "BWA_R15_EXPERT_CACHE_OUTPUT=$CACHE_ROOT" "$environment" 2>/dev/null && printf '%s\n' "$pid"
  done
}
mapfile -t PIDS < <(tagged_pids); printf 'exact cache target session=%s pids=%s\n' "$SESSION" "${PIDS[*]:-none}"
if ((DRY_RUN)); then printf 'dry-run: no changes\n'; exit 0; fi
tmux has-session -t "$SESSION" 2>/dev/null && tmux send-keys -t "$SESSION" C-c || true
for ((second=0; second<GRACE; second++)); do mapfile -t PIDS < <(tagged_pids); ((${#PIDS[@]}==0)) && break; sleep 1; done
for signal in TERM KILL; do mapfile -t PIDS < <(tagged_pids); ((${#PIDS[@]})) || break; kill "-$signal" "${PIDS[@]}" 2>/dev/null || true; sleep 5; done
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
printf 'stopped only exact expert cache job; completed shards preserved\n'
