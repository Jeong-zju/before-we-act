#!/usr/bin/env bash
set -Eeuo pipefail
OUTPUT_ROOT=""; GRACE=30; DRY_RUN=0
while (($#)); do case "$1" in --output-root) OUTPUT_ROOT="$2"; shift 2 ;; --grace-seconds) GRACE="$2"; shift 2 ;; --dry-run) DRY_RUN=1; shift ;; *) exit 2 ;; esac; done
[[ -n "$OUTPUT_ROOT" ]] || { printf '%s\n' '--output-root required' >&2; exit 2; }
tagged_pids() {
  local environment pid
  for environment in /proc/[0-9]*/environ; do
    [[ -r "$environment" ]] || continue; pid="${environment#/proc/}"; pid="${pid%/environ}"
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    grep -Fzqx "BWA_R15_EXPERT_OUTPUT=$OUTPUT_ROOT" "$environment" 2>/dev/null && printf '%s\n' "$pid"
  done
}
mapfile -t PIDS < <(tagged_pids); printf 'exact expert target session=bwa-r15-expert-collect pids=%s\n' "${PIDS[*]:-none}"
if ((DRY_RUN)); then printf 'dry-run: no changes\n'; exit 0; fi
tmux has-session -t bwa-r15-expert-collect 2>/dev/null && tmux send-keys -t bwa-r15-expert-collect C-c || true
for ((second=0; second<GRACE; second++)); do mapfile -t PIDS < <(tagged_pids); ((${#PIDS[@]}==0)) && break; sleep 1; done
for signal in TERM KILL; do mapfile -t PIDS < <(tagged_pids); ((${#PIDS[@]})) || break; kill "-$signal" "${PIDS[@]}" 2>/dev/null || true; sleep 5; done
tmux has-session -t bwa-r15-expert-collect 2>/dev/null && tmux kill-session -t bwa-r15-expert-collect || true
printf 'stopped only exact expert collection; partial raw outputs preserved\n'
