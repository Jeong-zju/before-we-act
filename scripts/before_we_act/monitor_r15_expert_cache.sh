#!/usr/bin/env bash
set -Eeuo pipefail
CACHE_ROOT=""; ONCE=0; INTERVAL=30
while (($#)); do case "$1" in --cache-root) CACHE_ROOT="$2"; shift 2 ;; --once) ONCE=1; shift ;; --interval) INTERVAL="$2"; shift 2 ;; *) exit 2 ;; esac; done
[[ -n "$CACHE_ROOT" ]] || { printf '%s\n' '--cache-root required' >&2; exit 2; }
while :; do
  ((ONCE)) || printf '\033[2J\033[H'
  printf 'R15 expert feature cache | %s | root=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CACHE_ROOT"
  printf 'process='; jq -c . "$CACHE_ROOT/process.json" 2>/dev/null || printf 'none\n'
  printf 'state='; jq -c . "$CACHE_ROOT/state.json" 2>/dev/null || printf 'NOT_STARTED\n'
  printf 'heartbeat='; jq -c . "$CACHE_ROOT/heartbeat.json" 2>/dev/null || printf 'none\n'
  printf 'GPU='; gpu="$(jq -r '.gpu // empty' "$CACHE_ROOT/process.json" 2>/dev/null || true)"; [[ "$gpu" =~ ^[0-3]$ ]] && nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits || printf 'unknown\n'
  du -sh "$CACHE_ROOT" 2>/dev/null || true
  tail -8 "$CACHE_ROOT/cache.log" 2>/dev/null || true
  ((ONCE)) && break; sleep "$INTERVAL"
done
