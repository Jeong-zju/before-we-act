#!/usr/bin/env bash
set -Eeuo pipefail
OUTPUT_ROOT=""; ONCE=0; INTERVAL=30
while (($#)); do case "$1" in --output-root) OUTPUT_ROOT="$2"; shift 2 ;; --once) ONCE=1; shift ;; --interval) INTERVAL="$2"; shift 2 ;; *) exit 2 ;; esac; done
[[ -n "$OUTPUT_ROOT" ]] || { printf '%s\n' '--output-root required' >&2; exit 2; }
while :; do
  ((ONCE)) || printf '\033[2J\033[H'
  printf 'R15 expert collection | %s | output=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUTPUT_ROOT"
  printf 'identity='; jq -c . "$OUTPUT_ROOT/identity.json" 2>/dev/null || printf 'none\n'
  jq . "$OUTPUT_ROOT/status.json" 2>/dev/null || printf 'state=NOT_STARTED\n'
  printf 'heartbeat='; jq -c . "$OUTPUT_ROOT/heartbeat.json" 2>/dev/null || printf 'none\n'
  heartbeat_at="$(jq -r '.updated_at // empty' "$OUTPUT_ROOT/heartbeat.json" 2>/dev/null || true)"
  if [[ -n "$heartbeat_at" ]]; then printf 'heartbeat_age_seconds=%s\n' "$(( $(date -u +%s) - $(date -u -d "$heartbeat_at" +%s) ))"; fi
  printf 'progress='; if [[ -f "$OUTPUT_ROOT/collector.log" ]]; then sed -n '/^{/p' "$OUTPUT_ROOT/collector.log" | jq -sc 'map(select(has("completed_successes"))) | last // {}'; else printf '{}\n'; fi
  child="$(jq -r '.child_pid // 0' "$OUTPUT_ROOT/status.json" 2>/dev/null || true)"; [[ "$child" =~ ^[1-9][0-9]*$ ]] && ps -o pid,lstart,etime,stat,cmd -p "$child" || true
  gpu="$(jq -r '.gpu // empty' "$OUTPUT_ROOT/identity.json" 2>/dev/null || true)"; [[ "$gpu" =~ ^[0-3]$ ]] && { printf 'GPU='; nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits; } || true
  printf 'disk='; du -sh "$OUTPUT_ROOT" 2>/dev/null || true
  printf 'raw files:\n'; find "$OUTPUT_ROOT/raw" -type f -printf '  %s %p\n' 2>/dev/null | sort || true
  printf 'recent log:\n'; tail -8 "$OUTPUT_ROOT/collector.log" 2>/dev/null || true
  ((ONCE)) && break; sleep "$INTERVAL"
done
