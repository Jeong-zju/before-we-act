#!/usr/bin/env bash
set -Eeuo pipefail
OUTPUT_ROOT=""; ONCE=0; INTERVAL=30
while (($#)); do case "$1" in --output-root) OUTPUT_ROOT="$2"; shift 2 ;; --once) ONCE=1; shift ;; --interval) INTERVAL="$2"; shift 2 ;; *) exit 2 ;; esac; done
[[ -n "$OUTPUT_ROOT" ]] || { printf '%s\n' '--output-root required' >&2; exit 2; }
while :; do
  ((ONCE)) || printf '\033[2J\033[H'
  printf 'R15 expert collection | %s | output=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUTPUT_ROOT"
  jq . "$OUTPUT_ROOT/status.json" 2>/dev/null || printf 'state=NOT_STARTED\n'
  printf 'heartbeat='; jq -c . "$OUTPUT_ROOT/heartbeat.json" 2>/dev/null || printf 'none\n'
  printf 'disk='; du -sh "$OUTPUT_ROOT" 2>/dev/null || true
  printf 'raw files:\n'; find "$OUTPUT_ROOT/raw" -type f -printf '  %s %p\n' 2>/dev/null | sort || true
  printf 'recent log:\n'; tail -8 "$OUTPUT_ROOT/collector.log" 2>/dev/null || true
  ((ONCE)) && break; sleep "$INTERVAL"
done
