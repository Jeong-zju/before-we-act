#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1/assets
ONCE=0
INTERVAL=20
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid interval\n' >&2; exit 2; }
render() {
  printf 'R13N asset monitor | %s\n' "$(date -u +%FT%TZ)"
  if [[ -f "$RUN_ROOT/state.json" ]]; then
    jq -r '"status=\(.status) stage=\(.stage) task=\(.task)\nrepo=\(.repo)\npid=\(.pid) child=\(.child_pid) started=\(.started_at)\ndetail=\(.detail)\ndata=\(.data_root) cache=\(.hf_home) log=\(.log)"' "$RUN_ROOT/state.json"
  else
    printf 'status=NOT_STARTED\n'
  fi
  if [[ -f "$RUN_ROOT/heartbeat.json" ]]; then
    local updated now age
    updated="$(jq -r '.updated_at_epoch // 0' "$RUN_ROOT/heartbeat.json")"
    now="$(date +%s)"
    age="$(awk -v n="$now" -v u="$updated" 'BEGIN {printf "%.1f", n-u}')"
    printf 'heartbeat_age_seconds=%s\n' "$age"
  fi
  df -h /workspace | tail -n 1
  [[ -f "$RUN_ROOT/download.log" ]] && tail -n 5 "$RUN_ROOT/download.log" | sed 's/^/  /'
}
while true; do
  ((ONCE)) || printf '\033[2J\033[H'
  render
  ((ONCE)) && exit 0
  sleep "$INTERVAL"
done
