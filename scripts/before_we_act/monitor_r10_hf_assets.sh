#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/shared/r10_hf_assets
ONCE=0
INTERVAL=10
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
  local state="$RUN_ROOT/state.json" heartbeat="$RUN_ROOT/heartbeat" age=unknown
  printf 'R10 S0/Hugging Face asset monitor | %s\n' "$(date -u +%FT%TZ)"
  printf 'run_root=%s\n' "$RUN_ROOT"
  if [[ -f "$heartbeat" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$heartbeat") ))
  fi
  if [[ -f "$state" ]]; then
    jq -r '"status=\(.status) stage=\(.stage) task=\(.task)\nrepo=\(.repo)\npid=\(.pid) child=\(.child_pid) files=\(.completed_episode_files)/\(.total_episode_files)\nstarted=\(.started_at) updated=\(.updated_at)\nlog=\(.log)\ndetail=\(.detail)\ncontract=\(.download_contract)"' "$state"
  else
    printf 'status=NOT_STARTED\n'
  fi
  printf 'heartbeat_age_seconds=%s\n' "$age"
  if [[ -f "$RUN_ROOT/download.log" ]]; then
    printf 'recent_log:\n'
    tail -n 5 "$RUN_ROOT/download.log" | sed 's/^/  /'
  fi
}

while true; do
  ((ONCE)) || printf '\033[2J\033[H'
  render
  ((ONCE)) && exit 0
  sleep "$INTERVAL"
done
