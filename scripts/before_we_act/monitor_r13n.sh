#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1
ONCE=0
INTERVAL=30
while (($#)); do
  case "$1" in --run-root) RUN_ROOT="$2"; shift 2;; --once) ONCE=1; shift;; --interval) INTERVAL="$2"; shift 2;; *) printf 'unknown argument: %s\n' "$1" >&2; exit 2;; esac
done
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid interval\n' >&2; exit 2; }
render() {
  printf 'R13N B6 monitor | %s\nrun_root=%s\n' "$(date -u +%FT%TZ)" "$RUN_ROOT"
  if [[ -f "$RUN_ROOT/status.json" ]]; then
    jq -r '"status=\(.status) stage=\(.stage) program=\(.program)\ndetail=\(.detail)\nbranch=\(.branch) commit=\(.commit) tmux=\(.tmux_session)\npid=\(.pid) child=\(.child_pid) started=\(.started_at)"' "$RUN_ROOT/status.json"
  else printf 'status=NOT_STARTED\n'; fi
  local heartbeat="$RUN_ROOT/heartbeat.json" age=unknown
  if [[ -f "$heartbeat" ]]; then age="$(awk -v n="$(date +%s)" -v u="$(jq -r '.updated_at_epoch // 0' "$heartbeat")" 'BEGIN{printf "%.1f",n-u}')"; fi
  printf 'runner_heartbeat_age_seconds=%s\n' "$age"
  if [[ -f "$RUN_ROOT/train/formal/progress.jsonl" ]]; then
    tail -n 1 "$RUN_ROOT/train/formal/progress.jsonl" | jq -r '"training=[\(.update)/\(.target_updates)] loss=\(.loss) l1=\(.l1) kl=\(.plan_kl) lr=\(.learning_rate) eta_hours=\(.eta_hours) gpu_memory_gb=\(.gpu_memory_gb)"'
  fi
  local complete=0 successes=0
  if [[ -d "$RUN_ROOT/evaluation" ]]; then
    complete="$(find "$RUN_ROOT/evaluation" -mindepth 2 -maxdepth 2 -type f -name '*.json' ! -name '*heartbeat*' -print0 2>/dev/null | xargs -0 -r jq -s 'map(.episodes // 0)|add // 0')"
    successes="$(find "$RUN_ROOT/evaluation" -mindepth 2 -maxdepth 2 -type f -name '*.json' ! -name '*heartbeat*' -print0 2>/dev/null | xargs -0 -r jq -s 'map(.successes // 0)|add // 0')"
  fi
  printf 'closed_loop_progress=%s/360 successes=%s\n' "${complete:-0}" "${successes:-0}"
  if [[ -d "$RUN_ROOT/evaluation" ]]; then
    find "$RUN_ROOT/evaluation" -mindepth 2 -maxdepth 2 -type f -name '*.json' ! -name '*heartbeat*' -print0 2>/dev/null | xargs -0 -r jq -r '"  \(.stage)/\(.task): \(.successes)/\(.episodes) p95_ms=\(.latency_ms.p95)"' | sort
  fi
  printf 'gpu:\n'; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | sed 's/^/  /'
  local stage="$(jq -r '.stage // "unknown"' "$RUN_ROOT/status.json" 2>/dev/null || printf unknown)"
  printf 'queue='; case "$stage" in cache_prepare|cache_index) printf 'preflight -> train130k -> offline -> Discovery20 -> Validation20 -> Formal20 -> acceptance\n';; preflight*) printf 'train130k -> offline -> Discovery20 -> Validation20 -> Formal20 -> acceptance\n';; train) printf 'offline -> Discovery20 -> Validation20 -> Formal20 -> acceptance\n';; offline) printf 'Discovery20 -> Validation20 -> Formal20 -> acceptance\n';; discovery) printf 'Validation20 -> Formal20 -> acceptance\n';; validation) printf 'Formal20 -> acceptance\n';; formal) printf 'acceptance\n';; complete) printf 'empty\n';; *) printf 'inspect status\n';; esac
  if find "$RUN_ROOT/logs" -type f -name '*.log' -print0 2>/dev/null | xargs -0 -r grep -HnE 'CUDA out of memory|NaN|FloatingPointError|Traceback \(most recent call last\)' | tail -n 5; then :; fi
  [[ -f "$RUN_ROOT/acceptance.json" ]] && jq -r '"acceptance=\(.status) native=\(.candidate_native_episodes)/360 fallback=\(.fallback_episodes) discovery=\(.stage_totals.discovery.successes)/120 validation=\(.stage_totals.validation.successes)/120 formal=\(.stage_totals.formal.successes)/120"' "$RUN_ROOT/acceptance.json"
  printf 'recent_pipeline_log:\n'; [[ -f "$RUN_ROOT/logs/pipeline.log" ]] && tail -n 8 "$RUN_ROOT/logs/pipeline.log" | sed 's/^/  /'
}
while true; do ((ONCE)) || printf '\033[2J\033[H'; render; ((ONCE)) && exit 0; sleep "$INTERVAL"; done
