#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${TEMPORAL_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
printf 'time=%s\n' "$(date -u +%FT%TZ)"
jq -c '{status,stage,detail,updated_at_epoch}' "${RUN_ROOT}/pipeline_status.json" 2>/dev/null || true
for file in \
  "${RUN_ROOT}/history_only/discovery/status.json" \
  "${RUN_ROOT}/hidden_residual/formal/status.json" \
  "${RUN_ROOT}/history_only/validation5_status.json" \
  "${RUN_ROOT}/hidden_residual/validation20_status.json"; do
  if [[ -f "${file}" ]]; then
    jq -c --arg path "${file}" \
      '{path:$path,status,stage,variant,update,target_updates,loss,action,eta_hours,detail}' \
      "${file}"
  fi
done
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
alerts=$({ grep -RilE 'Traceback|CUDA out of memory|FloatingPointError|non-finite|NCCL.*error' \
  "${RUN_ROOT}/logs" "${RUN_ROOT}/history_only" "${RUN_ROOT}/hidden_residual" 2>/dev/null || true; } | wc -l)
printf 'alert_logs=%s\n' "${alerts}"
