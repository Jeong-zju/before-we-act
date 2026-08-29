#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON=${BWA_DOWNLOAD_PYTHON:-/workspace/venvs/robofactory/bin/python}
SCRIPT=/workspace/repos/before-we-act/deployment/vla_baselines/download_datasets.py
LOG_ROOT=/workspace/bwa_vla_runs/download_logs
mkdir -p "$LOG_ROOT"
tasks=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
pids=()

stop_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap stop_children TERM INT

for task in "${tasks[@]}"; do
  setsid "$PYTHON" "$SCRIPT" --task "$task" > "$LOG_ROOT/$task.log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "dataset download failed: ${tasks[$i]} (log: $LOG_ROOT/${tasks[$i]}.log)" >&2
    failed=1
  fi
done
(( failed == 0 ))
