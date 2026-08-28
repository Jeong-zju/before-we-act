#!/usr/bin/env bash
set -Eeuo pipefail
CHECKPOINT=${1:?checkpoint required}
ROOT=/workspace/bwa_rdt_runs/smoke/rdt/closed_loop
export BWA_GPU_COUNT=4 BWA_VALIDATION_PARALLEL=1
exec /workspace/venvs/robofactory/bin/python /workspace/repos/before-we-act/deployment/vla_baselines/validation_launcher.py \
  --policy rdt --checkpoint "$CHECKPOINT" --output-root "$ROOT" --episodes 1 --seed 20260820 \
  --max-steps-override 1 --smoke
