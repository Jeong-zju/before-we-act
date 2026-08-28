#!/usr/bin/env bash
set -Eeuo pipefail
CHECKPOINT=${1:-/workspace/bwa_pi05_runs/smoke/pi05/final}
ROOT=${BWA_RUN_ROOT:-/workspace/bwa_pi05_runs}/smoke/pi05/closed_loop
exec /workspace/venvs/robofactory/bin/python \
  /workspace/repos/before-we-act/deployment/vla_baselines/validation_launcher.py \
  --policy pi05 --checkpoint "$CHECKPOINT" --output-root "$ROOT" \
  --episodes 1 --seed 20260820 --max-steps-override 2 --smoke
