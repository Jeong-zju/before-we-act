#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}/smoke/openvla/closed_loop
CHECKPOINT=${1:-${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}/smoke/openvla_oft/final}
exec /workspace/venvs/robofactory/bin/python \
  /workspace/repos/before-we-act/deployment/vla_baselines/validation_launcher.py \
  --policy openvla \
  --checkpoint "$CHECKPOINT" \
  --output-root "$ROOT" \
  --episodes 1 \
  --seed 20260819 \
  --max-steps-override 2 \
  --smoke
