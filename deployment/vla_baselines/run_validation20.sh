#!/usr/bin/env bash
set -Eeuo pipefail

POLICY=${1:?usage: run_validation20.sh rdt|openvla|pi05|gaudp checkpoint}
CHECKPOINT=${2:?usage: run_validation20.sh rdt|openvla|pi05|gaudp checkpoint}
ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}/formal/${POLICY}/validation20
exec /workspace/venvs/robofactory/bin/python /workspace/repos/before-we-act/deployment/vla_baselines/validation_launcher.py \
  --policy "$POLICY" --checkpoint "$CHECKPOINT" --output-root "$ROOT"
