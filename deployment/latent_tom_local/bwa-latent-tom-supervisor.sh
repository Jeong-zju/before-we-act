#!/usr/bin/env bash
# Crash-resumable, GPU-aware supervisor entry point for this pipeline.
set -Eeuo pipefail

ROOT=${BWA_LATENT_TOM_ROOT:-/workspace/repos/before-we-act/deployment/latent_tom_local}
export PYTHONUNBUFFERED=1
export BWA_VLA_PIPELINE=${BWA_LATENT_TOM_PIPELINE:-${ROOT}/pipeline.json}
export BWA_VLA_STATE_ROOT=${BWA_LATENT_TOM_STATE_ROOT:-/workspace/bwa_latent_tom_runs/supervisor}
exec "${BWA_PYTHON_BIN:-/venv/main/bin/python}" \
  "${ROOT}/../vla_baselines/orchestrator.py"
