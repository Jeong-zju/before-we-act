#!/bin/bash
set -euo pipefail
. /opt/supervisor-scripts/utils/environment.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.hf_home
export BWA_VLA_PIPELINE=${BWA_GAUDP_PIPELINE:-/workspace/bwa_gau_dp_pipeline/pipeline.json}
export BWA_VLA_STATE_ROOT=${BWA_GAUDP_STATE_ROOT:-/workspace/bwa_gau_dp_runs/supervisor}
exec /venv/main/bin/python /workspace/repos/before-we-act/deployment/vla_baselines/orchestrator.py
