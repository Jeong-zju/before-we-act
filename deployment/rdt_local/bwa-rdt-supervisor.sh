#!/bin/bash
set -Eeuo pipefail
[[ ! -f /opt/supervisor-scripts/utils/environment.sh ]] || . /opt/supervisor-scripts/utils/environment.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.hf_home
export BWA_VLA_PIPELINE=${BWA_VLA_PIPELINE:-/workspace/bwa_rdt_pipeline/pipeline.json}
export BWA_VLA_STATE_ROOT=${BWA_VLA_STATE_ROOT:-/workspace/bwa_rdt_runs/supervisor}
exec /venv/main/bin/python /workspace/repos/before-we-act/deployment/vla_baselines/orchestrator.py
