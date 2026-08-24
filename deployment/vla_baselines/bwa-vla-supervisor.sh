#!/bin/bash
set -euo pipefail
[[ ! -f /opt/supervisor-scripts/utils/environment.sh ]] || . /opt/supervisor-scripts/utils/environment.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.hf_home
export BWA_VLA_PIPELINE=${BWA_VLA_PIPELINE:-/workspace/bwa_openvla_pipeline/pipeline.json}
export BWA_VLA_STATE_ROOT=${BWA_VLA_STATE_ROOT:-/workspace/bwa_vla_runs/supervisor}
exec /workspace/venvs/robofactory/bin/python /workspace/repos/before-we-act/deployment/vla_baselines/orchestrator.py
