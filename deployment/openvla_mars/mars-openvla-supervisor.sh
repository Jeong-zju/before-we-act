#!/bin/bash
set -euo pipefail
[[ ! -f /opt/supervisor-scripts/utils/environment.sh ]] || . /opt/supervisor-scripts/utils/environment.sh
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/.hf_home
export BWA_SIM_PYTHON=/workspace/venvs/robofactory-main/bin/python
export MARS_OPENVLA_RUN_ROOT=${MARS_OPENVLA_RUN_ROOT:-/workspace/bwa_mars_openvla_runs}
export PYTHONPATH=/workspace/repos/before-we-act:/workspace/repos/RoboFactory-MARS
exec /workspace/venvs/openvla/bin/python -m deployment.openvla_mars.supervisor
