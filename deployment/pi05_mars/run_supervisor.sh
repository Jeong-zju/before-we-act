#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/workspace/repos/before-we-act:/workspace/repos/openpi/src:/workspace/repos/RoboFactory
export OPENPI_MARS_CONTROL_ROOT=/workspace/datasets/mars_control
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
export HF_HOME=/workspace/.hf_home
export PI05_MARS_POLICY_PYTHON=${PI05_MARS_POLICY_PYTHON:-/workspace/venvs/openpi/bin/python}
export PI05_MARS_SIM_PYTHON=${PI05_MARS_SIM_PYTHON:-/workspace/venvs/robofactory/bin/python}
exec /workspace/venvs/openpi/bin/python -m deployment.pi05_mars.supervisor
