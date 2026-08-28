#!/usr/bin/env bash
set -Eeuo pipefail
export MARS_DP_REPO=${MARS_DP_REPO:-/workspace/repos/before-we-act}
export MARS_DP_ROBOFACTORY=${MARS_DP_ROBOFACTORY:-/workspace/repos/RoboFactory}
export MARS_DP_DATA_ROOT=${MARS_DP_DATA_ROOT:-/workspace/datasets/mars_control}
export MARS_DP_RUN_ROOT=${MARS_DP_RUN_ROOT:-/workspace/runs/mars_dp_v2}
export MARS_DP_PYTHON=${MARS_DP_PYTHON:-/venv/main/bin/python}
cd "${MARS_DP_REPO}"
exec "${MARS_DP_PYTHON}" -u -m deployment.mars_dp.supervisor
