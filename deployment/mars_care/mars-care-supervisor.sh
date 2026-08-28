#!/bin/bash
set -Eeuo pipefail
export MARS_CARE_REPO=/workspace/repos/before-we-act
export MARS_ROBOFACTORY_ROOT=/workspace/repos/RoboFactory
export MARS_PYTHON=/workspace/venvs/mars/bin/python
export PYTHONPATH="${MARS_CARE_REPO}:${MARS_ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${MARS_CARE_REPO}"
exec "${MARS_PYTHON}" -u -m deployment.mars_care.supervisor

