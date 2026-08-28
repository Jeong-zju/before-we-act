#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${MARS_LATENT_TOM_ROOT:-/workspace/repos/before-we-act/deployment/mars_latent_tom}
export PYTHONUNBUFFERED=1
exec "${MARS_LATENT_TOM_PYTHON:-/venv/main/bin/python}" -m deployment.mars_latent_tom.supervisor
