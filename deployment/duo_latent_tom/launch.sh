#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${DUO_LATENT_TOM_REPO:-/workspace/repos/before-we-act}
export PYTHONPATH="${ROOT}:${LATENT_TOM_ROOT:-/workspace/repos/LatentToM}:${DUOBENCH_ROOT:-/workspace/repos/duobench}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${DUO_LATENT_TOM_PYTHON:-/venv/main/bin/python}" -u -m deployment.duo_latent_tom.supervisor
