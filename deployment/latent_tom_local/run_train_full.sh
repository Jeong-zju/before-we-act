#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${BWA_LATENT_TOM_ROOT:-/workspace/repos/before-we-act/deployment/latent_tom_local}
# shellcheck disable=SC1091
source "${ROOT}/fixed_params.env"
OUT=${BWA_LATENT_TOM_OUTPUT:-/workspace/bwa_latent_tom_runs/formal}
export PYTHONPATH=${BWA_LATENT_TOM_UPSTREAM:-/workspace/repos/LatentToM}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export BWA_DATASET_ROOT=${BWA_DATASET_ROOT:-/workspace/datasets/robofactory_multitask}
exec "${BWA_PYTHON_BIN:-/venv/main/bin/python}" -u "${ROOT}/train_local.py" --output "${OUT}" --resume
