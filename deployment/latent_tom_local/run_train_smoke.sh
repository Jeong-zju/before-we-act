#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/repos/before-we-act/deployment/latent_tom_local
OUT=/workspace/bwa_latent_tom_runs/formal
export PYTHONPATH=/workspace/repos/LatentToM:${ROOT}
export BWA_DATASET_ROOT=/workspace/datasets/robofactory_multitask
exec /venv/main/bin/python -u "${ROOT}/train_local.py" --output "${OUT}" --steps 10 --batch-size 8 --grad-accum 1 --workers 4 --save-every 10 --smoke
