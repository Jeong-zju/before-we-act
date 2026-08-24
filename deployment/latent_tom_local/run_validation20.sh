#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/repos/before-we-act/deployment/latent_tom_local
OUT=/workspace/bwa_latent_tom_runs/formal/validation20
export PYTHONPATH=/workspace/repos/LatentToM:${ROOT}:/workspace/repos/RoboFactory
exec /venv/main/bin/python -u "${ROOT}/evaluate_closed_loop.py" --checkpoint /workspace/bwa_latent_tom_runs/formal/last.pt --output "${OUT}" --episodes 20 --seed 20260820 --device cuda:0 --diffusion-steps 20 --replan-interval 8
