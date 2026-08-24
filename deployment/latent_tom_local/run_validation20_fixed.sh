#!/usr/bin/env bash
set -euo pipefail

ROOT=${BWA_LATENT_TOM_ROOT:-/workspace/repos/before-we-act/deployment/latent_tom_local}
# shellcheck disable=SC1091
source "${ROOT}/fixed_params.env"
cd "${ROOT}"
export PYTHONPATH=${BWA_LATENT_TOM_UPSTREAM:-/workspace/repos/LatentToM}:${ROOT}:${BWA_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}${PYTHONPATH:+:${PYTHONPATH}}

exec "${BWA_PYTHON_BIN:-/venv/main/bin/python}" -u evaluate_closed_loop.py \
  --checkpoint "${BWA_LATENT_TOM_CHECKPOINT:-${BWA_LATENT_TOM_OUTPUT:-/workspace/bwa_latent_tom_runs/formal}/last.pt}" \
  --output "${BWA_LATENT_TOM_VALIDATION_OUTPUT:-${BWA_LATENT_TOM_OUTPUT:-/workspace/bwa_latent_tom_runs/formal}/validation20_fixed}" \
  --config-root "${BWA_ROBOFACTORY_CONFIG_ROOT:-${BWA_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}/robofactory/configs/table}" \
  --episodes "${LATENT_TOM_VALIDATION_EPISODES}" \
  --seed "${LATENT_TOM_VALIDATION_SEED}" \
  --device "${BWA_DEVICE:-cuda:0}" \
  --diffusion-steps "${LATENT_TOM_DIFFUSION_STEPS}" \
  --replan-interval "${LATENT_TOM_REPLAN_INTERVAL}"
