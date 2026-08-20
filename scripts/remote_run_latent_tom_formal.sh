#!/usr/bin/env bash
set -euo pipefail

REPO=${LATENT_TOM_REPO:-/workspace/latent-tom}
RUN_ROOT=${BWA_RUN_ROOT:-/workspace/bwa-baselines-runs}
PYTHON=${BWA_PYTHON:-/venv/robofactory-act/bin/python}
GPU=${BWA_GPU:-3}
OUT=${RUN_ROOT}/formal/latent_tom

mkdir -p "${OUT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

exec "${PYTHON}" -u diffusion_policy/workspace/train_diffusion_individual_camera_workspace.py \
  --config-name=sheaf_individual_camera_diffusion_workspace.yaml \
  task=robofactory_six_task \
  hydra.run.dir="${OUT}" \
  training.device=cuda:0 \
  training.resume=true \
  training.num_epochs=300 \
  training.use_ema=false \
  training.max_train_steps=200 \
  training.max_val_steps=20 \
  training.checkpoint_every=10 \
  training.val_every=1 \
  dataloader.batch_size=4 \
  dataloader.num_workers=4 \
  val_dataloader.batch_size=4 \
  val_dataloader.num_workers=4 \
  policy.down_dims='[256,512,1024]' \
  policy.num_inference_steps=20 \
  logging.mode=offline \
  2>&1
