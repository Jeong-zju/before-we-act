#!/usr/bin/env bash
set -euo pipefail

REPO=${MANIFLOW_REPO:-/workspace/maniflow}
RUN_ROOT=${BWA_RUN_ROOT:-/workspace/bwa-baselines-runs}
PYTHON=${BWA_PYTHON:-/venv/robofactory-act/bin/python}
GPU=${BWA_GPU:-2}
OUT=${RUN_ROOT}/formal/maniflow

mkdir -p "${OUT}"
cd "${REPO}/maniflow/workspace"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

exec "${PYTHON}" -u train_maniflow_robotwin_workspace.py \
  --config-name=maniflow_image_timm_policy_robotwin.yaml \
  robotwin_task=robofactory_six_task_image \
  hydra.run.dir="${OUT}" \
  training.device=cuda:0 \
  training.resume=true \
  training.num_epochs=501 \
  training.max_train_steps=200 \
  training.max_val_steps=20 \
  training.val_every=1 \
  training.checkpoint_every=10 \
  training.sample_every=5 \
  dataloader.batch_size=16 \
  dataloader.num_workers=4 \
  val_dataloader.batch_size=16 \
  val_dataloader.num_workers=4 \
  policy.n_layer=12 \
  policy.n_head=8 \
  policy.n_emb=768 \
  policy.obs_encoder.model_name=resnet34.a1_in1k \
  policy.obs_encoder.pretrained=false \
  policy.obs_encoder.feature_aggregation=avg \
  policy.obs_encoder.transforms=null \
  logging.mode=offline \
  checkpoint.save_ckpt=true \
  2>&1
