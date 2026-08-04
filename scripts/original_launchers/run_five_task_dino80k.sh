#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export HF_TOKEN="$(cat /workspace/.hf_dinov3_token)"
cd /workspace/RoboFactory
# Keep recoverable temporary milestones during the long frozen-DINO run.  The
# official evaluation remains checkpoint_080000; earlier checkpoints are
# deleted after that evaluation passes.
exec /workspace/venvs/robofactory-act/bin/python3.10 -u /workspace/act_liftbarrier/train_act.py \
  --data 'runs/strict640x480_v2/lift_barrier/*.h5,runs/strict640x480_v2/camera_alignment/*.h5,runs/strict640x480_v2/three_robots_stack_cube/*.h5,runs/strict640x480_v2/long_pipeline_delivery/*.h5,runs/strict640x480_v2/take_photo/*.h5' \
  --shared --shared-arms 0,1,2,3 --vision-backbone dinov3_vitb16_frozen --camera-width 640 --camera-height 480 \
  --batch-size 40 --workers 0 --lazy-cache-episodes 64 --episode-block-updates 64 --updates 80000 --save-updates 20000,40000,60000,80000 --task-balanced --seed 20260728 \
  --output runs/strict640x480_v2/results/frozen_dinov3_act_all5_80k
