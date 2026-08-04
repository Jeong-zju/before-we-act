#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export DEFM_CHECKPOINT=/workspace/external/defm/defm_vit_s14.pth
export HF_TOKEN="$(cat /workspace/.hf_dinov3_token)"
cd /workspace/RoboFactory
exec /workspace/venvs/robofactory-act/bin/python3.10 -u /workspace/act_liftbarrier/train_stereo_variants.py \
  --variant arca --data 'runs/strict640x480_v2/lift_barrier/*.h5,runs/strict640x480_v2/camera_alignment/*.h5,runs/strict640x480_v2/three_robots_stack_cube/*.h5,runs/strict640x480_v2/long_pipeline_delivery/*.h5,runs/strict640x480_v2/take_photo/*.h5' \
  --shared-arms 0,1,2,3 --batch-size 40 --workers 0 --cache-episodes 64 --episode-block-updates 64 --updates 80000 --save-updates 80000 --task-balanced --seed 20260728 \
  --output runs/strict640x480_v2/results/local_arca_all5_80k
