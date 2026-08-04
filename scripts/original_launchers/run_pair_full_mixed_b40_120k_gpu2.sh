#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export DEFM_CHECKPOINT=/workspace/external/defm/defm_vit_s14.pth
export HF_TOKEN="$(cat /workspace/.hf_dinov3_token)"
cd /workspace/RoboFactory
exec /workspace/venvs/robofactory-act/bin/python -u /workspace/act_liftbarrier/train_pair_route_single_b40_120k.py \
  --data 'runs/strict640x480_v2/lift_barrier/*.h5,runs/strict640x480_v2/camera_alignment/*.h5,runs/strict640x480_v2/three_robots_stack_cube/*.h5,runs/strict640x480_v2/long_pipeline_delivery/*.h5,runs/strict640x480_v2/take_photo/*.h5' \
  --normalization runs/strict640x480_v2/results/local_arca_all5_80k/checkpoint_080000.pt \
  --teacher runs/strict640x480_v2/results/role_observability/complementarity_teacher_seed_20260801/teacher.pt \
  --output runs/strict640x480_v2/results/pair_route_full_mixed_single_b40_120k \
  --updates 120000 --batch-size 40 --save-updates 60000,80000,100000,120000 \
  --sampler-mode mixed_team_items --workers 8 \
  --relation-weight 0.05 --anchor-weight 0.02 \
  --capability-weight 0.05 --specialization-weight 0.01 --counterfactual-every 4 \
  --experiment-label pair_route_full_mixed_no_task_block --gpu-label GPU2
