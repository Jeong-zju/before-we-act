#!/usr/bin/env bash
set -euo pipefail
export DEFM_CHECKPOINT=/workspace/external/defm/defm_vit_s14.pth
export HF_TOKEN="$(cat /workspace/.hf_dinov3_token)"
cd /workspace/RoboFactory
common=(
  --data 'runs/strict640x480_v2/lift_barrier/*.h5,runs/strict640x480_v2/camera_alignment/*.h5,runs/strict640x480_v2/three_robots_stack_cube/*.h5,runs/strict640x480_v2/long_pipeline_delivery/*.h5,runs/strict640x480_v2/take_photo/*.h5'
  --normalization runs/strict640x480_v2/results/local_arca_all5_80k/checkpoint_080000.pt
  --teacher runs/strict640x480_v2/results/role_observability/complementarity_teacher_seed_20260801/teacher.pt
  --updates 2 --batch-size 40 --save-updates '' --allow-preflight --log-every 1
  --capability-weight 0.05 --specialization-weight 0.01 --counterfactual-every 1
)
CUDA_VISIBLE_DEVICES=2 /workspace/venvs/robofactory-act/bin/python -u /workspace/act_liftbarrier/train_pair_route_single_b40_120k.py \
  "${common[@]}" --output /tmp/pair_route_shared_full_smoke2 \
  --relation-weight 0.05 --anchor-weight 0.02 --experiment-label pair_route_full_smoke --gpu-label GPU2 \
  >/tmp/pair_route_shared_full_smoke2.log 2>&1 &
full_pid=$!
CUDA_VISIBLE_DEVICES=3 /workspace/venvs/robofactory-act/bin/python -u /workspace/act_liftbarrier/train_pair_route_single_b40_120k.py \
  "${common[@]}" --output /tmp/pair_route_shared_spec_cap_smoke2 \
  --relation-weight 0 --anchor-weight 0 --experiment-label pair_route_specialization_capability_only_smoke --gpu-label GPU3 \
  >/tmp/pair_route_shared_spec_cap_smoke2.log 2>&1 &
ablation_pid=$!
wait "$full_pid"
wait "$ablation_pid"
