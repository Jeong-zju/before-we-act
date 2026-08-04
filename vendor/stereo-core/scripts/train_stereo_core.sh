#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-$repo_root/data/RoboFactory-5Task-RGBD-Decentralized/data}"
model_root="${MODEL_ROOT:-$repo_root/artifacts/Stereo-CoRE}"
output="${OUTPUT:-$repo_root/runs/stereo_core_120k}"
workers="${WORKERS:-8}"

export PYTHONPATH="$repo_root/stereo_core:${PYTHONPATH:-}"

python -u "$repo_root/stereo_core/train_pair_route_single_b40_120k.py" \
  --data "$data_root/lift_barrier/*.h5,$data_root/camera_alignment/*.h5,$data_root/three_robots_stack_cube/*.h5,$data_root/long_pipeline_delivery/*.h5,$data_root/take_photo/*.h5" \
  --normalization "$model_root/training_artifacts/stereo_core/normalization.pt" \
  --teacher "$model_root/training_artifacts/complementarity_teacher_seed_20260801.pt" \
  --output "$output" \
  --updates 120000 \
  --batch-size 40 \
  --save-updates 60000,80000,100000,120000 \
  --sampler-mode weighted_items \
  --workers "$workers" \
  --relation-weight 0 \
  --anchor-weight 0 \
  --specialization-weight 0 \
  --capability-weight 0.05 \
  --counterfactual-every 4 \
  --experiment-label stereo_core \
  --gpu-label GPU0
