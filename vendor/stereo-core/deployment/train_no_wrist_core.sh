#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/workspace/no_wrist_stereo_core
data_root=/workspace/datasets/robofactory_multitask
output=${OUTPUT:-/workspace/runs/no_wrist_stereo_core_120k}
updates=${UPDATES:-120000}
workers=${WORKERS:-8}
extra=()
if [[ "$updates" != 120000 ]]; then
  extra+=(--allow-preflight)
fi
if [[ -f "$output/checkpoint_latest.pt" ]]; then
  extra+=(--resume "$output/checkpoint_latest.pt")
fi

export PYTHONPATH="$repo_root/stereo_core${PYTHONPATH:+:$PYTHONPATH}"
exec /venv/robofactory-act/bin/python -u \
  "$repo_root/stereo_core/train_no_wrist_pair.py" \
  --manifests \
    "$data_root/lift_barrier/training_manifest.json" \
    "$data_root/camera_alignment/training_manifest.json" \
    "$data_root/three_robots_stack_cube/training_manifest.json" \
    "$data_root/long_pipeline_delivery/training_manifest.json" \
    "$data_root/take_photo/training_manifest.json" \
  --dino-model /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
  --output "$output" \
  --updates "$updates" \
  --batch-size 40 \
  --workers "$workers" \
  --save-every 1000 \
  --milestones 20000,40000,60000,80000,100000,120000 \
  "${extra[@]}"
