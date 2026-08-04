#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-$repo_root/data/RoboFactory-5Task-RGBD-Decentralized/data}"
export PYTHONPATH="$repo_root/stereo_core:${PYTHONPATH:-}"

for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  python "$repo_root/stereo_core/audit_strict640x480_corpus.py" \
    --task "$task" \
    --data "$data_root/$task/*.h5" \
    --expected-episodes 100 \
    --output "$repo_root/runs/audits/$task.json"
done
