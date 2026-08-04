#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 <checkpoint.pt> <task> [device]" >&2
  exit 2
fi

checkpoint="$1"
task="$2"
device="${3:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/protocol/frozen100/$task.json"
output="$repo_root/runs/eval_100_$task/result.json"

mkdir -p "$(dirname "$output")"
export PYTHONPATH="$repo_root/stereo_core:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="$device" python -u "$repo_root/stereo_core/evaluate_stereo_act.py" \
  --checkpoint "$checkpoint" \
  --task "$task" \
  --seed-file "$manifest" \
  --episodes 100 \
  --max-steps 1500 \
  --workers 1 \
  --devices 0 \
  --output "$output"
