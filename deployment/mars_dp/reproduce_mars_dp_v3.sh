#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="${MARS_DP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export MARS_DP_REPO="$repo_root"
export MARS_DP_ROBOFACTORY="${MARS_DP_ROBOFACTORY:-/workspace/repos/RoboFactory}"
export MARS_DP_DATA_ROOT="${MARS_DP_DATA_ROOT:-/workspace/datasets/mars_control}"
export MARS_DP_RUN_ROOT="${MARS_DP_RUN_ROOT:-/workspace/runs/mars_dp_v3}"
export MARS_DP_PYTHON="${MARS_DP_PYTHON:-/venv/main/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$MARS_DP_REPO:$MARS_DP_ROBOFACTORY:$MARS_DP_ROBOFACTORY/robofactory/policy/Diffusion-Policy${PYTHONPATH:+:$PYTHONPATH}"

cd "$MARS_DP_REPO"
"$MARS_DP_PYTHON" -u -m deployment.mars_dp.verify_frozen_config
"$MARS_DP_PYTHON" -c 'import h5py, torch, diffusers; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
"$MARS_DP_PYTHON" -u -m deployment.mars_dp.audit
"$MARS_DP_PYTHON" -u -m deployment.mars_dp.preflight_v2
exec "$MARS_DP_PYTHON" -u -m deployment.mars_dp.supervisor_v3
