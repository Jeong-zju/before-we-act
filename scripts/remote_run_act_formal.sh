#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/bwa-baselines-runs/formal/act
mkdir -p "$ROOT"
printf '%s\n' '{"baseline":"act","status":"starting","updates":0,"target_updates":120000,"device":"cuda:0"}' > "$ROOT/status.json"
cd /workspace/bwa-baselines
export CUDA_VISIBLE_DEVICES=0
/venv/robofactory-act/bin/python -u stereo_core/train_act.py \
  --data unused \
  --shared --shared-arms 0,1,2,3 --output "$ROOT" \
  --zarr-agent 0=/workspace/bwa-baselines-runs/dp/six_tasks_agent0.zarr \
  --zarr-agent 1=/workspace/bwa-baselines-runs/latent_tom/six_tasks_agent1.zarr \
  --zarr-agent 2=/workspace/bwa-baselines-runs/act/zarr_agent2 \
  --zarr-agent 3=/workspace/bwa-baselines-runs/act/zarr_agent3 \
  --horizon 100 --enc-layers 4 --dec-layers 7 --d-model 384 \
  --batch-size 40 --updates 120000 \
  --save-updates 20000,40000,60000,80000,100000,120000 \
  --workers 0 --lazy-cache-episodes 4 --episode-block-updates 64 \
  --task-balanced --camera-width 320 --camera-height 240 \
  --stats-root /workspace/datasets/robofactory_multitask \
  --validation-updates 64 --resume --seed 20260819 > "$ROOT/train.log" 2>&1
printf '%s\n' '{"baseline":"act","status":"complete","updates":120000,"target_updates":120000,"device":"cuda:0"}' > "$ROOT/status.json"
