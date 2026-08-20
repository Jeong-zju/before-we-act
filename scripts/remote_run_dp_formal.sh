#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/bwa-baselines-runs/dp
POLICY_ROOT=/workspace/RoboFactory/robofactory/policy/Diffusion-Policy
mkdir -p "$RUN_ROOT/train-fixed"
printf '%s\n' '{"baseline":"dp","status":"starting","epochs":0,"target_epochs":300,"device":"cuda:1"}' > "$RUN_ROOT/status.json"

cd "$POLICY_ROOT"
set +e
HYDRA_FULL_ERROR=1 /venv/robofactory-act/bin/python -u train.py \
  --config-name=robot_dp.yaml \
  task.name=six_tasks \
  task.dataset.zarr_path="$RUN_ROOT/six_tasks_agent0.zarr" \
  task.dataset.max_train_episodes=null \
  'task.shape_meta.obs.agent_pos.shape=[9]' \
  'task.shape_meta.action.shape=[8]' \
  training.debug=False \
  training.device=cuda:1 \
  training.num_epochs=300 \
  training.max_train_steps=200 \
  training.max_val_steps=20 \
  training.resume=False \
  logging.mode=offline \
  hydra.run.dir="$RUN_ROOT/train-fixed" \
  > "$RUN_ROOT/train-fixed.log" 2>&1
returncode=$?
set -e

if [[ "$returncode" -eq 0 ]]; then
  printf '%s\n' '{"baseline":"dp","status":"complete","epochs":300,"target_epochs":300,"device":"cuda:1"}' > "$RUN_ROOT/status.json"
else
  printf '{"baseline":"dp","status":"failed","returncode":%d,"device":"cuda:1"}\n' "$returncode" > "$RUN_ROOT/status.json"
fi
exit "$returncode"
