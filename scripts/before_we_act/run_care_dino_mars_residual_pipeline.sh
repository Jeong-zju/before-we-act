#!/bin/bash
set -euo pipefail
repo=/workspace/repos/care-official
py=/workspace/venvs/mars/bin/python
raw=/workspace/datasets/mars_control/raw
dino=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
cache=/workspace/runs/care_dino_mars/dino_cache
run=/workspace/runs/care_dino_mars_residual
formal=${run}/reference_formal
validation=${run}/validation20
ckpt=${formal}/checkpoint_latest.pt
mkdir -p "${formal}" "${validation}"
cd "${repo}"

resume=()
if test -f "${ckpt}"; then resume=(--resume "${ckpt}"); fi
"${py}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m before_we_act.train_mars_temporal_policy --stage formal \
  --raw-root "${raw}" --normalization "${run}/mars_norm.json" \
  --visual-cache "${cache}" --dino-model "${dino}" --output "${formal}" \
  --updates 120000 --workers 8 --save-every 5000 --log-every 20 "${resume[@]}"

"${py}" - "${ckpt}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
assert x["update"]==120000
assert x["config"]["action_encoding"]=="joint_residual_gripper_absolute"
assert x["config"]["role_context"]=="own_base_xy_in_task_context"
PY

for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
  if test -f "${validation}/${task}.json"; then continue; fi
  CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_temporal_policy \
    --checkpoint "${ckpt}" --dino-model "${dino}" --task "${task}" \
    --robofactory-root /workspace/repos/RoboFactory \
    --output "${validation}/${task}.json" --episodes 20 --seed-start 20260827 \
    --max-steps 1500 --device cuda:0 \
    >"${validation}/${task}.log" 2>"${validation}/${task}.err.log"
done
exec "${py}" -m before_we_act.summarize_mars_validation20 --root "${validation}"
