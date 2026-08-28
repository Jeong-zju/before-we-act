#!/bin/bash
set -euo pipefail
repo=/workspace/repos/care-official
py=/workspace/venvs/mars/bin/python
ckpt=/workspace/runs/care_dino_mars/reference_formal/checkpoint_latest.pt
dino=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
rf=/workspace/repos/RoboFactory
out=/workspace/runs/care_dino_mars/reference_validation20
mkdir -p "${out}"
cd "${repo}"

run_task() {
  local gpu=$1 task=$2
  if test -f "${out}/${task}.json"; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${py}" -m before_we_act.evaluate_mars_temporal_policy \
    --checkpoint "${ckpt}" --dino-model "${dino}" --task "${task}" \
    --robofactory-root "${rf}" --output "${out}/${task}.json" \
    --episodes 20 --seed-start 20260827 --max-steps 1500 --device cuda:0 \
    >"${out}/${task}.log" 2>"${out}/${task}.err.log"
}

# SAPIEN's Vulkan renderer is not process-safe on the second visible device on
# this host.  Run the remaining environments serially on the verified GPU0;
# completed JSON files are preserved and JSONL provides per-episode resume.
run_task 0 strike_cube_hard
run_task 0 four_robots_stack_cube
run_task 0 place_cube_in_cup
run_task 0 three_robots_place_shoes
exec "${py}" -m before_we_act.summarize_mars_validation20 --root "${out}"
