#!/bin/bash
set -euo pipefail

repo=/workspace/repos/care-official
python_bin=/workspace/venvs/mars/bin/python
run=/workspace/runs/care_official_mars_v1
manifest=${run}/contract/care_family_manifest.json
checkpoint=${MARS_CARE_BELIEF_CHECKPOINT:-${run}/belief_smoke/deployment_checkpoint.pt}
output=${run}/care_branch_smoke/families
logs=${run}/care_branch_smoke/logs
robofactory=/workspace/repos/RoboFactory
export PYTHONPATH=${repo}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${output}" "${logs}"
cd "${repo}"

run_task() {
  local gpu=$1
  local task=$2
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m before_we_act.mars_care_branch_collector \
    --manifest "${manifest}" \
    --checkpoint "${checkpoint}" \
    --output-root "${output}" \
    --robofactory-root "${robofactory}" \
    --task "${task}" \
    --limit 1 \
    --device cuda:0 \
    --render-device cuda:0 \
    >"${logs}/${task}.log" 2>&1
}

# The instance Vulkan driver loses devices when two render contexts run at
# once, including when they target distinct GPUs.  Render globally serially.
run_task 0 place_cube_in_cup
run_task 0 strike_cube_hard
run_task 0 three_robots_place_shoes
run_task 0 four_robots_stack_cube

"${python_bin}" -m scripts.before_we_act.prepare_mars_care_training quality \
  --family-root "${output}" \
  --quality-root "${run}/care_branch_smoke/quality"

"${python_bin}" -m scripts.before_we_act.prepare_mars_care_training prepare \
  --family-root "${output}" \
  --quality-root "${run}/care_branch_smoke/quality" \
  --reference-checkpoint "${checkpoint}" \
  --output "${run}/care_branch_smoke/prepared.pt" \
  --manifest-output "${run}/care_branch_smoke/prepared_manifest.json" \
  --expected-families 4

scorer=${run}/care_branch_smoke/scorer
"${python_bin}" -m before_we_act.train_mars_care_belief \
  --prepared-data "${run}/care_branch_smoke/prepared.pt" \
  --output "${scorer}" \
  --seed 20260818 \
  --variant care \
  --stage smoke \
  --updates 2 \
  --save-every 2 \
  --eval-every 2 \
  --device cuda:0

"${python_bin}" -m before_we_act.train_mars_care_belief \
  --prepared-data "${run}/care_branch_smoke/prepared.pt" \
  --output "${scorer}" \
  --seed 20260818 \
  --variant care \
  --stage smoke \
  --updates 4 \
  --save-every 2 \
  --eval-every 2 \
  --device cuda:0

"${python_bin}" -m scripts.before_we_act.build_mars_care_smoke_deployment \
  --prepared-data "${run}/care_branch_smoke/prepared.pt" \
  --training-checkpoint "${scorer}/checkpoint_latest.pt" \
  --reference-checkpoint "${checkpoint}" \
  --output "${run}/care_branch_smoke/care_smoke_deployment.pt" \
  --device cuda:0

for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
  "${python_bin}" -m before_we_act.evaluate_mars_care_closed_loop \
    --reference-checkpoint "${checkpoint}" \
    --care-checkpoint "${run}/care_branch_smoke/care_smoke_deployment.pt" \
    --task "${task}" \
    --robofactory-root "${robofactory}" \
    --output "${run}/care_branch_smoke/closed_loop/${task}.json" \
    --episodes 1 \
    --max-steps 2 \
    --mode care \
    --device cuda:0 \
    --render-device cuda:0 \
    >"${logs}/closed_loop_${task}.log" 2>&1
done

"${python_bin}" -m scripts.before_we_act.verify_mars_care_smoke \
  --root "${run}/care_branch_smoke" \
  --output "${run}/contract/care_end_to_end_smoke_receipt.json"
