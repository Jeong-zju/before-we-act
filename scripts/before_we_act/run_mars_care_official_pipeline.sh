#!/bin/bash
set -Eeuo pipefail

repo=/workspace/repos/care-official
python_bin=/workspace/venvs/mars/bin/python
raw=/workspace/datasets/mars_control/raw
robofactory=/workspace/repos/RoboFactory
dino=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
visual_cache=/workspace/runs/care_dino_mars/dino_cache
run=/workspace/runs/care_official_mars_v1
settings=${repo}/configs/before_we_act/care_mars_bench_port.json
normalization=${run}/contract/mars_norm_absolute.json
manifest=${run}/contract/care_family_manifest.json
status=${run}/pipeline_status.json
logs=${run}/logs/formal

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${logs}"
cd "${repo}"

set_status() {
  "${python_bin}" -m scripts.before_we_act.update_mars_care_pipeline_status \
    --output "${status}" --stage "$1" --status "$2" --detail "${3:-}"
}

on_error() {
  local code=$?
  set_status "${current_stage:-UNKNOWN}" FAILED "exit=${code} line=${BASH_LINENO[0]}"
  exit "${code}"
}
trap on_error ERR

json_passed() {
  "${python_bin}" - "$1" "$2" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")==sys.argv[2] else 1)
PY
}

wait_all() {
  local failed=0 pid
  for pid in "$@"; do wait "${pid}" || failed=1; done
  test "${failed}" -eq 0
}

current_stage=PREFLIGHT
set_status "${current_stage}" RUNNING
for receipt in \
  "${run}/contract/b0h_f1_receipt.json" \
  "${run}/contract/b0h_closed_loop_smoke_receipt.json" \
  "${run}/contract/belief_closed_loop_smoke_receipt.json" \
  "${run}/contract/care_end_to_end_smoke_receipt.json"; do
  json_passed "${receipt}" PASSED
done
json_passed "${visual_cache}/cache_receipt.json" PASSED
set_status "${current_stage}" PASSED "all train/resume/four-task closed-loop smokes passed"

current_stage=B0H_FORMAL
set_status "${current_stage}" RUNNING "120000 updates, DDP on all four RTX 5090 GPUs"
b0h=${run}/b0h_formal
b0h_checkpoint=${b0h}/checkpoint_latest.pt
if ! "${python_bin}" - "${b0h_checkpoint}" <<'PY'
import sys,torch
try: x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
except Exception: raise SystemExit(1)
c=x.get("config",{})
ok=x.get("update")==120000 and c.get("action_encoding")=="absolute_pd_joint_pos" and c.get("vision")=="dinov3_vitb16_frozen" and c.get("strict_local") is True
raise SystemExit(0 if ok else 1)
PY
then
  resume=()
  test ! -f "${b0h_checkpoint}" || resume=(--resume "${b0h_checkpoint}")
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m before_we_act.train_mars_temporal_policy \
    --stage formal --raw-root "${raw}" --normalization "${normalization}" \
    --visual-cache "${visual_cache}" --dino-model "${dino}" --output "${b0h}" \
    --updates 120000 --workers 8 --save-every 5000 --log-every 20 "${resume[@]}" \
    >"${logs}/b0h_formal.log" 2>&1
fi
set_status "${current_stage}" PASSED "${b0h_checkpoint}"

current_stage=ACTION_CONTEXT_CACHE
set_status "${current_stage}" RUNNING "all 600 episodes, four RTX 5090 shards"
action_cache=${run}/action_context_full
if ! json_passed "${action_cache}/cache_receipt.json" PASSED 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m scripts.before_we_act.build_mars_action_context_cache \
    --raw-root "${raw}" --normalization "${normalization}" \
    --visual-cache "${visual_cache}" --temporal-checkpoint "${b0h_checkpoint}" \
    --dino-model "${dino}" --output "${action_cache}" --batch-size 32 \
    >"${logs}/action_context_full.log" 2>&1
fi
set_status "${current_stage}" PASSED "all 600 demonstrations cached"

current_stage=BELIEF_FORMAL
set_status "${current_stage}" RUNNING "three seeds; three independent RTX 5090 workers"
belief_root=${run}/belief_training
run_belief() {
  # Bind positional arguments before expanding dependent variables under set -u.
  local gpu seed output
  gpu=$1
  seed=$2
  output=${belief_root}/seed_${seed}
  if json_passed "${output}/status.json" COMPLETE 2>/dev/null; then return; fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m before_we_act.train_mars_predictive_team_belief \
    --raw-root "${raw}" --normalization "${normalization}" \
    --visual-cache "${visual_cache}" --action-context-cache "${action_cache}" \
    --b0h-checkpoint "${b0h_checkpoint}" --output "${output}" \
    --seed "${seed}" --updates 120000 --workers 4 --save-every 5000 --log-every 100 \
    >"${logs}/belief_${seed}.log" 2>&1
}
run_belief 0 20260815 & pid0=$!
run_belief 1 20260816 & pid1=$!
run_belief 2 20260817 & pid2=$!
wait_all "${pid0}" "${pid1}" "${pid2}"
set_status "${current_stage}" PASSED "all three 120000-update seeds complete"

current_stage=BELIEF_SELECT
set_status "${current_stage}" RUNNING
belief_selected=${run}/belief_selected
"${python_bin}" -m scripts.before_we_act.select_mars_care_belief \
  --training-root "${belief_root}" --output-root "${belief_selected}"
reference_checkpoint=${belief_selected}/deployment_checkpoint.pt
set_status "${current_stage}" PASSED "${reference_checkpoint}"

current_stage=CARE_BRANCHES
set_status "${current_stage}" RUNNING "120 families x 24 branches; globally serial Vulkan"
families=${run}/care_families
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
  --manifest "${manifest}" --checkpoint "${reference_checkpoint}" \
  --output-root "${families}" --robofactory-root "${robofactory}" \
  --device cuda:0 --render-device cuda:0 \
  >"${logs}/care_branches.log" 2>&1
set_status "${current_stage}" PASSED "120 complete families"

current_stage=CARE_PREPARE
set_status "${current_stage}" RUNNING
quality=${run}/care_quality
prepared=${run}/care_prepared.pt
prepared_manifest=${run}/care_prepared_manifest.json
"${python_bin}" -m scripts.before_we_act.prepare_mars_care_training quality \
  --family-root "${families}" --quality-root "${quality}" \
  >"${logs}/care_quality.log" 2>&1
if test ! -f "${prepared}"; then
  "${python_bin}" -m scripts.before_we_act.prepare_mars_care_training prepare \
    --family-root "${families}" --quality-root "${quality}" \
    --reference-checkpoint "${reference_checkpoint}" \
    --output "${prepared}" --manifest-output "${prepared_manifest}" \
    >"${logs}/care_prepare.log" 2>&1
fi
set_status "${current_stage}" PASSED "all 120 families in scorer training"

current_stage=CARE_SCORERS
set_status "${current_stage}" RUNNING "four variants x three seeds, four parallel RTX 5090 workers"
scorer_root=${run}/care_training
jobs=()
for variant in care reactive_only replay_only capacity; do
  for seed in 20260818 20260819 20260820; do jobs+=("${variant}:${seed}"); done
done
run_scorer() {
  local gpu spec variant seed output
  gpu=$1
  spec=$2
  variant=${spec%%:*}
  seed=${spec##*:}
  output=${scorer_root}/${variant}/seed_${seed}
  if json_passed "${output}/status.json" COMPLETED 2>/dev/null; then return; fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m before_we_act.train_mars_care_belief \
    --prepared-data "${prepared}" --output "${output}" \
    --seed "${seed}" --variant "${variant}" --stage formal --updates 4000 \
    --batch-size 48 --eval-every 200 --save-every 200 --device cuda:0 \
    >"${logs}/care_scorer_${variant}_${seed}.log" 2>&1
}
for ((index=0; index<${#jobs[@]}; index+=4)); do
  pids=()
  for slot in 0 1 2 3; do
    run_scorer "${slot}" "${jobs[index+slot]}" & pids+=("$!")
  done
  wait_all "${pids[@]}"
done
set_status "${current_stage}" PASSED "12 scorer runs complete"

current_stage=CARE_CALIBRATE
set_status "${current_stage}" RUNNING
offline=${run}/care_offline
"${python_bin}" -m scripts.before_we_act.select_calibrate_mars_care \
  --settings "${settings}" --prepared-data "${prepared}" \
  --training-root "${scorer_root}" --reference-checkpoint "${reference_checkpoint}" \
  --output-root "${offline}" --device cuda:0 \
  >"${logs}/care_select_calibrate.log" 2>&1
care_checkpoint=${offline}/care_deployment_checkpoint.pt
set_status "${current_stage}" PASSED "${care_checkpoint}"

current_stage=VALIDATION20
set_status "${current_stage}" RUNNING "selector-off and CARE, 20 independent seeds/task; globally serial Vulkan"
validation=${run}/validation20
for mode in selector_off care; do
  mkdir -p "${validation}/${mode}"
  for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
    case "${task}" in
      place_cube_in_cup|strike_cube_hard) max_steps=500 ;;
      three_robots_place_shoes) max_steps=1200 ;;
      four_robots_stack_cube) max_steps=800 ;;
    esac
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.evaluate_mars_care_closed_loop \
      --reference-checkpoint "${reference_checkpoint}" --care-checkpoint "${care_checkpoint}" \
      --task "${task}" --robofactory-root "${robofactory}" \
      --output "${validation}/${mode}/${task}.json" --episodes 20 --seed-start 20260827 \
      --max-steps "${max_steps}" --mode "${mode}" --device cuda:0 --render-device cuda:0 \
      >"${logs}/validation20_${mode}_${task}.log" 2>&1
  done
  "${python_bin}" -m scripts.before_we_act.summarize_mars_care_validation \
    --root "${validation}/${mode}" --mode "${mode}" \
    --output "${validation}/${mode}/summary.json"
done
set_status "${current_stage}" PASSED "four-task Validation20 complete"

current_stage=COMPLETE
set_status "${current_stage}" COMPLETE "${validation}/care/summary.json"
