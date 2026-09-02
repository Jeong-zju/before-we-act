#!/usr/bin/env bash
set -Eeuo pipefail

# Independent event-aware sampling ablation.  This service intentionally waits
# for the frozen main run to finish so it cannot contend for Vulkan or alter any
# main-run artifact.  It is single-shot: supervisor must not auto-retry it.
repo=${MARS_CARE_REPO:-/workspace/repos/care-official}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
run=${MARS_CARE_RUN_ROOT:-/workspace/runs/care_official_mars_v1}
raw=${MARS_RAW_ROOT:-/workspace/datasets/mars_control/raw}
robofactory=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
visual_cache=${MARS_VISUAL_CACHE:-/workspace/runs/care_dino_mars/dino_cache}
settings=${repo}/configs/before_we_act/care_mars_bench_port.json
spec=${repo}/configs/before_we_act/care_mars_event_aware_ablation_v1.json
main_manifest=${run}/contract/care_family_manifest.json
main_families=${run}/care_families
reference_checkpoint=${run}/belief_selected/deployment_checkpoint.pt
main_status=${run}/pipeline_status.json
root=${run}/sampling_ablation/event_aware_hybrid_v1
manifest=${root}/manifest.json
smoke_manifest=${root}/smoke_manifest.json
families=${root}/families
quality=${root}/quality
prepared=${root}/prepared.pt
prepared_manifest=${root}/prepared_manifest.json
training=${root}/training
offline=${root}/offline
validation=${root}/validation20/care
logs=${root}/logs
status=${root}/status.json

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${root}" "${logs}"
cd "${repo}"

set_status() {
  "${python_bin}" - "${status}" "$1" "${2:-}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); stage=sys.argv[2]; detail=sys.argv[3]
old={}
if path.exists(): old=json.loads(path.read_text())
event={"time_utc":datetime.now(timezone.utc).isoformat(),"stage":stage,"detail":detail}
history=list(old.get("history",[]))
if not history or history[-1] != event: history.append(event)
value={"format_version":"before-we-act.care-mars-sampling-ablation-status/1","stage":stage,
       "detail":detail,"updated_at_utc":event["time_utc"],"history":history,
       "main_protocol_unchanged":True,"promotion_scope":"next_formal_run_only"}
path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

failed() {
  local code=$?
  set_status FAILED "exit=${code} line=${BASH_LINENO[0]}; no automatic retry"
  exit "${code}"
}
trap failed ERR

stage_of_main() {
  "${python_bin}" - "${main_status}" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("stage","MISSING"))
except Exception: print("MISSING")
PY
}

set_status WAITING_FOR_MAIN "independent ablation waits for main COMPLETE; zero GPU/Vulkan use"
while [[ $(stage_of_main) != COMPLETE ]]; do sleep 60; done

set_status FREEZE_MANIFEST "same episodes/focal arms/strata; event ranking cannot see branch outcomes"
if [[ ! -f "${manifest}" || ! -f "${smoke_manifest}" ]]; then
  "${python_bin}" -m scripts.before_we_act.prepare_mars_care_sampling_ablation manifest \
    --main-manifest "${main_manifest}" --visual-cache-root "${visual_cache}" \
    --spec "${spec}" --output "${manifest}" --smoke-output "${smoke_manifest}" \
    >"${logs}/manifest.log" 2>&1
fi

set_status SMOKE_BRANCH_COLLECTION "16 preregistered matched families; globally serial Vulkan"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
  --manifest "${smoke_manifest}" --checkpoint "${reference_checkpoint}" \
  --output-root "${families}" --robofactory-root "${robofactory}" \
  --device cuda:0 --render-device cuda:0 >"${logs}/smoke_branches.log" 2>&1

set_status SMOKE_SIGNAL_GATE "branch-signal density and effective-pair gate"
"${python_bin}" -m scripts.before_we_act.prepare_mars_care_sampling_ablation report \
  --main-family-root "${main_families}" --hybrid-family-root "${families}" \
  --manifest "${manifest}" --spec "${spec}" --smoke-only \
  --output "${root}/smoke_signal_report.json" >"${logs}/smoke_signal_report.log" 2>&1

if ! "${python_bin}" - "${root}/smoke_signal_report.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("signal_gate_passed") is True else 1)
PY
then
  set_status REJECTED_SIGNAL_GATE "hybrid did not materially increase preregistered smoke signal; no full ablation launched"
  exit 0
fi

set_status FULL_BRANCH_COLLECTION "signal gate passed; collect remaining independent 120-family ablation"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
  --manifest "${manifest}" --checkpoint "${reference_checkpoint}" \
  --output-root "${families}" --robofactory-root "${robofactory}" \
  --device cuda:0 --render-device cuda:0 >"${logs}/full_branches.log" 2>&1

"${python_bin}" -m scripts.before_we_act.prepare_mars_care_sampling_ablation report \
  --main-family-root "${main_families}" --hybrid-family-root "${families}" \
  --manifest "${manifest}" --spec "${spec}" \
  --output "${root}/full_signal_report.json" >"${logs}/full_signal_report.log" 2>&1

set_status PREPARE "quality audit and all-family scorer data"
"${python_bin}" -m scripts.before_we_act.prepare_mars_care_training quality \
  --family-root "${families}" --quality-root "${quality}" >"${logs}/quality.log" 2>&1
if [[ ! -f "${prepared}" ]]; then
  "${python_bin}" -m scripts.before_we_act.prepare_mars_care_training prepare \
    --family-root "${families}" --quality-root "${quality}" \
    --reference-checkpoint "${reference_checkpoint}" --output "${prepared}" \
    --manifest-output "${prepared_manifest}" >"${logs}/prepare.log" 2>&1
fi

set_status SCORERS "matched main recipe: four variants x three seeds x 4000 updates"
jobs=()
for variant in care reactive_only replay_only capacity; do
  for seed in 20260818 20260819 20260820; do jobs+=("${variant}:${seed}"); done
done
run_scorer() {
  local gpu=$1 spec_job=$2 variant seed output
  variant=${spec_job%%:*}; seed=${spec_job##*:}; output=${training}/${variant}/seed_${seed}
  if [[ -f "${output}/status.json" ]] && "${python_bin}" - "${output}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)
PY
  then return; fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m before_we_act.train_mars_care_belief \
    --prepared-data "${prepared}" --output "${output}" --seed "${seed}" \
    --variant "${variant}" --stage formal --updates 4000 --batch-size 48 \
    --eval-every 200 --save-every 200 --device cuda:0 \
    >"${logs}/scorer_${variant}_${seed}.log" 2>&1
}
wait_all() { local failed_job=0 pid; for pid in "$@"; do wait "${pid}" || failed_job=1; done; test "${failed_job}" -eq 0; }
for ((index=0; index<${#jobs[@]}; index+=4)); do
  pids=()
  for slot in 0 1 2 3; do run_scorer "${slot}" "${jobs[index+slot]}" & pids+=("$!"); done
  wait_all "${pids[@]}"
done

set_status CALIBRATE "matched offline selection and conformal calibration"
"${python_bin}" -m scripts.before_we_act.select_calibrate_mars_care \
  --settings "${settings}" --prepared-data "${prepared}" --training-root "${training}" \
  --reference-checkpoint "${reference_checkpoint}" --output-root "${offline}" \
  --device cuda:0 >"${logs}/calibrate.log" 2>&1

set_status VALIDATION20 "same 20 seeds/task and task-specific horizons as main CARE"
mkdir -p "${validation}"
for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
  case "${task}" in
    place_cube_in_cup|strike_cube_hard) max_steps=500 ;;
    three_robots_place_shoes) max_steps=1200 ;;
    four_robots_stack_cube) max_steps=800 ;;
  esac
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.evaluate_mars_care_closed_loop \
    --reference-checkpoint "${reference_checkpoint}" \
    --care-checkpoint "${offline}/care_deployment_checkpoint.pt" \
    --task "${task}" --robofactory-root "${robofactory}" \
    --output "${validation}/${task}.json" --episodes 20 --seed-start 20260827 \
    --max-steps "${max_steps}" --mode care --device cuda:0 --render-device cuda:0 \
    >"${logs}/validation20_${task}.log" 2>&1
done

"${python_bin}" -m scripts.before_we_act.prepare_mars_care_sampling_ablation final \
  --smoke-report "${root}/smoke_signal_report.json" \
  --main-validation-root "${run}/validation20/care" --hybrid-validation-root "${validation}" \
  --spec "${spec}" --output "${root}/final_report.json" >"${logs}/final_report.log" 2>&1
if "${python_bin}" - "${root}/final_report.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("eligible_for_next_formal_main_protocol") is True else 1)
PY
then
  set_status ELIGIBLE_NEXT_RUN "all preregistered gates passed; current run remains unchanged"
else
  set_status REJECTED_FINAL_GATE "final success gate failed; retain fixed stratified protocol"
fi
