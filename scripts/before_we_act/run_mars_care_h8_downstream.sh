#!/usr/bin/env bash
set -Eeuo pipefail

# Single-shot continuation of the H8 collection.  This process may wait for
# the collection supervisor, but it never restarts a failed stage and never
# touches the archived H1/formal run.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
py=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
rf=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
root=${MARS_CARE_H8_FORMAL_ROOT:-/workspace/runs/care_mars_optimization_v2/h8_fixed_stratified_formal_v1}
formal=${MARS_CARE_FORMAL_ROOT:-/workspace/runs/care_official_mars_v1}
families=${root}/families
receipt=${root}/corpus_receipt.json
prepared_root=${root}/prepared
quality=${prepared_root}/quality
prepared=${prepared_root}/care_h8_prepared.pt
prepared_manifest=${prepared_root}/care_h8_prepared_manifest.json
oof_root=${root}/oof_v3
oof_folds=${oof_root}/fold_manifest.json
oof_report=${oof_root}/aggregate.json
final=${root}/final_v3.pt
deployment=${root}/care_v3_deployment.pt
deployment_smoke=${root}/care_v3_deployment_smoke.pt
smoke=${root}/smoke
validation=${root}/validation20_v3
status=${root}/downstream_status.json
logs=${root}/downstream_logs
reference_source=${MARS_CARE_REFERENCE_CHECKPOINT:-${formal}/belief_selected/deployment_checkpoint.pt}
reference=${root}/contract/reference_action_contract.pt
manifest=${MARS_CARE_FAMILY_MANIFEST:-${formal}/contract/care_family_manifest.json}

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${logs}" "${prepared_root}"
cd "${repo}"

write_status() {
  local stage=$1 detail=${2:-} state=${3:-RUNNING}
  "${py}" - "${status}" "${stage}" "${detail}" "${state}" "${receipt}" "${prepared}" "${oof_report}" "${deployment}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); stage,detail,state=sys.argv[2:5]
receipt,prepared,oof,deployment=map(Path,sys.argv[5:9])
old=json.loads(path.read_text()) if path.is_file() else {}
history=list(old.get("history", [])); history.append({"time_utc":datetime.now(timezone.utc).isoformat(),"stage":stage,"detail":detail,"state":state})
value={"format_version":"before-we-act.care-mars-h8-downstream-status/1","stage":stage,"state":state,"detail":detail,"updated_at_utc":datetime.now(timezone.utc).isoformat(),"automatic_retry":False,"validation20_used_for_tuning":False,"legacy_h1_corpus_unchanged":True,"history":history}
for key,p in (("corpus_receipt",receipt),("prepared_data",prepared),("oof_report",oof),("deployment_checkpoint",deployment)):
    value[key]=str(p)
    if p.is_file():
        try: value[key+"_status"]=json.loads(p.read_text()).get("status") if p.suffix==".json" else "PRESENT"
        except Exception: value[key+"_status"]="PRESENT"
path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

failed() {
  local code=$?
  write_status "${current_stage:-UNKNOWN}" "exit=${code} line=${BASH_LINENO[0]}; no automatic retry" FAILED || true
  exit "${code}"
}
trap failed ERR

current_stage=WAIT_FOR_CORPUS
write_status WAIT_FOR_CORPUS "waiting for H8 corpus_receipt.json; collection supervisor owns GPU0" RUNNING
while [[ ! -f "${receipt}" ]]; do
  if [[ -f "${root}/status.json" ]] && "${py}" - "${root}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage")=="FAILED" else 1)
PY
  then
    write_status WAIT_FOR_CORPUS "H8 collection failed; downstream remains stopped" FAILED
    exit 2
  fi
  sleep 60
done
"${py}" - "${receipt}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get("status")!="PASSED" or int(x.get("family_count",-1))!=120 or int(x.get("intervention_steps",-1))!=8:
    raise SystemExit("H8 corpus receipt is not a complete PASSED 120-family H8 corpus")
PY

current_stage=H8_TRAINING_SMOKE
write_status "${current_stage}" "corpus passed; preparing all-family H8 tensors and model API smoke" RUNNING
"${py}" -m scripts.before_we_act.migrate_mars_reference_checkpoint --source "${reference_source}" --output "${reference}" >"${logs}/reference_contract_migration.log" 2>&1
"${py}" - "${receipt}" "${reference}" <<'PY'
import json,sys,torch
r=json.load(open(sys.argv[1])); c=torch.load(sys.argv[2],map_location="cpu",weights_only=False)
assert c.get("source_checkpoint_sha256")==r.get("reference_checkpoint_sha256")
PY
if [[ ! -f "${prepared}" ]]; then
  "${py}" -m scripts.before_we_act.prepare_mars_care_training quality --family-root "${families}" --quality-root "${quality}" >"${logs}/quality.log" 2>&1
  "${py}" -m scripts.before_we_act.prepare_mars_care_training prepare --family-root "${families}" --quality-root "${quality}" --reference-checkpoint "${reference}" --output "${prepared}" --manifest-output "${prepared_manifest}" >"${logs}/prepare.log" 2>&1
fi
"${py}" - "${prepared}" "${prepared_manifest}" <<'PY'
import json,sys,torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); m=json.load(open(sys.argv[2]))
assert p.get("intervention_steps")==8 and m.get("intervention_steps")==8
assert p["memory"].shape[0]==120 and p["candidate_chunks"].shape[1:]==(6,100,8)
PY
if [[ ! -f "${smoke}/model_4_updates.pt" ]]; then
  "${py}" -m scripts.before_we_act.run_mars_care_oof_v3 fit-final --prepared-data "${prepared}" --output "${smoke}/model_4_updates.pt" --seed 20260903 --updates 4 --batch-size 16 --action-prefix-steps 8 --device cuda:0 >"${logs}/training_smoke.log" 2>&1
else
  echo '{"status":"reused","output":"'"${smoke}/model_4_updates.pt"'"}' >"${logs}/training_smoke.log"
fi
write_status "${current_stage}" "H8 prepared-data and 4-update CARE-v3 training smoke passed" PASSED

current_stage=H8_CLOSED_LOOP_SMOKE
write_status "${current_stage}" "pre-formal four-task 2-step selector-off/decentralized interface smoke" RUNNING
pre_smoke="${smoke}/preformal"
pre_deployment="${pre_smoke}/care_v3_interface_smoke.pt"
mkdir -p "${pre_smoke}"
if [[ ! -f "${pre_deployment}" ]]; then
  "${py}" scripts/before_we_act/build_mars_care_v3_deployment.py --prepared-data "${prepared}" --final-checkpoint "${smoke}/model_4_updates.pt" --reference-checkpoint "${reference}" --output "${pre_deployment}" --promotion-scope smoke --interface-smoke >"${logs}/preformal_deployment.log" 2>&1
fi
for mode in selector_off decentralized; do
  for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
    out="${pre_smoke}/${mode}/${task}.json"
    [[ -f "${out}" ]] && continue
    CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_care_closed_loop_v2 --reference-checkpoint "${reference}" --care-v3-checkpoint "${pre_deployment}" --task "${task}" --robofactory-root "${rf}" --output "${out}" --episodes 1 --seed-start $((20260950 + ${#task})) --max-steps 2 --mode "${mode}" --device cuda:0 --render-device cuda:0 >"${logs}/preformal_${mode}_${task}.log" 2>&1
  done
done
"${py}" - "${pre_smoke}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); tasks=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
for mode in ("selector_off","decentralized"):
    for task in tasks:
        x=json.loads((root/mode/f"{task}.json").read_text())
        if x.get("status")!="complete" or x.get("episodes")!=1 or x.get("rows", [{}])[0].get("steps")!=2 or x.get("strict_decentralized") is not True:
            raise SystemExit(f"pre-formal H8 closed-loop smoke failed: {mode}/{task}")
(root/"receipt.json").write_text(json.dumps({"status":"PASSED","passed":True,"tasks":tasks,"modes":["selector_off","decentralized"],"steps":2,"validation20_used_for_tuning":False},indent=2)+"\n")
PY
write_status "${current_stage}" "H8 four-task closed-loop smoke passed before OOF formal training" PASSED

current_stage=OOF_FOLDS
write_status "${current_stage}" "strict 5-fold x 3-seed OOF, all H8/H16/H32/H64 horizons, 4 GPU workers" RUNNING
if [[ ! -f "${oof_folds}" ]]; then
  "${py}" -m scripts.before_we_act.run_mars_care_oof_v3 folds --prepared-data "${prepared}" --output "${oof_folds}" --n-splits 5 --fold-seed 20260901 >"${logs}/folds.log" 2>&1
fi
jobs=()
for seed in 20260904 20260905 20260906; do for fold in 0 1 2 3 4; do jobs+=("${fold}:${seed}"); done; done
job_ok() { [[ -f "${oof_root}/fold_$1/seed_$2/checkpoint.pt" && -f "${oof_root}/fold_$1/seed_$2/predictions.json" ]]; }
run_job() {
  local gpu=$1 spec=$2 fold seed
  fold=${spec%%:*}
  seed=${spec##*:}
  if job_ok "${fold}" "${seed}"; then return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${py}" -m scripts.before_we_act.run_mars_care_oof_v3 train-fold --prepared-data "${prepared}" --output-root "${oof_root}" --fold "${fold}" --seed "${seed}" --n-splits 5 --fold-seed 20260901 --updates 4000 --batch-size 48 --action-prefix-steps 8 --device cuda:0 >"${logs}/oof_fold_${fold}_seed_${seed}.log" 2>&1
}
for ((index=0; index<${#jobs[@]}; index+=4)); do
  pids=()
  for slot in 0 1 2 3; do
    [[ $((index+slot)) -lt ${#jobs[@]} ]] || break
    run_job "${slot}" "${jobs[index+slot]}" & pids+=("$!")
  done
  failed_job=0; for pid in "${pids[@]}"; do wait "${pid}" || failed_job=1; done
  [[ "${failed_job}" -eq 0 ]]
done

current_stage=OOF_AGGREGATE
write_status "${current_stage}" "aggregating leakage-checked all-horizon OOF and simultaneous calibration" RUNNING
"${py}" -m scripts.before_we_act.run_mars_care_oof_v3 aggregate --prepared-data "${prepared}" --output-root "${oof_root}" --output "${oof_report}" --seeds 20260904,20260905,20260906 --n-splits 5 --fold-seed 20260901 --nominal 0.90 >"${logs}/oof_aggregate.log" 2>&1
"${py}" - "${oof_report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); c=x.get("calibration",{})
assert x.get("horizon_oof_complete") is True and c.get("family_max_includes_all_requested_horizons") is True
assert x.get("family_disjoint") is True and x.get("validation20_used_for_tuning") is False
PY
write_status "${current_stage}" "strict all-horizon OOF and calibration passed" PASSED

current_stage=FINAL_TRAINING
write_status "${current_stage}" "four independent 4000-update all-family CARE-v3 fits on four RTX 5090 GPUs after OOF freeze" RUNNING
if [[ ! -f "${final}" ]]; then
  final_pids=()
  for slot in 0 1 2 3; do
    seed=$((20260907 + slot))
    candidate="${root}/final_v3_seed_${seed}.pt"
    if [[ ! -f "${candidate}" ]]; then
      CUDA_VISIBLE_DEVICES="${slot}" "${py}" -m scripts.before_we_act.run_mars_care_oof_v3 fit-final --prepared-data "${prepared}" --output "${candidate}" --seed "${seed}" --updates 4000 --batch-size 48 --action-prefix-steps 8 --device cuda:0 >"${logs}/final_training_seed_${seed}.log" 2>&1 &
      final_pids+=("$!")
    fi
  done
  final_failed=0; for pid in "${final_pids[@]}"; do wait "${pid}" || final_failed=1; done
  [[ "${final_failed}" -eq 0 ]]
  "${py}" - "${root}" "${final}" <<'PY'
import json,os,sys
from pathlib import Path
root,out=Path(sys.argv[1]),Path(sys.argv[2])
paths=sorted(root.glob("final_v3_seed_*.pt"))
if len(paths)!=4: raise SystemExit(f"expected four final seed fits, got {len(paths)}")
import torch
rows=[]
for p in paths:
    value=torch.load(p,map_location="cpu",weights_only=False)
    rows.append((float(value.get("last_loss",float("inf"))),int(value["seed"]),p,value))
# The deploy seed is pre-registered; training loss is recorded for audit but
# never used to select against an unseen validation set.
selected = [row for row in rows if row[1] == 20260907]
if len(selected) != 1: raise SystemExit("pre-registered final seed 20260907 is missing")
loss,seed,path,value=selected[0]
value=dict(value); value["selected_from_parallel_final_seeds"]=[{"seed":s,"last_loss":l,"checkpoint":str(p)} for l,s,p,_ in rows]; value["selected_final_seed"]=seed
tmp=out.with_name(f".{out.name}.{os.getpid()}.tmp"); torch.save(value,tmp); os.replace(tmp,out)
(root/"final_seed_selection.json").write_text(json.dumps({"status":"PASSED","selected_seed":seed,"selected_last_loss":loss,"candidates":[{"seed":s,"last_loss":l,"checkpoint":str(p)} for l,s,p,_ in rows],"validation20_used_for_tuning":False},indent=2)+"\n")
PY
fi
write_status "${current_stage}" "four all-family 4000-update fits complete; pre-registered seed 20260907 selected" PASSED

current_stage=DEPLOYMENT_BUILD
write_status "${current_stage}" "building fail-closed H8 V3 smoke deployment artifact" RUNNING
if [[ ! -f "${deployment_smoke}" ]]; then
  "${py}" scripts/before_we_act/build_mars_care_v3_deployment.py --prepared-data "${prepared}" --final-checkpoint "${final}" --oof-report "${oof_report}" --reference-checkpoint "${reference}" --output "${deployment_smoke}" --promotion-scope smoke >"${logs}/deployment_smoke.log" 2>&1
fi
write_status "${current_stage}" "V3 smoke deployment artifact built; H8 prefix and OOF provenance pinned" PASSED

current_stage=CLOSED_LOOP_SMOKE
write_status "${current_stage}" "paired selector-off vs decentralized CARE on independent seeds; task max-step contracts enforced" RUNNING
for mode in selector_off decentralized; do
  for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
    case "${task}" in
      place_cube_in_cup|strike_cube_hard) max_steps=500;;
      three_robots_place_shoes) max_steps=1200;;
      four_robots_stack_cube) max_steps=800;;
    esac
    out="${smoke}/${mode}/${task}.json"
    [[ -f "${out}" ]] && continue
    CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_care_closed_loop_v2 --reference-checkpoint "${reference}" --care-v3-checkpoint "${deployment_smoke}" --task "${task}" --robofactory-root "${rf}" --output "${out}" --episodes 1 --seed-start $((20261000 + ${#task})) --max-steps "${max_steps}" --mode "${mode}" --device cuda:0 --render-device cuda:0 >"${logs}/smoke_${mode}_${task}.log" 2>&1
  done
done
"${py}" - "${smoke}" <<'PY'
import json,sys
from pathlib import Path
tasks=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
root=Path(sys.argv[1]); rows={}
for mode in ("selector_off","decentralized"):
    rows[mode]={}
    for t in tasks:
        x=json.loads((root/mode/f"{t}.json").read_text())
        assert x.get("status")=="complete" and x.get("episodes")==1 and x.get("strict_decentralized") is True
        rows[mode][t]=x
care_success=sum(int(rows["decentralized"][t]["successes"]) for t in tasks)
base_success=sum(int(rows["selector_off"][t]["successes"]) for t in tasks)
overrides=sum(int(r["overrides"]) for t in tasks for r in rows["decentralized"][t]["rows"])
conflicts=sum(int(r["simultaneous_override_conflict_steps"]) for t in tasks for r in rows["decentralized"][t]["rows"])
passed=care_success>=base_success and overrides>0 and conflicts==0
report={"status":"PASSED" if passed else "FAILED","passed":passed,"tasks":tasks,"paired_same_seeds":True,"selector_off_successes":base_success,"decentralized_care_successes":care_success,"care_overrides":overrides,"simultaneous_override_conflict_steps":conflicts,"gate":{"no_smoke_success_regression":care_success>=base_success,"effective_override_present":overrides>0,"no_override_conflict":conflicts==0},"rows":rows,"validation20_used_for_tuning":False}
(root/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
raise SystemExit(0 if passed else 2)
PY
write_status "${current_stage}" "paired four-task selector-off/decentralized smoke gate passed" PASSED

current_stage=FORMAL_DEPLOYMENT_BUILD
write_status "${current_stage}" "promoting the same checkpoint only after independent smoke gate" RUNNING
if [[ ! -f "${deployment}" ]]; then
  "${py}" scripts/before_we_act/build_mars_care_v3_deployment.py --prepared-data "${prepared}" --final-checkpoint "${final}" --oof-report "${oof_report}" --paired-smoke-report "${smoke}/receipt.json" --reference-checkpoint "${reference}" --output "${deployment}" --promotion-scope formal >"${logs}/deployment_formal.log" 2>&1
fi
write_status "${current_stage}" "formal V3 deployment artifact admitted after smoke" PASSED

current_stage=VALIDATION20
write_status "${current_stage}" "paired selector-off and decentralized CARE, 20 untouched seeds/task, globally serial Vulkan" RUNNING
for mode in selector_off decentralized; do
  mkdir -p "${validation}/${mode}"
  for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
    case "${task}" in
      place_cube_in_cup|strike_cube_hard) max_steps=500;;
      three_robots_place_shoes) max_steps=1200;;
      four_robots_stack_cube) max_steps=800;;
    esac
    out="${validation}/${mode}/${task}.json"
    [[ -f "${out}" ]] && continue
    CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_care_closed_loop_v2 --reference-checkpoint "${reference}" --care-v3-checkpoint "${deployment}" --task "${task}" --robofactory-root "${rf}" --output "${out}" --episodes 20 --seed-start 20261200 --max-steps "${max_steps}" --mode "${mode}" --device cuda:0 --render-device cuda:0 >"${logs}/validation20_${mode}_${task}.log" 2>&1
  done
  "${py}" -m scripts.before_we_act.summarize_mars_care_validation --root "${validation}/${mode}" --mode "${mode}" --output "${validation}/${mode}/summary.json"
done
write_status "${current_stage}" "new paired four-task Validation20 complete" PASSED

current_stage=COMPLETE
write_status "${current_stage}" "${validation}/decentralized/summary.json" COMPLETE
