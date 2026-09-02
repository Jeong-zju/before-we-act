#!/usr/bin/env bash
set -Eeuo pipefail

# Explicitly authorized exploratory path.  This is intentionally isolated
# from the formal run and never writes formal deployment/Validation20 files.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
py=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
rf=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
formal_root=${MARS_CARE_H8_FORMAL_ROOT:-/workspace/runs/care_mars_optimization_v2/h8_fixed_stratified_formal_v1}
root=${MARS_CARE_EXPLORATORY_ROOT:-${formal_root}/exploratory_validation20}
prepared=${formal_root}/prepared/care_h8_prepared.pt
oof=${formal_root}/oof_v3/aggregate.json
final=${formal_root}/final_v3.pt
reference=${formal_root}/contract/reference_action_contract.pt
deployment=${root}/care_v3_exploratory_bypass.pt
validation=${root}/validation20
status=${root}/status.json
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${root}" "${logs}" "${validation}"
cd "${repo}"

write_status() {
  local stage=$1 detail=${2:-} state=${3:-RUNNING}
  "${py}" - "${status}" "${stage}" "${detail}" "${state}" "${deployment}" "${validation}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); stage, detail, state=sys.argv[2:5]
deployment, validation=Path(sys.argv[5]), Path(sys.argv[6])
old=json.loads(path.read_text()) if path.is_file() else {}
history=list(old.get("history", [])); now=datetime.now(timezone.utc).isoformat()
history.append({"time_utc": now, "stage": stage, "detail": detail, "state": state})
value={"format_version":"before-we-act.care-mars-exploratory-status/1",
       "stage":stage,"state":state,"detail":detail,"updated_at_utc":now,
       "admission_bypassed":True,"formal_gate_preserved":True,
       "validation20_used_for_tuning":False,"history":history,
       "deployment":str(deployment),"validation_root":str(validation)}
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2)+"\n"); os.replace(tmp,path)
PY
}

failed() {
  local code=$?
  write_status "${current_stage:-UNKNOWN}" "exit=${code}; exploratory run stopped" FAILED || true
  exit "${code}"
}
trap failed ERR

current_stage=PREFLIGHT
write_status "${current_stage}" "checking immutable prepared/OOF/final/reference inputs; formal artifacts untouched" RUNNING
for p in "${prepared}" "${oof}" "${final}" "${reference}"; do [[ -f "${p}" ]] || { echo "missing exploratory input: ${p}" >&2; exit 2; }; done
"${py}" - "${oof}" "${final}" "${prepared}" <<'PY'
import json,sys,torch
oof=json.load(open(sys.argv[1])); final=torch.load(sys.argv[2],map_location="cpu",weights_only=False); prepared=torch.load(sys.argv[3],map_location="cpu",weights_only=False)
assert oof.get("status")=="COMPLETE" and oof.get("horizon_oof_complete") is True
assert final.get("format_version")=="before-we-act.care-mars-final-training-v3/1"
assert final.get("prepared_data_sha256")
assert prepared.get("intervention_steps")==8
PY

current_stage=EXPLORATORY_DEPLOYMENT
write_status "${current_stage}" "building isolated deployment with explicit admission bypass; formal deployment remains absent" RUNNING
if [[ ! -f "${deployment}" ]]; then
  "${py}" scripts/before_we_act/build_mars_care_v3_deployment.py \
    --prepared-data "${prepared}" --final-checkpoint "${final}" --oof-report "${oof}" \
    --reference-checkpoint "${reference}" --output "${deployment}" \
    --promotion-scope exploratory >"${logs}/deployment.log" 2>&1
fi
"${py}" - "${deployment}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
assert x.get("provenance",{}).get("admission_bypassed") is True
assert x.get("provenance",{}).get("promotion_scope")=="exploratory"
PY
write_status "${current_stage}" "isolated exploratory deployment built and provenance-marked" PASSED

current_stage=VALIDATION20
write_status "${current_stage}" "exploratory paired selector-off/decentralized Validation20; untouched seeds" RUNNING
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
    CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_care_closed_loop_v2 \
      --reference-checkpoint "${reference}" --care-v3-checkpoint "${deployment}" \
      --task "${task}" --robofactory-root "${rf}" --output "${out}" --episodes 20 \
      --seed-start 20261300 --max-steps "${max_steps}" --mode "${mode}" --device cuda:0 \
      --render-device cuda:0 >"${logs}/validation20_${mode}_${task}.log" 2>&1
  done
  "${py}" -m scripts.before_we_act.summarize_mars_care_validation \
    --root "${validation}/${mode}" --mode "${mode}" --output "${validation}/${mode}/summary.json"
done

current_stage=COMPLETE
write_status "${current_stage}" "exploratory Validation20 complete; admission_bypassed=true" COMPLETE
"${py}" - "${root}" "${oof}" <<'PY'
import json,sys
from pathlib import Path
root,oof=Path(sys.argv[1]),Path(sys.argv[2]); report=json.loads(oof.read_text())
payload={"format_version":"before-we-act.care-mars-exploratory-validation20/1",
         "status":"COMPLETE","admission_bypassed":True,
         "formal_gate_eligible_for_deployment":bool(report.get("calibration",{}).get("eligible_for_deployment",False)),
         "validation20_used_for_tuning":False,
         "modes":{"selector_off":str(root/"validation20/selector_off/summary.json"),
                  "decentralized":str(root/"validation20/decentralized/summary.json")}}
(root/"receipt.json").write_text(json.dumps(payload,indent=2)+"\n")
PY
