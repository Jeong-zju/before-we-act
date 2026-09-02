#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot full H8 collection on the immutable 120-family fixed-stratified
# manifest.  The Vulkan driver has proven globally serial, so collection uses
# GPU0; later scorer/OOF stages use all four GPUs.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
robofactory=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
formal=${MARS_CARE_FORMAL_ROOT:-/workspace/runs/care_official_mars_v1}
manifest=${MARS_CARE_FAMILY_MANIFEST:-${formal}/contract/care_family_manifest.json}
gate=${MARS_CARE_H8_GATE:-/workspace/runs/care_mars_optimization_v2/branch_duration_parity_h8_v1/audit.json}
root=${MARS_CARE_H8_FORMAL_ROOT:-/workspace/runs/care_mars_optimization_v2/h8_fixed_stratified_formal_v1}
families=${root}/families
status=${root}/status.json
contract=${root}/preregistered_contract.json
receipt=${root}/corpus_receipt.json
logs=${root}/logs
reference_source=${MARS_CARE_REFERENCE_SOURCE_CHECKPOINT:-${formal}/belief_selected/deployment_checkpoint.pt}
reference=${MARS_CARE_REFERENCE_CHECKPOINT:-${root}/contract/reference_action_contract.pt}

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${families}" "${logs}"
cd "${repo}"

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${families}" "${reference_source}" "${manifest}" "${gate}" <<'PY'
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); stage,detail=sys.argv[2],sys.argv[3]; families=Path(sys.argv[4])
reference,manifest,gate=map(Path,sys.argv[5:8]); old=json.loads(path.read_text()) if path.exists() else {}
history=list(old.get("history",[])); now=datetime.now(timezone.utc).isoformat()
counts=Counter(p.parent.name for p in families.glob("*/*.json")); completed=sum(counts.values())
elapsed=sum(float(json.load(open(p)).get("wall_seconds",0.0)) for p in families.glob("*/*.json"))
mean=elapsed/completed if completed else 180.0; remaining=max(0,120-completed)*mean
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
history.append({"time_utc":now,"stage":stage,"detail":detail,"completed_families":completed})
value={"format_version":"before-we-act.care-mars-h8-formal-status/1","stage":stage,"detail":detail,
"updated_at_utc":now,"completed_families":completed,"target_families":120,
"completed_by_task":{k:counts[k] for k in ["place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube"]},
"estimated_remaining_seconds":remaining,"intervention_steps":8,"families_per_task":30,"branches_per_family":24,
"reference_checkpoint":str(reference),"reference_checkpoint_sha256":digest(reference),
"family_manifest":str(manifest),"family_manifest_sha256":digest(manifest),
"admission_gate":str(gate),"admission_gate_sha256":digest(gate),
"fixed_stratified":True,"validation20_used_for_tuning":False,"legacy_h1_corpus_unchanged":True,
"automatic_retry":False,"globally_serial_vulkan":True,"history":history}
path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

failed() {
  local code=$?
  write_status FAILED "exit=${code} line=${BASH_LINENO[0]}; completed prefix retained; no automatic retry" || true
  exit "${code}"
}
trap failed ERR

if [[ ! -x "${python_bin}" || ! -f "${reference_source}" || ! -f "${manifest}" || ! -f "${gate}" ]]; then
  write_status FAILED "missing Python/reference-source/manifest/H8 gate; no automatic retry"
  exit 2
fi
if [[ ! -f "${reference}" ]]; then
  "${python_bin}" -m scripts.before_we_act.migrate_mars_reference_checkpoint \
    --source "${reference_source}" --output "${reference}"
fi
"${python_bin}" - "${reference}" <<'PY'
import sys, torch
from before_we_act.mars_action_contract import validate_checkpoint_action_contract
value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
validate_checkpoint_action_contract(value)
PY
reference_source_sha256=$(sha256sum "${reference_source}" | awk '{print $1}')
"${python_bin}" - "${reference}" "${reference_source_sha256}" <<'PY'
import sys, torch
value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if value.get("source_checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("reference action-contract wrapper source hash drifted")
PY
if [[ "$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')" -ne 4 ]]; then
  write_status FAILED "expected four visible RTX 5090 GPUs; no automatic retry"
  exit 2
fi
"${python_bin}" - "${gate}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); a=x.get("aggregate",{})
checks=(x.get("format_version")=="before-we-act.care-mars-branch-parity-audit/2",
        x.get("eligible_for_scorer_smoke") is True,a.get("family_count")==4,
        a.get("execution_parity") is True,a.get("support_complete") is True,
        a.get("all_candidates_legal") is True,a.get("maximum_restore_error")==0.0,
        a.get("maximum_replay_teammate_action_error")==0.0,
        a.get("maximum_candidate0_reference_action_error")==0.0,
        a.get("hard_safety_pair_count")==0,a.get("signal_density_gain_over_h1",0)>=0.10,
        a.get("signal_density_ratio_over_h1",0)>=1.5)
raise SystemExit(0 if all(checks) else 2)
PY
if [[ -f "${status}" ]] && "${python_bin}" - "${status}" "${receipt}" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])) if __import__('pathlib').Path(sys.argv[2]).is_file() else {}
raise SystemExit(0 if s.get("stage")=="COMPLETE" and r.get("status")=="PASSED" else 1)
PY
then echo "H8 formal corpus already complete; preserving ${root}"; exit 0; fi

"${python_bin}" - "${contract}" "${reference_source}" "${manifest}" "${gate}" "${root}" <<'PY'
from datetime import datetime,timezone
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); reference,manifest,gate,root=map(Path,sys.argv[2:6])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
value={"format_version":"before-we-act.care-mars-h8-preregistered-contract/1","created_at_utc":datetime.now(timezone.utc).isoformat(),
"sampling":"frozen fixed-stratified 20 critical + 10 uniform per task","family_count":120,"families_per_task":30,
"branches_per_family":24,"intervention_steps":8,"outcome_horizons":[8,16,32,64],
"reference_policy":"B-core/TUNE","reference_checkpoint":str(reference),"reference_checkpoint_sha256":digest(reference),
"family_manifest":str(manifest),"family_manifest_sha256":digest(manifest),"admission_gate":str(gate),"admission_gate_sha256":digest(gate),
"output_root":str(root),"all_family_training":True,"validation20_used_for_tuning":False,"legacy_h1_corpus_unchanged":True,
"automatic_retry":False,"globally_serial_vulkan":True}
if path.exists():
 old=json.load(open(path)); comparable={k:v for k,v in old.items() if k!="created_at_utc"}; expected={k:v for k,v in value.items() if k!="created_at_utc"}
 if comparable!=expected: raise SystemExit("refusing H8 preregistration drift")
else:
 tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY

write_status RUNNING "full 120-family H8 collection; globally serial Vulkan on GPU0"
for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
  write_status RUNNING "task=${task}; collecting/reusing frozen 30-family H8 prefix"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
    --manifest "${manifest}" --checkpoint "${reference}" \
    --checkpoint-identity-sha256 "${reference_source_sha256}" \
    --output-root "${families}" --robofactory-root "${robofactory}" \
    --task "${task}" --intervention-steps 8 \
    --device cuda:0 --render-device cuda:0 \
    >"${logs}/${task}.log" 2>&1
done

write_status VERIFYING "strict 120-family tensor, support, restore, replay and reference parity audit"
"${python_bin}" -m scripts.before_we_act.verify_mars_care_h8_corpus \
  --manifest "${manifest}" --family-root "${families}" \
  --reference-checkpoint "${reference_source}" --intervention-steps 8 \
  --output "${receipt}" >"${logs}/verify.log" 2>&1
write_status COMPLETE "H8 fixed-stratified corpus passed; downstream preparation may start"
