#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot corrected H8 parity smoke.  It never overwrites v1 evidence and
# never starts a scorer/belief-head fit.  Vulkan remains globally serial.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
robofactory=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
formal=${MARS_CARE_FORMAL_ROOT:-/workspace/runs/care_official_mars_v1}
reference=${MARS_CARE_REFERENCE_CHECKPOINT:-${formal}/belief_selected/deployment_checkpoint.pt}
manifest=${MARS_CARE_FAMILY_MANIFEST:-${formal}/contract/care_family_manifest.json}
baseline=${MARS_CARE_BRANCH_DURATION_ROOT:-/workspace/runs/care_mars_optimization_v2/branch_duration_smoke_v1}/duration_8/families
root=${MARS_CARE_BRANCH_PARITY_ROOT:-/workspace/runs/care_mars_optimization_v2/branch_duration_parity_h8_v1}
status=${root}/status.json
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${root}" "${logs}"
cd "${repo}"

tasks=(place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube)

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${reference}" "${manifest}" "${baseline}" <<'PY'
from datetime import datetime, timezone
import hashlib, json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); stage,detail=sys.argv[2],sys.argv[3]
reference,manifest,baseline=Path(sys.argv[4]),Path(sys.argv[5]),Path(sys.argv[6])
old=json.loads(path.read_text()) if path.exists() else {}
history=list(old.get("history",[])); now=datetime.now(timezone.utc).isoformat()
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
value={"format_version":"before-we-act.care-mars-branch-parity-status/1","stage":stage,
"detail":detail,"updated_at_utc":now,"duration":8,
"tasks":["place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube"],
"baseline_root":str(baseline),"reference_checkpoint":str(reference),
"reference_checkpoint_sha256":digest(reference),"family_manifest":str(manifest),
"family_manifest_sha256":digest(manifest),"validation20_used_for_tuning":False,
"main_protocol_unchanged":True,"automatic_retry":False,"globally_serial_vulkan":True,
"history":history+[{"time_utc":now,"stage":stage,"detail":detail}]}
path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

failed() { local code=$?; write_status FAILED "exit=${code} line=${BASH_LINENO[0]}; no automatic retry" || true; exit "${code}"; }
trap failed ERR

if [[ ! -x "${python_bin}" || ! -f "${reference}" || ! -f "${manifest}" ]]; then
  write_status FAILED "missing Python/reference/manifest; no automatic retry"; exit 2
fi
if [[ "$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')" -ne 4 ]]; then
  write_status FAILED "expected four visible GPUs; no automatic retry"; exit 2
fi
if [[ -f "${status}" ]] && "${python_bin}" - "${status}" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage")=="COMPLETE" else 1)
PY
then echo "parity smoke already complete; preserving ${root}"; exit 0; fi
if [[ -f "${root}/audit.json" ]]; then write_status FAILED "refusing to overwrite parity audit"; exit 2; fi

write_status RUNNING "corrected H8 restore parity smoke; globally serial Vulkan on GPU0"
for task in "${tasks[@]}"; do
  write_status RUNNING "duration=8 task=${task} fixed-family-count=1"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
    --manifest "${manifest}" --checkpoint "${reference}" \
    --output-root "${root}/families" --robofactory-root "${robofactory}" \
    --task "${task}" --limit 1 --intervention-steps 8 \
    --device cuda:0 --render-device cuda:0 \
    >"${logs}/duration_8_${task}.log" 2>&1
done

write_status AUDIT "comparing corrected execution with immutable v1 H8 and auditing qpos parity"
"${python_bin}" -m scripts.before_we_act.analyze_mars_care_branch_parity \
  --baseline-root "${baseline}" --corrected-root "${root}/families" \
  --baseline-h1-audit "${MARS_CARE_BRANCH_DURATION_ROOT:-/workspace/runs/care_mars_optimization_v2/branch_duration_smoke_v1}/audit.json" \
  --output "${root}/audit.json" >"${logs}/audit.log" 2>&1
write_status COMPLETE "corrected H8 parity audit complete; no scorer training was started"
