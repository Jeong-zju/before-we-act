#!/usr/bin/env bash
set -Eeuo pipefail

# Duration-only ablation on the same fixed-stratified family identities.  The
# completed formal H1 corpus and Validation20 are read-only and never changed.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
robofactory=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
formal=${MARS_CARE_FORMAL_ROOT:-/workspace/runs/care_official_mars_v1}
reference=${MARS_CARE_REFERENCE_CHECKPOINT:-${formal}/belief_selected/deployment_checkpoint.pt}
manifest=${MARS_CARE_FAMILY_MANIFEST:-${formal}/contract/care_family_manifest.json}
root=${MARS_CARE_BRANCH_DURATION_ROOT:-/workspace/runs/care_mars_optimization_v2/branch_duration_smoke_v1}
status=${root}/status.json
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${root}" "${logs}"
cd "${repo}"

durations=(1 4 8 16)
tasks=(place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube)

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${reference}" "${manifest}" <<'PY'
from datetime import datetime, timezone
import hashlib, json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); stage,detail=sys.argv[2],sys.argv[3]; reference,manifest=Path(sys.argv[4]),Path(sys.argv[5])
old=json.loads(path.read_text()) if path.exists() else {}; history=list(old.get("history",[])); now=datetime.now(timezone.utc).isoformat()
history.append({"time_utc":now,"stage":stage,"detail":detail})
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
value={"format_version":"before-we-act.care-mars-branch-duration-status/1","stage":stage,"detail":detail,"updated_at_utc":now,"durations":[1,4,8,16],"tasks":["place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube"],"families_per_task_duration":1,"reference_checkpoint":str(reference),"reference_checkpoint_sha256":digest(reference),"family_manifest":str(manifest),"family_manifest_sha256":digest(manifest),"fixed_stratified_main_protocol_unchanged":True,"validation20_used_for_tuning":False,"automatic_retry":False,"globally_serial_vulkan":True,"history":history}
path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

failed() { local code=$?; write_status FAILED "exit=${code} line=${BASH_LINENO[0]}; prefix retained; no automatic retry" || true; exit "${code}"; }
trap failed ERR

if [[ ! -x "${python_bin}" || ! -f "${reference}" || ! -f "${manifest}" ]]; then
  write_status FAILED "missing Python/reference/manifest; no automatic retry"; exit 2
fi
if [[ $("${python_bin}" -c 'import torch; print(torch.cuda.device_count())') -ne 4 ]]; then
  write_status FAILED "expected four visible RTX 5090 GPUs; no automatic retry"; exit 2
fi
if [[ -f "${status}" ]] && "${python_bin}" - "${status}" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage")=="COMPLETE" else 1)
PY
then echo "branch-duration smoke already complete; preserving ${root}"; exit 0; fi
if [[ -f "${root}/audit.json" ]]; then write_status FAILED "refusing to overwrite duration audit"; exit 2; fi

write_status RUNNING "duration-only H1/H4/H8/H16 smoke; globally serial Vulkan on GPU0"
for duration in "${durations[@]}"; do
  for task in "${tasks[@]}"; do
    output=${root}/duration_${duration}/families
    mkdir -p "${output}"
    write_status RUNNING "duration=${duration} task=${task} fixed-family-count=1"
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.mars_care_branch_collector \
      --manifest "${manifest}" --checkpoint "${reference}" \
      --output-root "${output}" --robofactory-root "${robofactory}" \
      --task "${task}" --limit 1 --intervention-steps "${duration}" \
      --device cuda:0 --render-device cuda:0 \
      >"${logs}/duration_${duration}_${task}.log" 2>&1
  done
done

write_status AUDIT "computing effective D/R pair density and parity gates"
"${python_bin}" -m scripts.before_we_act.analyze_mars_care_branch_duration \
  --root "${root}" --output "${root}/audit.json" >"${logs}/audit.log" 2>&1
write_status COMPLETE "branch-duration signal audit complete; no scorer training was started"

