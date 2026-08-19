#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_A4_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_STEP4R_RUN_ROOT:-/workspace/bwa_runs/a4-step4r-bounded-bt-beta0-25k-v1}"
BCORE_ROOT="${BWA_BCORE_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
REPAIR_ROOT="${BWA_A4_REPAIR_RUN_ROOT:-/workspace/bwa_runs/a4-bt-bottleneck-repair-v1}"
SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
ACTION_CACHE="${BWA_ACTION_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n2-action-context-v1}"
SCENARIO_SPLIT="${BWA_SCENARIO_SPLIT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief/contract/scenario_split.json}"
PYTHON="${BWA_A4_PYTHON:-/venv/robofactory-act/bin/python}"
A4_CONTRACT="${RUN_ROOT}/contract/a4_pilot_contract.json"
STABILITY_CONTRACT="${RUN_ROOT}/contract/step4r_stability_contract.json"
CANDIDATE_ROOT="${RUN_ROOT}/training/a4_no_bottleneck/seed_20260818"
PIPELINE_STATUS="${RUN_ROOT}/pipeline_status.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:/workspace/RoboFactory:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs"
cd "${ROOT}"

write_status() {
  local state="$1" stage="$2" detail="$3"
  "${PYTHON}" - "${PIPELINE_STATUS}" "${state}" "${stage}" "${detail}" <<'PY'
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,p)
PY
}

on_error() {
  local code=$?
  write_status FAILED error "Step-4R pipeline exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'write_status STOPPED interrupted signal; exit 130' INT TERM

if [[ -e "${A4_CONTRACT}" || -e "${CANDIDATE_ROOT}" ]]; then
  echo "Step-4R run root already contains a contract or candidate: ${RUN_ROOT}" >&2
  exit 2
fi

write_status RUNNING contract "freezing fresh-seed bounded-BT beta=0 25k contract"
"${PYTHON}" scripts/before_we_act/prepare_base_relative_belief.py \
  --base-n2-contract "${BCORE_ROOT}/contract/n2_contract.json" \
  --source-root "${ROOT}" --phase pilot --beta-b 0.0 \
  --pilot-seed 20260818 --pilot-updates 25000 --pilot-eval-every 5000 \
  --prior-fit 0.1 --bradley-terry 0.01 --temperature 0.005 \
  --margin-fraction 0.1 --margin-cap 0.01 --output "${A4_CONTRACT}"

"${PYTHON}" scripts/before_we_act/prepare_step4r_stability.py \
  --a4-contract "${A4_CONTRACT}" \
  --bcore-training-root "${BCORE_ROOT}/training" \
  --bounded-bt-5k-status \
    "${REPAIR_ROOT}/training/a4_no_bottleneck/seed_20260815/status.json" \
  --output "${STABILITY_CONTRACT}"

mkdir -p "${RUN_ROOT}/source_snapshot"
while IFS= read -r relative; do
  mkdir -p "${RUN_ROOT}/source_snapshot/$(dirname "${relative}")"
  cp "${ROOT}/${relative}" "${RUN_ROOT}/source_snapshot/${relative}"
done < <("${PYTHON}" - "${A4_CONTRACT}" <<'PY'
import json,sys
for path in json.load(open(sys.argv[1]))["source_code"]:
    print(path)
PY
)

write_status RUNNING f0 "checking bounded BT, beta=0 gradients, and fail-closed path"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_base_relative_belief.py \
  --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
  --contract "${A4_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
  --output "${RUN_ROOT}/contract/f0_receipt.json" \
  >"${RUN_ROOT}/logs/f0.log" 2>&1

write_status RUNNING training "bounded-BT beta=0 fresh seed 20260818, offline only"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_base_relative_belief \
  --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
  --contract "${A4_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
  --output "${CANDIDATE_ROOT}" --seed 20260818 \
  --variant a4_no_bottleneck --updates 25000 --workers 2 \
  --save-every 5000 --log-every 100 \
  >"${RUN_ROOT}/logs/train.log" 2>&1

write_status RUNNING analysis "applying the frozen five-point stability gates"
"${PYTHON}" scripts/before_we_act/analyze_step4r_stability.py \
  --stability-contract "${STABILITY_CONTRACT}" \
  --a4-contract "${A4_CONTRACT}" --candidate-root "${CANDIDATE_ROOT}" \
  --output "${RUN_ROOT}/stability_report.json" \
  >"${RUN_ROOT}/logs/analysis.log" 2>&1

verdict="$(jq -r '.status' "${RUN_ROOT}/stability_report.json")"
write_status COMPLETED complete "${verdict}"
trap - ERR
jq '{status,global_checks,point_checks}' "${RUN_ROOT}/stability_report.json"
