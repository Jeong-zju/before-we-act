#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/repos/care-official
py=/workspace/venvs/mars/bin/python
run=/workspace/runs/care_official_mars_v1
raw=/workspace/datasets/mars_control/raw
rf=/workspace/repos/RoboFactory
dino=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
status=${run}/pipeline_status.json
mkdir -p "${run}/logs"
cd "${repo}"
export PYTHONPATH=${repo}:${repo}/stereo_core${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled

set_status() {
  "${py}" -m scripts.before_we_act.update_mars_care_pipeline_status \
    --output "${status}" --stage "$1" --status "$2" --detail "${3:-}"
}
heartbeat() {
  while true; do
    "${py}" -m scripts.before_we_act.update_mars_care_pipeline_status \
      --output "${status}" --heartbeat-only 2>/dev/null || true
    sleep 30
  done
}
heartbeat & heartbeat_pid=$!
cleanup() { kill "${heartbeat_pid}" 2>/dev/null || true; }
failed() { code=$?; cleanup; set_status "${current_stage:-AUTONOMOUS}" FAILED "exit=${code} line=${BASH_LINENO[0]}" || true; exit "${code}"; }
trap cleanup EXIT
trap failed ERR

current_stage=CONTRACT_AUDIT
set_status "${current_stage}" RUNNING "all-data normalization, local policy I/O, image scaling, action bounds, live four-task simulator"
"${py}" -m scripts.before_we_act.audit_mars_care_contract \
  --raw-root "${raw}" --normalization "${run}/contract/mars_norm_absolute.json" \
  --settings "${repo}/configs/before_we_act/care_mars_bench_port.json" --dino-model "${dino}" \
  --robofactory-root "${rf}" --output "${run}/contract/interface_audit.json" \
  >"${run}/logs/contract_audit.log" 2>&1
set_status "${current_stage}" PASSED "600 episodes; absolute action; RGB /255 plus pinned DINO normalization; task-specific bounds and horizons"

current_stage=SMOKE
set_status "${current_stage}" RUNNING "B0-H/B-core resume, four-task closed loop, 96 branches, CARE scorer resume, end-to-end selector"
bash scripts/before_we_act/run_mars_care_smoke.sh >"${run}/logs/smoke_pipeline.log" 2>&1
set_status "${current_stage}" PASSED "all training and closed-loop smoke receipts passed"

current_stage=FORMAL
set_status "${current_stage}" RUNNING "supervisor entering all-data CARE formal pipeline"
bash scripts/before_we_act/run_mars_care_official_pipeline.sh
