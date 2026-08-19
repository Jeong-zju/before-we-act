#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_A4_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_A4_RUN_ROOT:-/workspace/bwa_runs/a4-base-relative-v1}"
BCORE_ROOT="${BWA_BCORE_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
ACTION_CACHE="${BWA_ACTION_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n2-action-context-v1}"
SCENARIO_SPLIT="${BWA_SCENARIO_SPLIT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief/contract/scenario_split.json}"
PYTHON="${BWA_A4_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
PILOT_STATUS="${RUN_ROOT}/pilot/seed_20260815/status.json"
FORMAL_CONTRACT="${RUN_ROOT}/contract/formal_contract.json"
STATUS="${RUN_ROOT}/training_pipeline_status.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs"
cd "${ROOT}"

write_status() {
  local status="$1" stage="$2" detail="$3"
  "${PYTHON}" - "${STATUS}" "${status}" "${stage}" "${detail}" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,p)
PY
}

on_error() {
  local code=$?
  write_status FAILED error "training pipeline exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'write_status STOPPED interrupted signal; exit 130' INT TERM

write_status WAITING loss_scale_pilot "waiting for the frozen 25k no-closed-loop pilot"
while true; do
  pilot="$(jq -r '.status // empty' "${PILOT_STATUS}" 2>/dev/null || true)"
  [[ "${pilot}" == PASSED_LOSS_SCALE_PILOT ]] && break
  [[ "${pilot}" =~ ^(FAILED|STOPPED) ]] && {
    echo "loss-scale pilot failed: ${pilot}" >&2
    exit 2
  }
  sleep 60
done

if [[ ! -f "${FORMAL_CONTRACT}" ]]; then
  write_status RUNNING formal_contract "freezing beta_b after the passed pilot"
  "${PYTHON}" scripts/before_we_act/prepare_base_relative_belief.py \
    --base-n2-contract "${BCORE_ROOT}/contract/n2_contract.json" \
    --source-root "${ROOT}" --phase formal \
    --beta-b 0.01 --prior-fit 0.1 --bradley-terry 0.01 --temperature 0.005 \
    --pilot-status "${PILOT_STATUS}" --output "${FORMAL_CONTRACT}" \
    >"${RUN_ROOT}/logs/formal_contract.log" 2>&1
fi

if [[ ! -f "${RUN_ROOT}/contract/f0_formal_receipt.json" ]]; then
  write_status RUNNING f0 "auditing restricted prior, split KL, BT and zero-init"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_base_relative_belief.py \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${FORMAL_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${RUN_ROOT}/contract/f0_formal_receipt.json" \
    >"${RUN_ROOT}/logs/f0_formal.log" 2>&1
fi

F1_REFERENCE="${RUN_ROOT}/f1/reference"
F1_RESUMED="${RUN_ROOT}/f1/resumed"
if [[ ! -f "${RUN_ROOT}/contract/f1_formal_receipt.json" ]]; then
  write_status RUNNING f1 "checking exact 4-update fresh versus 2+2 resume"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_base_relative_belief \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${FORMAL_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${F1_REFERENCE}" --seed 20260815 --variant a4_full \
    --updates 4 --workers 0 --save-every 2 --resume-audit \
    >"${RUN_ROOT}/logs/f1_reference.log" 2>&1
  mkdir -p "${F1_RESUMED}"
  cp "${F1_REFERENCE}/checkpoint_000002.pt" "${F1_RESUMED}/checkpoint_latest.pt"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_base_relative_belief \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${FORMAL_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${F1_RESUMED}" --seed 20260815 --variant a4_full \
    --updates 4 --workers 0 --save-every 2 --resume-audit \
    >"${RUN_ROOT}/logs/f1_resumed.log" 2>&1
  "${PYTHON}" scripts/before_we_act/verify_predictive_team_belief.py f1 \
    --reference "${F1_REFERENCE}/checkpoint_000004.pt" \
    --resumed "${F1_RESUMED}/checkpoint_000004.pt" \
    --output "${RUN_ROOT}/contract/f1_formal_receipt.json" \
    >"${RUN_ROOT}/logs/f1_verify.log" 2>&1
fi

train_seed() {
  local variant="$1" seed="$2" gpu="$3"
  local seed_root="${RUN_ROOT}/training/${variant}/seed_${seed}"
  local terminal
  terminal="$(jq -r '.status // empty' "${seed_root}/status.json" 2>/dev/null || true)"
  if [[ "${terminal}" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT|INCONCLUSIVE_TRAINING_NOT_CONVERGED)$ ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_base_relative_belief \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${FORMAL_CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${seed_root}" --seed "${seed}" --variant "${variant}" \
    --updates 120000 --workers 2 --save-every 5000 --log-every 100 \
    >"${RUN_ROOT}/logs/train_${variant}_${seed}.log" 2>&1
}

SEEDS=(20260815 20260816 20260817)
write_status RUNNING training "three matched full/no-bottleneck seed pairs"
PIDS=()
for gpu in 0 1 2; do
  seed="${SEEDS[$gpu]}"
  train_seed a4_full "${seed}" "${gpu}" & PIDS+=("$!")
  train_seed a4_no_bottleneck "${seed}" "${gpu}" & PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

write_status RUNNING offline_analysis "issuing hard-offline and action-relevance verdict"
"${PYTHON}" scripts/before_we_act/analyze_base_relative_belief.py \
  --contract "${FORMAL_CONTRACT}" --run-root "${RUN_ROOT}" \
  --bcore-training-root "${BCORE_ROOT}/training" \
  --output "${RUN_ROOT}/offline_acceptance.json" \
  >"${RUN_ROOT}/logs/offline_analysis.log" 2>&1
final="$(jq -r '.status' "${RUN_ROOT}/offline_acceptance.json")"
write_status PASSED complete "${final}"
trap - ERR
printf 'A4_TRAINING_COMPLETED status=%s\n' "${final}"
