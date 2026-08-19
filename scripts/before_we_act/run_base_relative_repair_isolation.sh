#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_A4_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_A4_REPAIR_RUN_ROOT:-/workspace/bwa_runs/a4-bt-bottleneck-repair-v1}"
BCORE_ROOT="${BWA_BCORE_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
ACTION_CACHE="${BWA_ACTION_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n2-action-context-v1}"
SCENARIO_SPLIT="${BWA_SCENARIO_SPLIT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief/contract/scenario_split.json}"
PYTHON="${BWA_A4_PYTHON:-/venv/robofactory-act/bin/python}"
CONTRACT="${RUN_ROOT}/contract/pilot_contract.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:/workspace/RoboFactory:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs"
cd "${ROOT}"

"${PYTHON}" scripts/before_we_act/prepare_base_relative_belief.py \
  --base-n2-contract "${BCORE_ROOT}/contract/n2_contract.json" \
  --source-root "${ROOT}" --phase pilot --beta-b 0.01 \
  --pilot-seed 20260815 --pilot-updates 5000 --pilot-eval-every 5000 \
  --prior-fit 0.1 --bradley-terry 0.01 --temperature 0.005 \
  --margin-fraction 0.1 --margin-cap 0.01 --output "${CONTRACT}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_base_relative_belief.py \
  --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
  --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
  --output "${RUN_ROOT}/contract/f0_receipt.json" \
  >"${RUN_ROOT}/logs/f0.log" 2>&1

train_arm() {
  local variant="$1" gpu="$2"
  local output="${RUN_ROOT}/training/${variant}/seed_20260815"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_base_relative_belief \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${output}" --seed 20260815 --variant "${variant}" \
    --updates 5000 --workers 2 --save-every 5000 --log-every 100 \
    --evaluate-at-end >"${RUN_ROOT}/logs/${variant}.log" 2>&1
}

train_arm a4_no_bottleneck 0 & control_pid=$!
train_arm a4_full 1 & full_pid=$!
wait "${control_pid}"
wait "${full_pid}"

"${PYTHON}" scripts/before_we_act/analyze_bottleneck_isolation.py \
  --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/isolation_report.json"
jq '{status,bounded_bt_checks,bottleneck_safety_checks,bottleneck_compression_checks}' \
  "${RUN_ROOT}/isolation_report.json"
