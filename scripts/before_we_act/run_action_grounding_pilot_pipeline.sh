#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BWA_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_ACTION_GROUNDED_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief}"
PYTHON="${BWA_PYTHON:-/venv/robofactory-act/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor/stereo-core:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

if [[ ! -f "${RUN_ROOT}/contract/r1_3_pilot_contract.json" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_action_grounding_pilot.py \
    --dataset-root /workspace/datasets/robofactory_multitask \
    --action-grounded-contract "${RUN_ROOT}/contract/r1_contract.json" \
    --collector "${REPO_ROOT}/scripts/before_we_act/run_action_grounding_pilot.py" \
    --output "${RUN_ROOT}/contract/r1_3_pilot_contract.json" \
    >"${RUN_ROOT}/logs/r1_3_prepare.log" 2>&1
fi
if [[ ! -f "${RUN_ROOT}/r1_3_counterfactual_pilot/conclusion.json" ]]; then
  "${PYTHON}" scripts/before_we_act/run_action_grounding_pilot.py \
    --contract "${RUN_ROOT}/contract/r1_3_pilot_contract.json" \
    --robofactory-root /workspace/RoboFactory \
    --output-root "${RUN_ROOT}/r1_3_counterfactual_pilot" \
    >"${RUN_ROOT}/logs/r1_3_pilot.log" 2>&1
fi
