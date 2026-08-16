#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BWA_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_ACTION_GROUNDED_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief}"
TEAM_SIGNAL_ROOT="${BWA_TEAM_SIGNAL_RUN_ROOT:-/workspace/bwa_runs/p1-b-core-n1-v1}"
TEAM_SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
TEMPORAL_ROOT="${BWA_TEMPORAL_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
PYTHON="${BWA_PYTHON:-/venv/robofactory-act/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor/stereo-core:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

if [[ ! -f "${RUN_ROOT}/contract/r1_0_receipt.json" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_action_grounded_belief.py \
    --signal-cache "${TEAM_SIGNAL_CACHE}" --signal-run "${TEAM_SIGNAL_ROOT}" \
    --temporal-run "${TEMPORAL_ROOT}" --output "${RUN_ROOT}" \
    >"${RUN_ROOT}/logs/r1_0_prepare.log" 2>&1
fi

if [[ ! -f "${RUN_ROOT}/contract/f0_f1_receipt.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_action_grounded_belief.py \
    --cache "${TEAM_SIGNAL_CACHE}" --contract "${RUN_ROOT}/contract/r1_contract.json" \
    --scenario-split "${RUN_ROOT}/contract/scenario_split.json" \
    --output "${RUN_ROOT}/contract/f0_f1_receipt.json" \
    >"${RUN_ROOT}/logs/r1_0_f0_f1.log" 2>&1
fi

seeds=(20260815 20260816 20260817)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  output="${RUN_ROOT}/r1_1_fair_probe/seed_${seed}"
  status="$(${PYTHON} -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "") if p.is_file() else "")' "${output}/status.json")"
  if [[ "${status}" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT|INCONCLUSIVE_TRAINING_NOT_CONVERGED)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_action_grounded_probe \
    --cache "${TEAM_SIGNAL_CACHE}" --contract "${RUN_ROOT}/contract/r1_contract.json" \
    --scenario-split "${RUN_ROOT}/contract/scenario_split.json" \
    --output "${output}" --seed "${seed}" \
    >"${RUN_ROOT}/logs/r1_1_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]:-}"; do
  [[ -n "${pid}" ]] && wait "${pid}"
done

CUDA_VISIBLE_DEVICES=3 "${PYTHON}" scripts/before_we_act/analyze_action_grounded_probe.py \
  --cache "${TEAM_SIGNAL_CACHE}" --contract "${RUN_ROOT}/contract/r1_contract.json" \
  --scenario-split "${RUN_ROOT}/contract/scenario_split.json" \
  --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/r1_1_fair_probe/conclusion.json" \
  >"${RUN_ROOT}/logs/r1_1_analyze.log" 2>&1
