#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BWA_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_TEAM_SIGNAL_RUN_ROOT:-/workspace/bwa_runs/p1-b-core-n1-v1}"
CACHE_ROOT="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
TEMPORAL_ROOT="${BWA_TEMPORAL_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
PYTHON="${BWA_PYTHON:-/venv/robofactory-act/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor/stereo-core:${PYTHONPATH:-}"

manifests=(
  /workspace/datasets/robofactory_multitask/lift_barrier/training_manifest.json
  /workspace/datasets/robofactory_multitask/camera_alignment/training_manifest.json
  /workspace/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json
  /workspace/datasets/robofactory_multitask/take_photo/training_manifest.json
  /workspace/datasets/robofactory_multitask/pass_shoe/training_manifest.json
  /workspace/datasets/robofactory_multitask/place_food/training_manifest.json
)
mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

if [[ ! -f "${RUN_ROOT}/contract/cache_receipt.json" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_raw_team_signal.py \
    --manifests "${manifests[@]}" \
    --temporal-contract "${TEMPORAL_ROOT}/contract/step2_contract.json" \
    --normalization "${TEMPORAL_ROOT}/contract/normalization.pt" \
    --visual-cache /workspace/bwa_runs/shared/p1-step2-dino-history-cache-v2 \
    --cache-output "${CACHE_ROOT}" --run-root "${RUN_ROOT}" \
    >"${RUN_ROOT}/logs/prepare.log" 2>&1
fi

if [[ ! -f "${RUN_ROOT}/contract/f0_f1_receipt.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_raw_team_signal.py \
    --cache "${CACHE_ROOT}" --contract "${RUN_ROOT}/contract/n1_contract.json" \
    --output "${RUN_ROOT}/contract/f0_f1_receipt.json" \
    >"${RUN_ROOT}/logs/f0_f1.log" 2>&1
fi

seeds=(20260815 20260816 20260817)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  output="${RUN_ROOT}/representation/seed_${seed}"
  if [[ "$("${PYTHON}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "") if p.is_file() else "")' "${output}/status.json")" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_raw_team_signal \
    --cache "${CACHE_ROOT}" --contract "${RUN_ROOT}/contract/n1_contract.json" \
    --output "${output}" --seed "${seed}" --data-seed 20260815 \
    >"${RUN_ROOT}/logs/representation_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]:-}"; do [[ -n "${pid}" ]] && wait "${pid}"; done

pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  rep_root="${RUN_ROOT}/representation/seed_${seed}"
  selected="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_update"])' "${rep_root}/status.json")"
  representation="${rep_root}/checkpoint_$(printf '%06d' "${selected}").pt"
  output="${RUN_ROOT}/probe/seed_${seed}"
  if [[ "$("${PYTHON}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "") if p.is_file() else "")' "${output}/status.json")" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_team_action_probe \
    --cache "${CACHE_ROOT}" --contract "${RUN_ROOT}/contract/n1_contract.json" \
    --representation "${representation}" --output "${output}" --seed "${seed}" --data-seed 20260815 \
    >"${RUN_ROOT}/logs/probe_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]:-}"; do [[ -n "${pid}" ]] && wait "${pid}"; done

"${PYTHON}" scripts/before_we_act/analyze_raw_team_signal.py \
  --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/n1_conclusion.json" \
  >"${RUN_ROOT}/logs/analyze.log" 2>&1
