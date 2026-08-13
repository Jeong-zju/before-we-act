#!/usr/bin/env bash
set -euo pipefail

repository=/workspace/fe-pc-wam
runner=scripts/before_we_act/run_ssc_v7_m3_r4_successor.py
gate=docs/experiments/ssc_v7/m3_r4_successor_preparation_gate.json
run_root=/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r4_successor_a1_v1
collection_root=${run_root}/confirmation_collections
manifest_root=${run_root}/confirmation_data
log_root=${run_root}/logs/preparation
python_bin=${repository}/.venv/bin/python
export PYTHONPATH=/workspace/RoboFactory:${PYTHONPATH:-}

mkdir -p "${collection_root}" "${log_root}"
cd "${repository}"

tasks=(
  lift_barrier
  camera_alignment
  long_pipeline_delivery
  take_photo
  pass_shoe
  place_food
)

pids=()
for task in "${tasks[@]}"; do
  receipt=${collection_root}/${task}/task_collection_receipt.json
  if [[ -f "${receipt}" ]]; then
    continue
  fi
  "${python_bin}" "${runner}" collect-confirmation-task \
    --gate "${gate}" \
    --task "${task}" \
    --output-root "${collection_root}/${task}" \
    >"${log_root}/collect_${task}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

if [[ ! -f "${manifest_root}/confirmation_manifest_receipt.json" ]]; then
  "${python_bin}" "${runner}" merge-confirmation \
    --gate "${gate}" \
    --data-root "${collection_root}" \
    --output-root "${manifest_root}" \
    >"${log_root}/merge_confirmation.log" 2>&1
fi

printf '%s\n' SSC_V7_M3_R4_SUCCESSOR_PREPARATION_COMPLETE
