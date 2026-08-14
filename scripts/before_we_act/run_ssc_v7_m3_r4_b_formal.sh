#!/usr/bin/env bash
set -euo pipefail

repository=/workspace/fe-pc-wam
runner=scripts/before_we_act/run_ssc_v7_m3_r4_b.py
run_root=/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r4_b_observability_v1
gate=${run_root}/frozen_gate/m3_r4_b_gate.json
log_root=${run_root}/logs
python_bin=${repository}/.venv/bin/python

mkdir -p "${log_root}/predictor" "${log_root}/formal"
cd "${repository}"

if [[ ! -f "${run_root}/predictors/predictor_receipt.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" "${runner}" fit-predictors \
    --gate "${gate}" \
    --output-root "${run_root}" \
    --device cuda:0 \
    >"${log_root}/predictor/fit_predictors.log" 2>&1
fi

jobs=(
  arb_hat_direct:0 row_shuffled_direct:0 time_only_direct:0 episode_shuffled_direct:0
  stale_8_direct:0 stale_16_direct:0
  arb_hat_direct:1 row_shuffled_direct:1 time_only_direct:1 episode_shuffled_direct:1
  stale_8_direct:1 stale_16_direct:1
  arb_hat_direct:2 row_shuffled_direct:2 time_only_direct:2 episode_shuffled_direct:2
  stale_8_direct:2 stale_16_direct:2
)

index=0
while [[ "${index}" -lt "${#jobs[@]}" ]]; do
  pids=()
  for gpu in 0 1 2 3; do
    if [[ "${index}" -ge "${#jobs[@]}" ]]; then
      break
    fi
    job=${jobs[$index]}
    condition=${job%%:*}
    seed_index=${job##*:}
    receipt=${run_root}/formal/branches/${condition}/seed_${seed_index}/branch_receipt.json
    if [[ -f "${receipt}" ]]; then
      index=$((index + 1))
      continue
    fi
    CUDA_VISIBLE_DEVICES=${gpu} "${python_bin}" "${runner}" train-branch \
      --gate "${gate}" \
      --output-root "${run_root}" \
      --condition "${condition}" \
      --seed-index "${seed_index}" \
      --device cuda:0 \
      >"${log_root}/formal/${condition}_seed_${seed_index}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
done

if [[ ! -f "${run_root}/formal/r4_b_observability_receipt.json" ]]; then
  "${python_bin}" "${runner}" aggregate \
    --gate "${gate}" \
    --output-root "${run_root}" \
    >"${log_root}/formal/aggregate.log" 2>&1
fi

printf '%s\n' SSC_V7_M3_R4_B_OBSERVABILITY_COMPLETE_R4_C_NOT_STARTED
