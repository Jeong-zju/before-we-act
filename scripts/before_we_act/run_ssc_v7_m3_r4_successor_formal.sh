#!/usr/bin/env bash
set -euo pipefail

repository=/workspace/fe-pc-wam
runner=scripts/before_we_act/run_ssc_v7_m3_r4_successor.py
run_root=/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r4_successor_a1_v1
gate=${run_root}/frozen_gate/m3_r4_successor_a1_a2_gate.json
log_root=${run_root}/logs/formal
python_bin=${repository}/.venv/bin/python

mkdir -p "${log_root}"
cd "${repository}"

if [[ ! -f "${gate}" ]]; then
  "${python_bin}" scripts/before_we_act/freeze_ssc_v7_m3_r4_successor_formal_gate.py \
    >"${log_root}/freeze_formal_gate.log" 2>&1
fi

if [[ ! -f "${run_root}/parameter_audit.json" ]]; then
  "${python_bin}" "${runner}" parameter-audit \
    --gate "${gate}" \
    --output-root "${run_root}" \
    >"${log_root}/parameter_audit.log" 2>&1
fi

if [[ ! -f "${run_root}/formal_cache/cache_receipt.json" ]]; then
  "${python_bin}" "${runner}" build-formal-cache \
    --gate "${gate}" \
    --output-root "${run_root}/formal_cache" \
    >"${log_root}/build_formal_cache.log" 2>&1
fi

if [[ ! -f "${run_root}/formal/hc/hc_receipt.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" "${runner}" train-hc \
    --gate "${gate}" \
    --output-root "${run_root}" \
    --device cuda:0 \
    >"${log_root}/train_hc.log" 2>&1
fi

jobs=(
  oracle_arb_query:0 zero_arb_query:0 noise_arb_query:0 label_shuffled_arb_query:0
  episode_shuffled_arb_query:0 arb_direct:0 sanitized_legacy_query:0 sanitized_legacy_direct:0
  oracle_arb_query:1 zero_arb_query:1 noise_arb_query:1 label_shuffled_arb_query:1
  episode_shuffled_arb_query:1 arb_direct:1 sanitized_legacy_query:1 sanitized_legacy_direct:1
  oracle_arb_query:2 zero_arb_query:2 noise_arb_query:2 label_shuffled_arb_query:2
  episode_shuffled_arb_query:2 arb_direct:2 sanitized_legacy_query:2 sanitized_legacy_direct:2
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
      >"${log_root}/${condition}_seed_${seed_index}.log" 2>&1 &
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

if [[ ! -f "${run_root}/formal/successor_a1_a2_receipt.json" ]]; then
  "${python_bin}" "${runner}" aggregate \
    --gate "${gate}" \
    --output-root "${run_root}" \
    >"${log_root}/aggregate.log" 2>&1
fi

printf '%s\n' SSC_V7_M3_R4_SUCCESSOR_A1_A2_COMPLETE_R4_B_NOT_STARTED
