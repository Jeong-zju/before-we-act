#!/usr/bin/env bash
set -euo pipefail

repository=/workspace/fe-pc-wam
root=/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r4_b_supplement_v1
gate="$root/frozen_gate/m3_r4_b_supplement_gate.json"
script="$repository/scripts/before_we_act/run_ssc_v7_m3_r4_b_supplement.py"
python_bin="$repository/.venv/bin/python"
log_root="$root/logs"
mkdir -p "$log_root"

conditions=(hc_hidden_only_direct phase_matched_row_shuffle_direct)
jobs=()
for condition in "${conditions[@]}"; do
  for seed_index in 0 1 2; do
    jobs+=("$condition:$seed_index")
  done
done

for batch_start in 0 4; do
  pids=()
  for gpu in 0 1 2 3; do
    job_index=$((batch_start + gpu))
    if (( job_index >= ${#jobs[@]} )); then
      continue
    fi
    IFS=: read -r condition seed_index <<<"${jobs[$job_index]}"
    log="$log_root/${condition}_seed_${seed_index}.log"
    (
      cd "$repository"
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
        "$python_bin" "$script" train-branch \
          --gate "$gate" \
          --output-root "$root" \
          --condition "$condition" \
          --seed-index "$seed_index" \
          --device cuda:0
    ) >"$log" 2>&1 &
    pids+=("$!")
  done
  batch_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      batch_failed=1
    fi
  done
  if (( batch_failed != 0 )); then
    exit 1
  fi
done

cd "$repository"
"$python_bin" "$script" aggregate --gate "$gate" --output-root "$root" --device cpu
