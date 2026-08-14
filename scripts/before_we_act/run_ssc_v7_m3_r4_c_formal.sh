#!/usr/bin/env bash
set -euo pipefail

repository=/workspace/fe-pc-wam
root=/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r4_c_sealed_test_v1
gate="$root/frozen_gate/m3_r4_c_gate.json"
python_bin="$repository/.venv/bin/python"
runner="$repository/scripts/before_we_act/run_ssc_v7_m3_r4_c.py"
collection_root="$root/test_collections"
sealed_root="$root/sealed_test"
log_root="$root/logs"
mkdir -p "$collection_root" "$log_root"

tasks=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
for batch_start in 0 4; do
  pids=()
  for slot in 0 1 2 3; do
    index=$((batch_start + slot))
    if (( index >= ${#tasks[@]} )); then
      continue
    fi
    task=${tasks[$index]}
    task_root="$collection_root/$task"
    receipt="$task_root/task_collection_receipt.json"
    if [[ -f "$receipt" ]]; then
      continue
    fi
    if [[ -e "$task_root" ]]; then
      echo "partial R4-C task output cannot be retried: $task_root" >&2
      exit 1
    fi
    (
      cd "$repository"
      "$python_bin" "$runner" collect-test-task \
        --gate "$gate" \
        --output-root "$task_root" \
        --task "$task" \
        --device cpu
    ) >"$log_root/collect_${task}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed != 0 )); then
    exit 1
  fi
done

if [[ ! -f "$sealed_root/test_manifest_receipt.json" ]]; then
  cd "$repository"
  "$python_bin" "$runner" merge-test \
    --gate "$gate" \
    --data-root "$collection_root" \
    --output-root "$sealed_root" \
    --device cpu \
    >"$log_root/merge_test.log" 2>&1
fi

if [[ ! -f "$root/formal/r4_c_sealed_test_receipt.json" ]]; then
  if [[ -f "$root/formal/test_opened_marker.json" ]]; then
    echo "R4-C test was opened and retry is forbidden" >&2
    exit 1
  fi
  cd "$repository"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" "$runner" evaluate-once \
    --gate "$gate" \
    --output-root "$root" \
    --device cuda:0 \
    >"$log_root/evaluate_once.log" 2>&1
fi

echo SSC_V7_M3_R4_C_FORMAL_COMPLETE
