#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CARE_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CARE_GATE_RUN_ROOT:-/workspace/bwa_runs/a5r5-care-gate-first-150-per-task-20260817-v1}"
PYTHON="${BWA_CARE_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONTRACT="${BWA_CARE_GATE_CONTRACT:-${ROOT}/docs/plans/contracts/care/a4r5/20260817/a4r5_care_gate_first_collection_contract.json}"
CHECKPOINT="${BWA_CARE_CHECKPOINT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1/training/seed_20260817/deployment_checkpoint.pt}"
SCAN_MANIFEST="${RUN_ROOT}/contract/gate_scan_manifest.json"
SCAN_ROOT="${RUN_ROOT}/prebranch_scan"
BRANCH_MANIFEST="${RUN_ROOT}/contract/gate_branch_manifest.json"
FAMILY_ROOT="${RUN_ROOT}/families"
RECEIPT="${RUN_ROOT}/gate_first_collection_receipt.json"
WORKERS=4
RUN_LABEL="${BWA_CARE_RUN_LABEL:-A5R5_CARE_GATE_FIRST_COLLECTION}"

wait_for_workers() {
  local remaining=$#
  local child_code=0
  local pid
  local -a worker_pids=("$@")
  while ((remaining > 0)); do
    if wait -n; then
      child_code=0
    else
      child_code=$?
    fi
    if ((child_code != 0)); then
      for pid in "${worker_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
      done
      for pid in "${worker_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
      done
      return "${child_code}"
    fi
    remaining=$((remaining - 1))
  done
}

export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs" "${SCAN_ROOT}" "${FAMILY_ROOT}"

if [[ ! -f "${SCAN_MANIFEST}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_care_gate_first_scan.py \
    --contract "${CONTRACT}" --checkpoint "${CHECKPOINT}" \
    --output "${SCAN_MANIFEST}" \
    >"${RUN_ROOT}/logs/prepare_scan.log" 2>&1
  sha256sum "${SCAN_MANIFEST}" >"${SCAN_MANIFEST}.sha256"
fi

if [[ ! -f "${BRANCH_MANIFEST}" ]]; then
  scan_pids=()
  for shard in $(seq 0 $((WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" -u \
      scripts/before_we_act/scan_care_formal_candidates.py \
      --manifest "${SCAN_MANIFEST}" --contract "${CONTRACT}" \
      --checkpoint "${CHECKPOINT}" --output-root "${SCAN_ROOT}" \
      --robofactory-root "${ROBOFACTORY_ROOT}" --device cuda:0 \
      --shard-index "${shard}" --shard-count "${WORKERS}" \
      >>"${RUN_ROOT}/logs/scan-shard-${shard}.log" 2>&1 &
    scan_pids+=("$!")
  done
  if wait_for_workers "${scan_pids[@]}"; then
    scan_code=0
  else
    scan_code=$?
  fi
  ((scan_code == 0)) || exit "${scan_code}"
  "${PYTHON}" scripts/before_we_act/finalize_care_gate_first_manifest.py \
    --scan-manifest "${SCAN_MANIFEST}" --scan-root "${SCAN_ROOT}" \
    --contract "${CONTRACT}" --output "${BRANCH_MANIFEST}" \
    >"${RUN_ROOT}/logs/finalize_manifest.log" 2>&1
  sha256sum "${BRANCH_MANIFEST}" >"${BRANCH_MANIFEST}.sha256"
fi

collection_pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" -u -m before_we_act.care_branch_collector \
    --manifest "${BRANCH_MANIFEST}" --contract "${CONTRACT}" \
    --checkpoint "${CHECKPOINT}" --output-root "${FAMILY_ROOT}" \
    --robofactory-root "${ROBOFACTORY_ROOT}" --device cuda:0 \
    --shard-index "${shard}" --shard-count "${WORKERS}" \
    >>"${RUN_ROOT}/logs/collect-shard-${shard}.log" 2>&1 &
  collection_pids+=("$!")
done
if wait_for_workers "${collection_pids[@]}"; then
  collection_code=0
else
  collection_code=$?
fi
((collection_code == 0)) || exit "${collection_code}"

"${PYTHON}" scripts/before_we_act/summarize_care_formal_collection.py \
  --manifest "${BRANCH_MANIFEST}" --contract "${CONTRACT}" \
  --family-root "${FAMILY_ROOT}" --output "${RECEIPT}" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
sha256sum "${RECEIPT}" >"${RECEIPT}.sha256"
printf '%s_COMPLETED receipt=%s\n' "${RUN_LABEL}" "${RECEIPT}"
