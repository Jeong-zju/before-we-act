#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CARE_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CARE_COMMON_RUN_ROOT:-/workspace/bwa_runs/a5r7-care-common-support-30-per-task-20260818-v1}"
PYTHON="${BWA_CARE_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONTRACT="${BWA_CARE_COMMON_CONTRACT:-${ROOT}/docs/plans/contracts/care/a4r7/20260818/a4r7_care_common_support_collection_contract.json}"
SOURCE_MANIFEST="${BWA_CARE_SOURCE_MANIFEST:-/workspace/bwa_runs/a5r6-care-compact-30-per-task-20260817-v1/contract/gate_branch_manifest.json}"
BRANCH_MANIFEST="${RUN_ROOT}/contract/common_support_branch_manifest.json"
FAMILY_ROOT="${RUN_ROOT}/families"
RECEIPT="${RUN_ROOT}/common_support_collection_receipt.json"
WORKERS=4

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
mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs" "${FAMILY_ROOT}"

if [[ ! -f "${BRANCH_MANIFEST}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_care_common_support_manifest.py \
    --source-manifest "${SOURCE_MANIFEST}" --contract "${CONTRACT}" \
    --output "${BRANCH_MANIFEST}" \
    >"${RUN_ROOT}/logs/prepare_manifest.log" 2>&1
  sha256sum "${BRANCH_MANIFEST}" >"${BRANCH_MANIFEST}.sha256"
fi

collection_pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" -u -m before_we_act.care_branch_collector \
    --manifest "${BRANCH_MANIFEST}" --contract "${CONTRACT}" \
    --checkpoint "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${BRANCH_MANIFEST}")" \
    --output-root "${FAMILY_ROOT}" --robofactory-root "${ROBOFACTORY_ROOT}" \
    --device cuda:0 --shard-index "${shard}" --shard-count "${WORKERS}" \
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
printf 'A5R7_CARE_COMMON_SUPPORT_COLLECTION_COMPLETED receipt=%s\n' "${RECEIPT}"
