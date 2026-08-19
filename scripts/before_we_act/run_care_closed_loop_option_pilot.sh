#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CARE_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CARE_OPTION_RUN_ROOT:-/workspace/bwa_runs/a5r9-care-common-snapshot-option-pilot-20260818-v1}"
SOURCE_SELECTION_MANIFEST="${BWA_CARE_SOURCE_SELECTION_MANIFEST:-/workspace/bwa_runs/a5r8-care-closed-loop-option-pilot-20260818-v1/contract/closed_loop_option_pilot_manifest.json}"
PYTHON="${BWA_CARE_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONTRACT="${BWA_CARE_OPTION_CONTRACT:-${ROOT}/docs/plans/contracts/care/a4r10/20260818/a4r10_care_common_snapshot_option_pilot_contract.json}"
SEPARATED_CONTRACT="${ROOT}/docs/plans/contracts/care/a4r8/20260818/a4r8_care_separated_gate_a_contract.json"
MANIFEST="${RUN_ROOT}/contract/common_snapshot_option_pilot_manifest.json"
FAMILY_ROOT="${RUN_ROOT}/families"
REPORT="${RUN_ROOT}/evaluation/closed_loop_option_pilot_report.json"
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
mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs" "${FAMILY_ROOT}" "${RUN_ROOT}/evaluation"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_care_common_snapshot_option_pilot.py \
    --source-selection-manifest "${SOURCE_SELECTION_MANIFEST}" \
    --contract "${CONTRACT}" \
    --output "${MANIFEST}" \
    >"${RUN_ROOT}/logs/prepare_manifest.log" 2>&1
  sha256sum "${MANIFEST}" >"${MANIFEST}.sha256"
fi

CHECKPOINT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${MANIFEST}")"
collection_pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" -u scripts/before_we_act/collect_care_closed_loop_option_pilot.py \
    --manifest "${MANIFEST}" \
    --contract "${CONTRACT}" \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${FAMILY_ROOT}" \
    --robofactory-root "${ROBOFACTORY_ROOT}" \
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

if [[ -f "${REPORT}" ]]; then
  printf 'Refusing to overwrite completed A5R9 pilot report: %s\n' "${REPORT}" >&2
  exit 1
fi
"${PYTHON}" scripts/before_we_act/evaluate_care_closed_loop_option_pilot.py \
  --manifest "${MANIFEST}" \
  --contract "${CONTRACT}" \
  --separated-contract "${SEPARATED_CONTRACT}" \
  --family-root "${FAMILY_ROOT}" \
  --output "${REPORT}" \
  >"${RUN_ROOT}/logs/evaluate.log" 2>&1
sha256sum "${REPORT}" >"${REPORT}.sha256"
printf 'A5R9_CARE_COMMON_SNAPSHOT_OPTION_PILOT_COMPLETED report=%s\n' "${REPORT}"
