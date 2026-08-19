#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CARE_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CARE_RUN_ROOT:-/workspace/bwa_runs/a5r2-care-branches-pilot-20260817-v3}"
PYTHON="${BWA_CARE_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONTRACT="${BWA_CARE_CONTRACT:-${ROOT}/docs/plans/contracts/care/a4r2/20260817/a4r2_care_contract.json}"
CHECKPOINT="${BWA_CARE_CHECKPOINT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1/training/seed_20260817/deployment_checkpoint.pt}"
MANIFEST="${RUN_ROOT}/contract/pilot_manifest.json"
FAMILY_ROOT="${RUN_ROOT}/families"
RECEIPT="${RUN_ROOT}/pilot_receipt.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs" "${FAMILY_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_care_branch_pilot.py \
    --contract "${CONTRACT}" --checkpoint "${CHECKPOINT}" \
    --families-per-task 2 --output "${MANIFEST}" \
    >"${RUN_ROOT}/logs/prepare.log" 2>&1
fi

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
PIDS=()
for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  gpu="$((index % 4))"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.care_branch_collector \
    --manifest "${MANIFEST}" --contract "${CONTRACT}" --checkpoint "${CHECKPOINT}" \
    --task "${task}" --output-root "${FAMILY_ROOT}" \
    --robofactory-root "${ROBOFACTORY_ROOT}" --device cuda:0 \
    >"${RUN_ROOT}/logs/${task}.log" 2>&1 &
  PIDS+=("$!")
done

code=0
for pid in "${PIDS[@]}"; do
  set +e
  wait "${pid}"
  child=$?
  set -e
  ((child == 0)) || code="${child}"
done
"${PYTHON}" scripts/before_we_act/summarize_care_branch_pilot.py \
  --manifest "${MANIFEST}" --family-root "${FAMILY_ROOT}" --output "${RECEIPT}" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
sha256sum "${MANIFEST}" >"${MANIFEST}.sha256"
sha256sum "${RECEIPT}" >"${RECEIPT}.sha256"
printf 'A5R2_CARE_RESOURCE_PILOT_COMPLETED receipt=%s worker_code=%s\n' "${RECEIPT}" "${code}"
((code == 0)) || exit "${code}"
