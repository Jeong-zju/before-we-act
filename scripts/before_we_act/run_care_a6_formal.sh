#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CARE_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CARE_A6_RUN_ROOT:-/workspace/bwa_runs/a6r1-care-owner-authorized-diagnostic-20260818-v1}"
PYTHON="${BWA_CARE_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
SETTINGS="${BWA_CARE_SETTINGS:-${ROOT}/configs/before_we_act/care_robofactory_reproduction.json}"
CONTRACT="${BWA_CARE_A6_CONTRACT:-${ROOT}/docs/plans/contracts/care/a6r1/20260818/a6r1_care_owner_authorized_diagnostic_contract.json}"
SOURCE_ROOT="/workspace/bwa_runs/a5r7-care-common-support-30-per-task-20260818-v1"
FAMILY_ROOT="${SOURCE_ROOT}/families"
QUALITY_ROOT="${SOURCE_ROOT}/quality/a5r7q1-simulator-labels"
REFERENCE="/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1/training/seed_20260817/deployment_checkpoint.pt"
PREPARED="${RUN_ROOT}/data/care_prepared.pt"
PREPARED_MANIFEST="${RUN_ROOT}/data/care_prepared_manifest.json"
TRAIN_ROOT="${RUN_ROOT}/training"
OFFLINE_ROOT="${RUN_ROOT}/offline"
CARE_CHECKPOINT="${OFFLINE_ROOT}/care_deployment_checkpoint.pt"
SEED_ROOT="${RUN_ROOT}/validation20/seeds"
VALIDATION_ROOT="${RUN_ROOT}/validation20"
SUMMARY="${VALIDATION_ROOT}/summary.json"
OLD_VALIDATION_ROOT="/workspace/bwa_runs/w10-six-task-v1/seeds/validation"
CONFIRMATION_ROOT="/workspace/bwa_runs/b-core/n3-minimal-attribution-v1/confirmation50/seeds"

export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/data" "${TRAIN_ROOT}" "${OFFLINE_ROOT}" "${VALIDATION_ROOT}"
"${PYTHON}" scripts/before_we_act/verify_frozen_settings.py \
  --repo-root "${ROOT}" --settings "${SETTINGS}" \
  --robofactory-root "${ROBOFACTORY_ROOT}" \
  >"${RUN_ROOT}/frozen_settings_receipt.txt"

mapfile -t tasks < <(jq -r '.tasks | keys[]' "${SETTINGS}")
mapfile -t variants < <(jq -r '.care.variants[]' "${SETTINGS}")
mapfile -t seeds < <(jq -r '.care.seeds[]' "${SETTINGS}")
CARE_UPDATES="$(jq -r '.care.updates' "${SETTINGS}")"
CARE_BATCH_SIZE="$(jq -r '.care.batch_size' "${SETTINGS}")"
CARE_LEARNING_RATE="$(jq -r '.care.learning_rate' "${SETTINGS}")"
CARE_WEIGHT_DECAY="$(jq -r '.care.weight_decay' "${SETTINGS}")"
CARE_EVAL_EVERY="$(jq -r '.care.evaluation_every_updates' "${SETTINGS}")"
CARE_EPISODES="$(jq -r '.closed_loop.episodes_per_task' "${SETTINGS}")"

wait_for_workers() {
  local code=0 child_code pid
  local -a pids=("$@")
  for pid in "${pids[@]}"; do
    if wait "${pid}"; then child_code=0; else child_code=$?; fi
    ((child_code == 0)) || code="${child_code}"
  done
  ((code == 0)) || return "${code}"
}

if [[ ! -f "${PREPARED}" ]]; then
  "${PYTHON}" -u scripts/before_we_act/prepare_care_training.py \
    --contract "${CONTRACT}" --family-root "${FAMILY_ROOT}" \
    --quality-root "${QUALITY_ROOT}" --reference-checkpoint "${REFERENCE}" \
    --output "${PREPARED}" --manifest "${PREPARED_MANIFEST}" --device cuda:0 \
    >"${RUN_ROOT}/logs/prepare_training.log" 2>&1
  sha256sum "${PREPARED}" >"${PREPARED}.sha256"
  sha256sum "${PREPARED_MANIFEST}" >"${PREPARED_MANIFEST}.sha256"
fi

training_pids=()
job=0
for variant in "${variants[@]}"; do
  for seed in "${seeds[@]}"; do
    output="${TRAIN_ROOT}/${variant}/seed_${seed}"
    status="${output}/status.json"
    if [[ -f "${status}" ]] && [[ "$(jq -r '.status' "${status}")" == COMPLETED ]]; then
      continue
    fi
    mkdir -p "${output}"
    gpu=$((job % 4))
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.train_care_belief \
      --prepared-data "${PREPARED}" --output "${output}" \
      --seed "${seed}" --variant "${variant}" --updates "${CARE_UPDATES}" \
      --batch-size "${CARE_BATCH_SIZE}" --eval-every "${CARE_EVAL_EVERY}" \
      --learning-rate "${CARE_LEARNING_RATE}" --weight-decay "${CARE_WEIGHT_DECAY}" \
      --device cuda:0 \
      >"${RUN_ROOT}/logs/train_${variant}_${seed}.log" 2>&1 &
    training_pids+=("$!")
    job=$((job + 1))
  done
done
if ((${#training_pids[@]})); then
  wait_for_workers "${training_pids[@]}"
fi

if [[ ! -f "${OFFLINE_ROOT}/offline_report.json" || ! -f "${CARE_CHECKPOINT}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/before_we_act/select_calibrate_care.py \
    --contract "${CONTRACT}" --prepared-data "${PREPARED}" \
    --training-root "${TRAIN_ROOT}" --reference-checkpoint "${REFERENCE}" \
    --output-root "${OFFLINE_ROOT}" --device cuda:0 \
    >"${RUN_ROOT}/logs/select_calibrate.log" 2>&1
fi

"${PYTHON}" scripts/before_we_act/prepare_care_validation20_seeds.py \
  --old-validation-root "${OLD_VALIDATION_ROOT}" \
  --confirmation-root "${CONFIRMATION_ROOT}" --output-root "${SEED_ROOT}" \
  >"${RUN_ROOT}/logs/prepare_validation20_seeds.log" 2>&1

run_task_pair() {
  local task="$1" gpu="$2" mode output log
  local max_steps
  max_steps="$(jq -r --arg task "${task}" '.tasks[$task].max_steps' "${SETTINGS}")"
  for mode in selector_off care; do
    output="${VALIDATION_ROOT}/${mode}/${task}.json"
    log="${RUN_ROOT}/logs/validation20_${mode}_${task}.log"
    if [[ -f "${output}" ]] && [[ "$(jq -r '.episodes' "${output}")" == 20 ]]; then
      continue
    fi
    mkdir -p "$(dirname "${output}")"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_care_closed_loop \
      --reference-checkpoint "${REFERENCE}" --care-checkpoint "${CARE_CHECKPOINT}" \
      --mode "${mode}" --task "${task}" --seed-file "${SEED_ROOT}/${task}.json" \
      --episodes "${CARE_EPISODES}" --max-steps "${max_steps}" --device cuda:0 \
      --robofactory-root "${ROBOFACTORY_ROOT}" --resume-log "${log}" \
      --output "${output}" >>"${log}" 2>&1
  done
}

run_wave() {
  local item task gpu
  local -a pids=()
  for item in "$@"; do
    task="${item%%:*}"
    gpu="${item##*:}"
    run_task_pair "${task}" "${gpu}" &
    pids+=("$!")
  done
  wait_for_workers "${pids[@]}"
}

run_wave camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave pass_shoe:0 place_food:1

if [[ ! -f "${SUMMARY}" ]]; then
  "${PYTHON}" scripts/before_we_act/summarize_care_validation20.py \
    --contract "${CONTRACT}" --offline-report "${OFFLINE_ROOT}/offline_report.json" \
    --validation-root "${VALIDATION_ROOT}" --seed-receipt "${SEED_ROOT}/receipt.json" \
    --output "${SUMMARY}" >"${RUN_ROOT}/logs/summarize_validation20.log" 2>&1
  sha256sum "${SUMMARY}" >"${SUMMARY}.sha256"
fi
printf 'A6R1_A7R1_CARE_OWNER_AUTHORIZED_DIAGNOSTIC_COMPLETED summary=%s\n' "${SUMMARY}"
