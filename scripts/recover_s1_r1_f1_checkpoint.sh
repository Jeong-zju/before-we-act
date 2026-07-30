#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S2_R3_RUN_ROOT:?set S2_R3_RUN_ROOT}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"

CONFIG="${FE_ROOT}/configs/wam_flow/s1_r1_f1_flow_cold.yaml"
RECOVERY_ROOT="${FE_ROOT}/artifacts/s1_r1_f1/recovery"
OUTPUT="${FE_ROOT}/artifacts/s1_r1_f1/checkpoint_080000.pt"
RESUME="${RECOVERY_ROOT}/resume.pt"
PROGRESS_LOG="${S2_R3_RUN_ROOT}/flow_recovery_progress.jsonl"
STAGE_LOG="${S2_R3_RUN_ROOT}/flow_recovery_stages.jsonl"
RECEIPT="${RECOVERY_ROOT}/recovery_receipt.json"

mkdir -p "${RECOVERY_ROOT}" "$(dirname "${OUTPUT}")"
if [[ -L "${OUTPUT}" && ! -e "${OUTPUT}" ]]; then
  printf 'Removing dangling promoted-checkpoint link before recovery: %s\n' \
    "${OUTPUT}"
  unlink "${OUTPUT}"
fi
if [[ -e "${OUTPUT}" ]]; then
  printf >&2 'Refusing to overwrite existing Flow checkpoint: %s\n' "${OUTPUT}"
  exit 3
fi

printf 'S1-R1 F1 checkpoint is absent; reconstructing the frozen promoted recipe.\n'
printf 'Recovery output: %s\nResume: %s\nProgress: %s\n' \
  "${OUTPUT}" "${RESUME}" "${PROGRESS_LOG}"
(
  cd "${FE_ROOT}"
  CUDA_VISIBLE_DEVICES=0 \
  LPD_STAGE_LOG="${STAGE_LOG}" \
  PYTHONUNBUFFERED=1 \
    uv run --frozen python scripts/train_agent_factorized_flow_wam.py \
      --config "${CONFIG}" \
      --device cuda:0 \
      --output "${OUTPUT}" \
      --resume "${RESUME}" \
      --progress-log "${PROGRESS_LOG}"
)

(
  cd "${FE_ROOT}"
  uv run --frozen python scripts/verify_s1_r1_f1_checkpoint.py \
    "${OUTPUT}" \
    --config "${CONFIG}" \
    >"${RECEIPT}"
)
test -s "${RECEIPT}"
printf 'Recovered and verified S1-R1 F1 checkpoint: %s\n' "${OUTPUT}"
