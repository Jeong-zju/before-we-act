#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in S3_R6_RUN_ROOT S3_R6_HF_TOKEN_FIFO UV_CACHE_DIR \
  UV_PROJECT_ENVIRONMENT ROBOFACTORY_ROOT RF_PYTHON S3_R6_P0_CONFIG; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'Missing %s\n' "${name}"; exit 2; fi
done
if [[ ! -p "${S3_R6_HF_TOKEN_FIFO}" ]]; then
  printf >&2 'Missing protected Hugging Face token FIFO: %s\n' \
    "${S3_R6_HF_TOKEN_FIFO}"
  exit 3
fi

HF_TOKEN_INPUT=""
SECRET_DIR=""
SECRET_FIFO=""
WRITER_PID=""
cleanup_secret() {
  if [[ -n "${WRITER_PID}" ]]; then
    kill "${WRITER_PID}" 2>/dev/null || true
    wait "${WRITER_PID}" 2>/dev/null || true
  fi
  [[ -n "${SECRET_FIFO}" ]] && unlink "${SECRET_FIFO}" 2>/dev/null || true
  [[ -n "${SECRET_DIR}" ]] && rmdir "${SECRET_DIR}" 2>/dev/null || true
  unlink "${S3_R6_HF_TOKEN_FIFO}" 2>/dev/null || true
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
}
trap cleanup_secret EXIT INT TERM

IFS= read -r HF_TOKEN_INPUT <"${S3_R6_HF_TOKEN_FIFO}"
unlink "${S3_R6_HF_TOKEN_FIFO}" 2>/dev/null || true
if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
  printf >&2 'The protected Hugging Face token input was invalid.\n'
  exit 3
fi
SECRET_DIR="$(mktemp -d "${S3_R6_RUN_ROOT}/.s3-r6-hf.XXXXXX")" || exit $?
chmod 700 "${SECRET_DIR}" || exit $?
SECRET_FIFO="${SECRET_DIR}/token.fifo"

deliver_to() {
  consumer="$1"
  mkfifo "${SECRET_FIFO}" || return $?
  chmod 600 "${SECRET_FIFO}" || return $?
  printf '%s\n' "${HF_TOKEN_INPUT}" >"${SECRET_FIFO}" &
  WRITER_PID=$!
  "${consumer}" || return $?
  wait "${WRITER_PID}" || return $?
  WRITER_PID=""
  unlink "${SECRET_FIFO}" 2>/dev/null || true
}

run_s0_prepare() {
  S0_RUN_ROOT="${S3_R6_RUN_ROOT}/bootstrap/s0" \
  S0_READY_FILE="${S3_R6_RUN_ROOT}/bootstrap/s0.ready" \
  S0_FAILED_FILE="${S3_R6_RUN_ROOT}/bootstrap/s0.failed" \
  S0_HF_TOKEN_FIFO="${SECRET_FIFO}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
  ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}" \
  RF_PYTHON="${RF_PYTHON}" \
    bash "${FE_ROOT}/scripts/prepare_s0_shared.sh"
}

run_s2_prepare() {
  S2_R4_RUN_ROOT="${S3_R6_RUN_ROOT}/bootstrap/s2_r4" \
  S2_R4_READY_FILE="${S3_R6_RUN_ROOT}/bootstrap/s2_r4.ready" \
  S2_R4_FAILED_FILE="${S3_R6_RUN_ROOT}/bootstrap/s2_r4.failed" \
  S2_R4_HF_TOKEN_FIFO="${SECRET_FIFO}" \
  S2_R4_P0_CONFIG="${S3_R6_P0_CONFIG}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
    bash "${FE_ROOT}/scripts/prepare_s2_r4_shared.sh"
}

# Keep S0's tested download/install contract, then extend it with the later
# five-task/PCA preparation. The token only crosses mode-0600 FIFOs.
deliver_to run_s0_prepare || exit $?
NEEDS_S2_PREP=0
for required in \
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/take_photo/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/three_robots_stack_cube/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/camera_alignment/training_manifest.json" \
  "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt"; do
  [[ -f "${required}" ]] || NEEDS_S2_PREP=1
done
if (( NEEDS_S2_PREP )); then
  deliver_to run_s2_prepare || exit $?
else
  printf 'Reusing complete five-task data and S2-R4 PCA artifact.\n'
fi
printf 'S3-R6 S0-compatible environment and five-task assets are ready.\n'
