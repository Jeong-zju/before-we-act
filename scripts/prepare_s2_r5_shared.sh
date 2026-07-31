#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S2_R5_RUN_ROOT:?set S2_R5_RUN_ROOT}"
: "${S2_R5_READY_FILE:?set S2_R5_READY_FILE}"
: "${S2_R5_FAILED_FILE:?set S2_R5_FAILED_FILE}"
: "${S2_R5_PROTECTED_P0_SOURCE:?set S2_R5_PROTECTED_P0_SOURCE}"
: "${S2_R5_P0_CONFIG:?set S2_R5_P0_CONFIG}"
STATUS_TOOL="${FE_ROOT}/scripts/s2_r5_runtime.py"
HEARTBEAT_PID=""
COMPLETED=0

status() {
  python3 "${STATUS_TOOL}" shared-status \
    --run-root "${S2_R5_RUN_ROOT}" --phase "$1" --program "$2" \
    --detail "${3:-}"
}

heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat \
      --run-root "${S2_R5_RUN_ROOT}" --shared || true
    sleep 20
  done
}

on_exit() {
  code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  if (( code != 0 )) && (( COMPLETED == 0 )); then
    touch "${S2_R5_FAILED_FILE}"
    status failed prepare_s2_r5_shared.sh \
      "preparation exited ${code}; inspect ${S2_R5_RUN_ROOT}/prepare.log" || true
  fi
}
trap on_exit EXIT
mkdir -p "${S2_R5_RUN_ROOT}"
exec > >(tee -a "${S2_R5_RUN_ROOT}/prepare.log") 2>&1
heartbeat_loop &
HEARTBEAT_PID=$!

status verifying prepare_s2_r5_shared.sh \
  "verifying shared dataset/DINO/PCA/Flow and protected R4-P0"

if [[ "${S2_R5_USE_S0_PREP:-0}" == "1" ]]; then
  : "${S2_R5_HF_TOKEN_FIFO:?S0 preparation needs token FIFO}"
  status s0_prepare prepare_s2_r4_shared.sh \
    "S0 path: dataset Xet/default workers; DINO no-Xet/one worker; mode-0600 FIFO"
  S2_R4_RUN_ROOT="${S2_R5_RUN_ROOT}" \
  S2_R4_READY_FILE="${S2_R5_RUN_ROOT}/s0_assets.ready" \
  S2_R4_FAILED_FILE="${S2_R5_RUN_ROOT}/s0_assets.failed" \
  S2_R4_HF_TOKEN_FIFO="${S2_R5_HF_TOKEN_FIFO}" \
  S2_R4_P0_CONFIG="${S2_R5_P0_CONFIG}" \
  S2_R4_FLOW_CHECKPOINT="${S2_R5_FLOW_CHECKPOINT:-}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
    bash "${FE_ROOT}/scripts/prepare_s2_r4_shared.sh"
  code=$?
  if (( code != 0 )); then exit "${code}"; fi
fi

required=(
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/take_photo/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/three_robots_stack_cube/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/camera_alignment/training_manifest.json"
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors"
  "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt"
  "${FE_ROOT}/artifacts/s1_r1_f1/checkpoint_080000.pt"
  "${S2_R5_PROTECTED_P0_SOURCE}"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    printf >&2 'Missing required shared input: %s\n' "${path}"
    printf >&2 'For missing HF assets rerun launcher with --prepare-from-s0.\n'
    exit 3
  fi
done

source_path="$(realpath "${S2_R5_PROTECTED_P0_SOURCE}")"
target_dir="${FE_ROOT}/artifacts/s2_r5_protected_p0"
target_path="${target_dir}/predictor.pt"
mkdir -p "${target_dir}"
if [[ -L "${target_path}" && ! -e "${target_path}" ]]; then unlink "${target_path}"; fi
if [[ ! -e "${target_path}" ]]; then
  ln -s "${source_path}" "${target_path}"
elif [[ "$(realpath "${target_path}")" != "${source_path}" ]]; then
  printf >&2 'Protected P0 target points elsewhere: %s\n' "${target_path}"
  exit 3
fi
sha256sum "${target_path}" | tee "${S2_R5_RUN_ROOT}/protected_p0_sha256.txt"
touch "${S2_R5_READY_FILE}"
status complete prepare_s2_r5_shared.sh \
  "shared five-task data/artifacts ready; protected P0 linked read-only by contract"
COMPLETED=1
printf 'S2-R5 shared preparation complete.\n'
