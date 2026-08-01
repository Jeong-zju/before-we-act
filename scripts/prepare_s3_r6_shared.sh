#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in S3_R6_RUN_ROOT S3_R6_READY_FILE S3_R6_FAILED_FILE; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'Missing %s\n' "${name}"; exit 2; fi
done
STATUS_TOOL="${FE_ROOT}/scripts/s3_r6_runtime.py"
HEARTBEAT_PID=""
COMPLETED=0

status() {
  python3 "${STATUS_TOOL}" shared-status --run-root "${S3_R6_RUN_ROOT}" \
    --phase "$1" --program "$2" --detail "${3:-}"
}
heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat --run-root "${S3_R6_RUN_ROOT}" \
      --shared || true
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
    touch "${S3_R6_FAILED_FILE}"
    status failed prepare_s3_r6_shared.sh \
      "preparation exited ${code}; inspect ${S3_R6_RUN_ROOT}/prepare.log" || true
  fi
}
trap on_exit EXIT
mkdir -p "${S3_R6_RUN_ROOT}" || exit $?
exec > >(tee -a "${S3_R6_RUN_ROOT}/prepare.log") 2>&1
heartbeat_loop & HEARTBEAT_PID=$!
status verifying prepare_s3_r6_shared.sh \
  "locating shared five-task data, DINO/PCA, protected-own and R5-P0"

if [[ "${S3_R6_USE_S0_PREP:-0}" == "1" ]]; then
  for name in S3_R6_HF_TOKEN_FIFO S3_R6_ROBOFACTORY_ROOT S3_R6_RF_PYTHON; do
    if [[ -z "${!name:-}" ]]; then
      printf >&2 'S0 preparation requires %s.\n' "${name}"; exit 3
    fi
  done
  status s0_prepare prepare_s3_r6_from_s0.sh \
    "S0 FIFO path: datasets, DINO and RoboFactory; then five-task/PCA verification"
  S3_R6_HF_TOKEN_FIFO="${S3_R6_HF_TOKEN_FIFO}" \
  S3_R6_ROBOFACTORY_ROOT="${S3_R6_ROBOFACTORY_ROOT}" \
  S3_R6_RF_PYTHON="${S3_R6_RF_PYTHON}" \
  S3_R6_P0_CONFIG="${FE_ROOT}/configs/wam_flow/s2_r5_protected_team.yaml" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
  ROBOFACTORY_ROOT="${S3_R6_ROBOFACTORY_ROOT}" \
  RF_PYTHON="${S3_R6_RF_PYTHON}" \
    bash "${FE_ROOT}/scripts/prepare_s3_r6_from_s0.sh" || exit $?
fi

find_latest() {
  pattern="$1"
  while IFS= read -r candidate; do printf '%s' "${candidate}"; return 0; done < <(
    find "${FE_ROOT}/outputs" -type f -path "${pattern}" -printf '%T@ %p\n' \
      2>/dev/null | sort -rn | cut -d' ' -f2-
  )
  return 1
}

PROTECTED_SOURCE="${S3_R6_PROTECTED_OWN_SOURCE:-}"
TEAM_SOURCE="${S3_R6_PROTECTED_TEAM_SOURCE:-}"
if [[ -z "${PROTECTED_SOURCE}" && -f "${FE_ROOT}/artifacts/s2_r5_protected_p0/predictor.pt" ]]; then
  PROTECTED_SOURCE="${FE_ROOT}/artifacts/s2_r5_protected_p0/predictor.pt"
fi
if [[ -z "${PROTECTED_SOURCE}" ]]; then
  PROTECTED_SOURCE="$(find_latest '*/s2_r4_runs/*/candidates/p0/checkpoints/predictor.pt')"
fi
if [[ -z "${TEAM_SOURCE}" ]]; then
  TEAM_SOURCE="$(find_latest '*/s2_r5_runs/*/candidates/p0/checkpoints/predictor.pt')"
fi
required=(
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/take_photo/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/three_robots_stack_cube/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/camera_alignment/training_manifest.json"
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors"
  "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt"
  "${S3_R6_ROBOFACTORY_ROOT:-}/robofactory/assets/scenes/table/table.glb"
  "${S3_R6_RF_PYTHON:-}"
  "${PROTECTED_SOURCE}" "${TEAM_SOURCE}"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    printf >&2 'Missing S3-R6 shared input: %s\n' "${path}"
    printf >&2 'HF assets may use --prepare-from-s0; parent checkpoints must come from accepted S2 runs.\n'
    exit 3
  fi
done

parent_dir="${FE_ROOT}/artifacts/s3_r6_parent"
mkdir -p "${parent_dir}" || exit $?
link_parent() {
  source_path="$(realpath "$1")"
  target_path="$2"
  if [[ -L "${target_path}" && ! -e "${target_path}" ]]; then unlink "${target_path}"; fi
  if [[ ! -e "${target_path}" ]]; then
    ln -s "${source_path}" "${target_path}" || return $?
  elif [[ "$(realpath "${target_path}")" != "${source_path}" ]]; then
    printf >&2 'S3 parent link already points elsewhere: %s\n' "${target_path}"
    return 3
  fi
}
link_parent "${PROTECTED_SOURCE}" "${parent_dir}/protected_own.pt" || exit $?
link_parent "${TEAM_SOURCE}" "${parent_dir}/protected_team.pt" || exit $?

protected_sha="$(sha256sum "${parent_dir}/protected_own.pt" | awk '{print $1}')"
team_sha="$(sha256sum "${parent_dir}/protected_team.pt" | awk '{print $1}')"
if [[ "${protected_sha}" != "c04f8ea12c5b6d8f7c04992d7dd4a8c0a33aa7d0058987679e6553b17e410a2f" ]]; then
  printf >&2 'Protected-own parent hash is not the accepted R4-P0 artifact: %s\n' "${protected_sha}"
  exit 3
fi
if [[ "${team_sha}" != "fcc0af76c2acd6805750f12e828a1249eb91e466e51f4aa77c118b6e9d330c67" ]]; then
  printf >&2 'Protected-team parent hash is not the accepted R5-P0 artifact: %s\n' "${team_sha}"
  exit 3
fi
sha256sum "${parent_dir}/protected_own.pt" \
  "${parent_dir}/protected_team.pt" "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt" \
  | tee "${S3_R6_RUN_ROOT}/shared_artifact_sha256.txt"
touch "${S3_R6_READY_FILE}"
status complete prepare_s3_r6_shared.sh \
  "shared five-task data/artifacts/RoboFactory and accepted S2 parents verified; Flow trains per candidate"
COMPLETED=1
printf 'S3-R6 shared preparation complete.\n'
