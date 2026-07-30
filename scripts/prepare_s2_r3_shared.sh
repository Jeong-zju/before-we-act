#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S2_R3_RUN_ROOT:?set S2_R3_RUN_ROOT}"
: "${S2_R3_READY_FILE:?set S2_R3_READY_FILE}"
: "${S2_R3_FAILED_FILE:?set S2_R3_FAILED_FILE}"
: "${S2_R3_HF_TOKEN_FIFO:?set S2_R3_HF_TOKEN_FIFO}"
: "${S2_R3_W0_CONFIG:?set S2_R3_W0_CONFIG}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"
SHARED_LOCK="${S2_R3_SHARED_PREPARE_LOCK:-${FE_ROOT}/outputs/s2_r3_runs/.shared_prepare.lock}"
STATUS_TOOL="${FE_ROOT}/scripts/s2_r3_runtime.py"
LOG_PATH="${S2_R3_RUN_ROOT}/prepare.log"
STAGES_LOG="${S2_R3_RUN_ROOT}/prepare_stages.jsonl"
PROGRESS_LOG="${S2_R3_RUN_ROOT}/prepare_progress.jsonl"
source "${FE_ROOT}/scripts/hf_download_retry.sh"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
if [[ ! "${HF_HUB_DOWNLOAD_TIMEOUT}" =~ ^[1-9][0-9]*$ || \
      ! "${HF_HUB_ETAG_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  printf >&2 'HF Hub timeout values must be positive integer seconds.\n'
  exit 3
fi
export \
  HF_HUB_DISABLE_XET=1 \
  HF_XET_HIGH_PERFORMANCE=0 \
  HF_HUB_DOWNLOAD_TIMEOUT \
  HF_HUB_ETAG_TIMEOUT

mkdir -p "${S2_R3_RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

status() {
  python3 "${STATUS_TOOL}" shared-status \
    --run-root "${S2_R3_RUN_ROOT}" \
    --phase "$1" \
    --program "$2" \
    --detail "${3:-}"
}

heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat \
      --run-root "${S2_R3_RUN_ROOT}" \
      --shared || true
    sleep 20
  done
}

HF_TOKEN_INPUT=""
HEARTBEAT_PID=""
on_exit() {
  local code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
  unlink "${S2_R3_HF_TOKEN_FIFO}" 2>/dev/null || true
  if (( code != 0 )); then
    touch "${S2_R3_FAILED_FILE}"
    status failed prepare_s2_r3_shared.sh \
      "shared preparation failed with code ${code}; inspect ${LOG_PATH}" || true
  fi
}
trap on_exit EXIT

if [[ ! -p "${S2_R3_HF_TOKEN_FIFO}" ]]; then
  printf >&2 'Missing protected Hugging Face token FIFO: %s\n' \
    "${S2_R3_HF_TOKEN_FIFO}"
  exit 3
fi
IFS= read -r HF_TOKEN_INPUT <"${S2_R3_HF_TOKEN_FIFO}"
unlink "${S2_R3_HF_TOKEN_FIFO}" 2>/dev/null || true
if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
  printf >&2 'The protected Hugging Face token input was invalid.\n'
  exit 3
fi

heartbeat_loop &
HEARTBEAT_PID=$!
mkdir -p "$(dirname "${SHARED_LOCK}")"
exec {S2_R3_PREPARE_LOCK_FD}>"${SHARED_LOCK}"
status waiting prepare_s2_r3_shared.sh "waiting for shared preparation lock"
flock -x "${S2_R3_PREPARE_LOCK_FD}"

status environment 'uv sync --frozen' "synchronizing pinned Python environment"
(
  cd "${FE_ROOT}"
  uv sync --frozen
)

DATA_ROOT="${FE_ROOT}/datasets/robofactory_multitask"
DATASET_SPECS=(
  "lift_barrier|zeno-ai/robofactory-lift-barrier-multiview|6ab620091677e69370412f08cd7adecacc28c146|2|36216038270"
  "long_pipeline_delivery|zeno-ai/robofactory-long-pipeline-delivery-multiview|fee628311ff52a3ae0ddfddf82379c63d28f7533|4|381222318154"
  "take_photo|zeno-ai/robofactory-take-photo-multiview|3966385a4c688a5610d4b6cde044150f6b73d320|4|140081439177"
  "three_robots_stack_cube|zeno-ai/robofactory-three-robots-stack-cube-multiview|d0ae346bf2ce63ec801af1f036c08a4a91faf366|3|220968602723"
  "camera_alignment|zeno-ai/robofactory-camera-alignment-multiview|e204af13f7191dfd86dab3da529316a51558f479|3|63550797663"
)

verify_dataset() {
  local slug="$1"
  local expected_agent_count="$2"
  python3 "${FE_ROOT}/scripts/verify_s2_r3_dataset_local.py" \
    --manifest "${DATA_ROOT}/${slug}/training_manifest.json" \
    --expected-task "${slug}" \
    --expected-episodes 150 \
    --expected-agent-count "${expected_agent_count}"
}

download_dataset() {
  local slug="$1"
  local repo="$2"
  local revision="$3"
  local expected_agent_count="$4"
  local destination="${DATA_ROOT}/${slug}"
  local download_pid
  local completed
  status dataset 'hf download' \
    "S0 transfer mode: ${repo}@${revision}; Xet enabled; workers=8; episodes=0/150"
  (
    cd "${FE_ROOT}"
    HF_TOKEN="${HF_TOKEN_INPUT}" hf_download_with_retry \
      "${slug} training dataset" \
      0 \
      "${repo}" \
        --repo-type dataset \
        --revision "${revision}" \
        --local-dir "datasets/robofactory_multitask/${slug}"
  ) &
  download_pid=$!
  while kill -0 "${download_pid}" 2>/dev/null; do
    completed=0
    if [[ -d "${destination}/hdf5" ]]; then
      completed="$(
        find "${destination}/hdf5" \
          -maxdepth 1 \
          -type f \
          -name 'episode_*.hdf5' \
          | wc -l
      )"
      completed="${completed//[[:space:]]/}"
    fi
    status dataset 'hf download' \
      "${repo}@${revision}; S0 mode Xet=on workers=8; complete episodes=${completed}/150; attempt max=5"
    sleep 15
  done
  wait "${download_pid}"
  verify_dataset "${slug}" "${expected_agent_count}"
}

MISSING_DATA=0
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r \
    slug _repo _revision expected_agent_count _expected_bytes <<<"${spec}"
  verify_dataset "${slug}" "${expected_agent_count}" >/dev/null 2>&1 \
    || MISSING_DATA=1
done
if (( MISSING_DATA )); then
  REQUIRED_GROWTH_BYTES=0
  for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r \
      slug _repo _revision _expected_agent_count expected_bytes <<<"${spec}"
    destination="${DATA_ROOT}/${slug}"
    current_bytes=0
    if [[ -d "${destination}" ]]; then
      current_bytes="$(du -sb "${destination}" | cut -f1)"
    fi
    if (( expected_bytes > current_bytes )); then
      REQUIRED_GROWTH_BYTES=$((
        REQUIRED_GROWTH_BYTES + expected_bytes - current_bytes
      ))
    fi
  done
  DOWNLOAD_HEADROOM_BYTES=$((32 * 1024 * 1024 * 1024))
  REQUIRED_BYTES=$((REQUIRED_GROWTH_BYTES + DOWNLOAD_HEADROOM_BYTES))
  AVAILABLE_BYTES="$(
    df --output=avail -B1 "${FE_ROOT}" | tail -1 | tr -dc '0-9'
  )"
  if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
    printf >&2 \
      'Five-task update needs %s bytes of net growth plus 32 GiB replacement headroom; only %s bytes are free.\n' \
      "${REQUIRED_GROWTH_BYTES}" "${AVAILABLE_BYTES}"
    exit 3
  fi
  status dataset prepare_s2_r3_shared.sh \
    "disk preflight passed: net growth=${REQUIRED_GROWTH_BYTES} bytes; free=${AVAILABLE_BYTES} bytes; headroom=32 GiB"
fi

for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r \
    slug repo revision expected_agent_count _expected_bytes <<<"${spec}"
  manifest="${DATA_ROOT}/${slug}/training_manifest.json"
  if verify_dataset "${slug}" "${expected_agent_count}" >/dev/null 2>&1; then
    status dataset prepare_s2_r3_shared.sh \
      "verified all 150 local ${slug} episodes plus complete agent cameras"
    continue
  fi
  if [[ -f "${manifest}" ]]; then
    status dataset prepare_s2_r3_shared.sh \
      "${slug} manifest exists but local files are incomplete; resuming in place"
  fi
  status dataset 'hf download' \
    "S0 Xet/default-8-worker in-place download/resume: ${repo}@${revision}"
  download_dataset \
    "${slug}" "${repo}" "${revision}" "${expected_agent_count}"
done

DINO_ROOT="${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd"
DINO_REVISION="dd0a398fa8e84f2a37179332f6c561d20276300b"
if [[ ! -f "${DINO_ROOT}/config.json" || \
      ! -f "${DINO_ROOT}/model.safetensors" ]]; then
  status dinov3 'hf download' \
    "direct single-worker in-place DINOv3 download with Xet disabled"
  (
    cd "${FE_ROOT}"
    HF_TOKEN="${HF_TOKEN_INPUT}" \
      uv run --frozen hf download \
        facebook/dinov3-vitl16-pretrain-lvd1689m \
        config.json model.safetensors \
        --revision "${DINO_REVISION}" \
        --local-dir artifacts/vision/dinov3_vitl16_lvd \
        --max-workers 1
  )
fi
HF_TOKEN_INPUT=""
unset HF_TOKEN_INPUT
status dinov3 prepare_dinov3_encoder.py \
  "verifying pinned local DINOv3 config and weights"
(
  cd "${FE_ROOT}"
  HF_HUB_OFFLINE=1 uv run --frozen python scripts/prepare_dinov3_encoder.py \
    --encoder dinov3_vitl16_lvd
)

FLOW_TARGET="${FE_ROOT}/artifacts/s1_r1_f1/checkpoint_080000.pt"
FLOW_SOURCE="${S2_R3_FLOW_CHECKPOINT:-}"
if [[ -n "${FLOW_SOURCE}" && ! -f "${FLOW_SOURCE}" ]]; then
  status flow_recovery recover_s1_r1_f1_checkpoint.sh \
    "configured Flow checkpoint is missing; falling back to automatic reconstruction"
  FLOW_SOURCE=""
fi
if [[ -z "${FLOW_SOURCE}" && -f "${FLOW_TARGET}" ]]; then
  FLOW_SOURCE="${FLOW_TARGET}"
fi
if [[ -z "${FLOW_SOURCE}" ]]; then
  while IFS= read -r candidate; do
    FLOW_SOURCE="${candidate}"
    break
  done < <(
    find "${FE_ROOT}/outputs/s1_r1_runs" \
      -path '*/candidates/f1/checkpoints/s1_r1_f1_flow_cold/checkpoint_080000.pt' \
      -type f 2>/dev/null | sort -r
  )
fi
if [[ -z "${FLOW_SOURCE}" || ! -f "${FLOW_SOURCE}" ]]; then
  status flow_recovery train_agent_factorized_flow_wam.py \
    "S1-R1 F1 checkpoint absent; auto-retraining 0/80000 on GPU0 with persistent resume"
  (
    cd "${FE_ROOT}"
    S2_R3_RUN_ROOT="${S2_R3_RUN_ROOT}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
      bash scripts/recover_s1_r1_f1_checkpoint.sh
  )
  FLOW_SOURCE="${FLOW_TARGET}"
fi
FLOW_SOURCE="$(realpath "${FLOW_SOURCE}")"
mkdir -p "$(dirname "${FLOW_TARGET}")"
if [[ -L "${FLOW_TARGET}" && ! -e "${FLOW_TARGET}" ]]; then
  unlink "${FLOW_TARGET}"
fi
if [[ ! -e "${FLOW_TARGET}" ]]; then
  ln -s "${FLOW_SOURCE}" "${FLOW_TARGET}"
elif [[ "$(realpath "${FLOW_TARGET}")" != "${FLOW_SOURCE}" ]]; then
  printf >&2 'Existing shared Flow checkpoint resolves to a different file: %s\n' \
    "${FLOW_TARGET}"
  exit 3
fi
status flow_checkpoint python \
  "verifying frozen S1-R1 F1 recipe, config and two manifest identities"
(
  cd "${FE_ROOT}"
  uv run --frozen python scripts/verify_s1_r1_f1_checkpoint.py \
    "${FLOW_TARGET}" \
    --config configs/wam_flow/s1_r1_f1_flow_cold.yaml
)

PCA_PATH="${FE_ROOT}/artifacts/s2_r3/dino_pca_statistics.pt"
status pca prepare_s2_r3_artifacts.py \
  "fitting/reusing train-only DINO PCA and target statistics on GPU0"
(
  cd "${FE_ROOT}"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
    uv run --frozen python scripts/prepare_s2_r3_artifacts.py \
      --config "${S2_R3_W0_CONFIG}" \
      --device cuda:0 \
      --output "${PCA_PATH}" \
      --progress-log "${PROGRESS_LOG}"
)

status hashes sha256sum "recording five manifests, DINO, PCA and Flow identities"
sha256sum \
  "${DATA_ROOT}/lift_barrier/training_manifest.json" \
  "${DATA_ROOT}/long_pipeline_delivery/training_manifest.json" \
  "${DATA_ROOT}/take_photo/training_manifest.json" \
  "${DATA_ROOT}/three_robots_stack_cube/training_manifest.json" \
  "${DATA_ROOT}/camera_alignment/training_manifest.json" \
  "${DINO_ROOT}/config.json" \
  "${DINO_ROOT}/model.safetensors" \
  "${PCA_PATH}" \
  "${FLOW_TARGET}" \
  | tee "${S2_R3_RUN_ROOT}/shared_artifact_sha256.txt"

touch "${S2_R3_READY_FILE}"
status complete prepare_s2_r3_shared.sh \
  "five-task dataset, DINO, PCA/statistics and Flow checkpoint ready"
printf 'S2-R3 shared preparation complete.\n'
