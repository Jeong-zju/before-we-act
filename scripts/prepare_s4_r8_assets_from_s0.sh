#!/usr/bin/env bash
set -Eeuo pipefail

# Asset-only S0 bootstrap for an empty S4-R8 server.  Accepted model parents
# are deliberately outside this boundary: they must be supplied separately
# and are verified by prepare_s4_r8_shared.sh.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${S4_R8_RUN_ROOT:?set S4_R8_RUN_ROOT}"
: "${S4_R8_HF_TOKEN_FIFO:?set S4_R8_HF_TOKEN_FIFO}"
: "${S4_R8_ROBOFACTORY_ROOT:?set S4_R8_ROBOFACTORY_ROOT}"
: "${S4_R8_RF_PYTHON:?set S4_R8_RF_PYTHON}"
: "${UV_CACHE_DIR:?set UV_CACHE_DIR}"
: "${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}"

if [[ ! -p "${S4_R8_HF_TOKEN_FIFO}" ]]; then
  printf >&2 'Missing protected Hugging Face token FIFO: %s\n' \
    "${S4_R8_HF_TOKEN_FIFO}"
  exit 3
fi

source "${FE_ROOT}/scripts/hf_download_retry.sh"
STATUS_TOOL="${FE_ROOT}/scripts/s4_r8_runtime.py"
DATA_ROOT="${FE_ROOT}/datasets/robofactory_multitask"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT

HF_TOKEN_INPUT=""
cleanup_secret() {
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
  unlink "${S4_R8_HF_TOKEN_FIFO}" 2>/dev/null || true
}
trap cleanup_secret EXIT INT TERM
IFS= read -r HF_TOKEN_INPUT <"${S4_R8_HF_TOKEN_FIFO}"
unlink "${S4_R8_HF_TOKEN_FIFO}" 2>/dev/null || true
if [[ "${HF_TOKEN_INPUT}" != hf_* || "${HF_TOKEN_INPUT}" =~ [[:space:]] ]]; then
  printf >&2 'The protected Hugging Face token input was invalid.\n'
  exit 3
fi

status() {
  python3 "${STATUS_TOOL}" shared-status \
    --run-root "${S4_R8_RUN_ROOT}" \
    --phase "$1" --program "$2" --detail "${3:-}" || true
}

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
    --expected-task "${slug}" --expected-episodes 150 \
    --expected-agent-count "${expected_agent_count}"
}

required_growth=0
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r slug _repo _revision _agents expected_bytes <<<"${spec}"
  destination="${DATA_ROOT}/${slug}"
  current_bytes=0
  if [[ -d "${destination}" ]]; then
    current_bytes="$(du -sb "${destination}" | cut -f1)"
  fi
  if (( expected_bytes > current_bytes )); then
    required_growth=$((required_growth + expected_bytes - current_bytes))
  fi
done
headroom=$((32 * 1024 * 1024 * 1024))
available="$(df --output=avail -B1 "${FE_ROOT}" | tail -1 | tr -dc '0-9')"
if (( available < required_growth + headroom )); then
  printf >&2 \
    'S4-R8 datasets need %s bytes of net growth plus 32 GiB headroom; only %s bytes are free.\n' \
    "${required_growth}" "${available}"
  exit 3
fi

mkdir -p "${DATA_ROOT}"
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r slug repo revision agents _expected_bytes <<<"${spec}"
  if verify_dataset "${slug}" "${agents}" >/dev/null 2>&1; then
    status dataset prepare_s4_r8_assets_from_s0.sh \
      "verified ${slug}: 150/150 episodes; fixed revision already local"
    continue
  fi
  status dataset 'hf download' \
    "${slug}: fixed revision ${revision}; Xet=on default-workers=8; complete=0/150"
  (
    cd "${FE_ROOT}"
    HF_TOKEN="${HF_TOKEN_INPUT}" hf_download_with_retry \
      "${slug} training dataset" 0 "${repo}" \
      --repo-type dataset --revision "${revision}" \
      --local-dir "datasets/robofactory_multitask/${slug}"
  ) &
  download_pid=$!
  while kill -0 "${download_pid}" 2>/dev/null; do
    completed=0
    if [[ -d "${DATA_ROOT}/${slug}/hdf5" ]]; then
      completed="$(find "${DATA_ROOT}/${slug}/hdf5" -maxdepth 1 -type f \
        -name 'episode_*.hdf5' | wc -l)"
      completed="${completed//[[:space:]]/}"
    fi
    status dataset 'hf download' \
      "${slug}: Xet=on default-workers=8; complete=${completed}/150; retry<=5"
    sleep 15
  done
  if ! wait "${download_pid}"; then
    printf >&2 'Dataset download failed: %s\n' "${slug}"
    exit 3
  fi
  verify_dataset "${slug}" "${agents}"
done

status dinov3 prepare_dinov3_encoder.py \
  "pinned DINOv3; Xet=off workers=1; reuse final local-dir and partial cache"
(
  cd "${FE_ROOT}"
  HF_HUB_DISABLE_XET=1 HF_TOKEN="${HF_TOKEN_INPUT}" \
    uv run --frozen python scripts/prepare_dinov3_encoder.py \
      --encoder dinov3_vitl16_lvd
)

status robofactory run_lpd_single_5090.sh \
  "pinned RoboFactory checkout/environment/assets; Xet=off workers=1"
(
  cd "${FE_ROOT}"
  HF_TOKEN="${HF_TOKEN_INPUT}" \
  GPU_INDEX=0 \
  ROBOFACTORY_ROOT="${S4_R8_ROBOFACTORY_ROOT}" \
  RF_PYTHON="${S4_R8_RF_PYTHON}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
    bash scripts/run_lpd_single_5090.sh prepare
)

HF_TOKEN_INPUT=""
unset HF_TOKEN_INPUT
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r slug _repo _revision agents _expected_bytes <<<"${spec}"
  verify_dataset "${slug}" "${agents}" >/dev/null
done
test -s "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors"
test -s "${S4_R8_ROBOFACTORY_ROOT}/robofactory/assets/scenes/table/table.glb"
test -x "${S4_R8_RF_PYTHON}"
status s0_assets_complete prepare_s4_r8_assets_from_s0.sh \
  "five datasets, DINOv3 and RoboFactory are complete; accepted parents remain fail-closed"

