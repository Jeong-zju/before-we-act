#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${FE_ROOT}/.." && pwd)"
MODE="${1:-full}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FE_ROOT}/.uv-cache}"
export LPD_EXPERIMENT_SLUG="lpd_static_dino_act_moe"
export LPD_CONFIG="${FE_ROOT}/configs/static_act/lpd_static_dino_act_moe.yaml"
export LPD_CHECKPOINT="${FE_ROOT}/checkpoints/lpd_static_dino_act_moe/checkpoint_080000.pt"
export LPD_POLICY_KIND="static_act"
export ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT:-${WORKSPACE}/RoboFactory}"
export RF_PYTHON="${RF_PYTHON:-${ROBOFACTORY_ROOT}/.venv/bin/python}"
ROBOFACTORY_REPO_URL="${ROBOFACTORY_REPO_URL:-https://github.com/MARS-EAI/RoboFactory.git}"
ROBOFACTORY_COMMIT_SHA="${ROBOFACTORY_COMMIT_SHA:-5868242322414a91454e22f1dd9641f613ba1bcf}"
ROBOFACTORY_ASSET_REVISION="${ROBOFACTORY_ASSET_REVISION:-58ad250efb3de75f956c852ba8ad50e7ca30409f}"

doctor() {
  command -v uv >/dev/null
  command -v git >/dev/null
  command -v jq >/dev/null
  command -v nvidia-smi >/dev/null
  nvidia-smi -L
  (
    cd "${FE_ROOT}"
    uv run --frozen python -c \
      'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
  )
}

prepare_data() {
  local lift="${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
  local lpd="${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"
  if [[ -f "${lift}" && -f "${lpd}" ]]; then
    return
  fi
  : "${HF_M2_DATASET_REPO:?set HF_M2_DATASET_REPO when data is not pre-mounted}"
  : "${HF_M2_DATASET_REVISION:?set immutable HF_M2_DATASET_REVISION}"
  [[ "${HF_M2_DATASET_REVISION}" =~ ^[0-9a-f]{40}$ ]]
  local available_gb
  available_gb="$(df --output=avail -BG "${FE_ROOT}" | tail -1 | tr -dc '0-9')"
  if (( available_gb < 550 )); then
    printf >&2 'Need at least 550 GiB free to download the existing M2 corpus.\n'
    exit 3
  fi
  (
    cd "${FE_ROOT}"
    uv run --frozen hf download "${HF_M2_DATASET_REPO}" \
      --repo-type dataset \
      --revision "${HF_M2_DATASET_REVISION}" \
      --local-dir datasets/robofactory_multitask
  )
  test -f "${lift}"
  test -f "${lpd}"
}

prepare_vision() {
  if [[ -f "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors" ]]; then
    return
  fi
  : "${HF_TOKEN:?HF_TOKEN is required for the gated DINOv3 artifact}"
  (
    cd "${FE_ROOT}"
    uv run --frozen python scripts/prepare_dinov3_encoder.py \
      --encoder dinov3_vitl16_lvd
  )
}

prepare_robofactory() {
  if [[ ! -d "${ROBOFACTORY_ROOT}/.git" ]]; then
    git clone "${ROBOFACTORY_REPO_URL}" "${ROBOFACTORY_ROOT}"
    git -C "${ROBOFACTORY_ROOT}" checkout --detach "${ROBOFACTORY_COMMIT_SHA}"
  fi
  test "$(git -C "${ROBOFACTORY_ROOT}" rev-parse HEAD)" = "${ROBOFACTORY_COMMIT_SHA}"
  test -z "$(git -C "${ROBOFACTORY_ROOT}" status --porcelain --untracked-files=no)"
  if [[ ! -x "${RF_PYTHON}" ]]; then
    uv venv --python 3.9 "${ROBOFACTORY_ROOT}/.venv"
    uv pip install --python "${RF_PYTHON}" \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision==0.22.1
    uv pip install --python "${RF_PYTHON}" \
      mani-skill==3.0.0b12 sapien==3.0.0b1 setuptools==80.9.0 \
      zarr==2.18.2 'numcodecs<0.16' hydra-core==1.3.2 dill==0.3.9 \
      einops==0.8.1 diffusers==0.32.2 huggingface-hub==0.35.3 \
      pandas==2.2.3 numba==0.60.0
    uv pip install --python "${RF_PYTHON}" --no-deps \
      --editable "${ROBOFACTORY_ROOT}"
  fi
  if [[ ! -d "${ROBOFACTORY_ROOT}/robofactory/assets" ]]; then
    (
      cd "${FE_ROOT}"
      uv run --frozen hf download sparklexfantasy/RoboFactory_asset \
        --repo-type dataset \
        --revision "${ROBOFACTORY_ASSET_REVISION}" \
        --local-dir "${ROBOFACTORY_ROOT}/robofactory/assets"
    )
  fi
}

prepare() {
  (
    cd "${FE_ROOT}"
    uv sync --frozen
  )
  prepare_data
  prepare_vision
  prepare_robofactory
  doctor
}

train() {
  if [[ -f "${LPD_CHECKPOINT}" ]]; then
    printf 'Reusing completed checkpoint %s\n' "${LPD_CHECKPOINT}"
    return
  fi
  (
    cd "${FE_ROOT}"
    PYTHONUNBUFFERED=1 uv run --frozen python scripts/train_static_rgb_act_moe.py \
      --config "${LPD_CONFIG}" \
      --device cuda:0
  )
  test -f "${LPD_CHECKPOINT}"
}

gate() {
  LPD_GATE_MODE=gate "${FE_ROOT}/scripts/run_lpd_fixed_seed_gate.sh"
}

formal() {
  LPD_GATE_MODE=formal "${FE_ROOT}/scripts/run_lpd_fixed_seed_gate.sh"
}

case "${MODE}" in
  doctor) doctor ;;
  prepare) prepare ;;
  train) train ;;
  gate) doctor; gate ;;
  formal) doctor; formal ;;
  full) prepare; train; gate ;;
  *)
    printf >&2 'usage: %s {doctor|prepare|train|gate|formal|full}\n' "$0"
    exit 2
    ;;
esac
