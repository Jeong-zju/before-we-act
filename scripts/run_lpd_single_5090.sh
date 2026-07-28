#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${FE_ROOT}/.." && pwd)"
MODE="${1:-full}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FE_ROOT}/.uv-cache}"
export LPD_EXPERIMENT_SLUG="${LPD_EXPERIMENT_SLUG:-lpd_static_dino_act_moe_uniform_loss}"
export LPD_CONFIG="${LPD_CONFIG:-${FE_ROOT}/configs/static_act/lpd_static_dino_act_moe.yaml}"
export LPD_CHECKPOINT="${LPD_CHECKPOINT:-${FE_ROOT}/checkpoints/lpd_static_dino_act_moe_uniform_loss/checkpoint_080000.pt}"
export LPD_POLICY_KIND="${LPD_POLICY_KIND:-static_act}"
export LPD_PROGRESS_LOG="${LPD_PROGRESS_LOG:-${FE_ROOT}/outputs/${LPD_EXPERIMENT_SLUG}/training_progress.jsonl}"
export ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT:-${WORKSPACE}/RoboFactory}"
export RF_PYTHON="${RF_PYTHON:-${ROBOFACTORY_ROOT}/.venv/bin/python}"
ROBOFACTORY_REPO_URL="${ROBOFACTORY_REPO_URL:-https://github.com/MARS-EAI/RoboFactory.git}"
ROBOFACTORY_COMMIT_SHA="${ROBOFACTORY_COMMIT_SHA:-5868242322414a91454e22f1dd9641f613ba1bcf}"
ROBOFACTORY_ASSET_REVISION="${ROBOFACTORY_ASSET_REVISION:-58ad250efb3de75f956c852ba8ad50e7ca30409f}"
LIFT_DATASET_REPO="${LIFT_DATASET_REPO:-zeno-ai/robofactory-lift-barrier-multiview}"
LIFT_DATASET_REVISION="${LIFT_DATASET_REVISION:-6ab620091677e69370412f08cd7adecacc28c146}"
LPD_DATASET_REPO="${LPD_DATASET_REPO:-zeno-ai/robofactory-long-pipeline-delivery-multiview}"
LPD_DATASET_REVISION="${LPD_DATASET_REVISION:-fee628311ff52a3ae0ddfddf82379c63d28f7533}"

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

verify_hf_token() {
  : "${HF_TOKEN:?HF_TOKEN is required for DINOv3 and dataset downloads}"
  if [[ "${HF_TOKEN}" != hf_* || "${HF_TOKEN}" == *[[:space:]]* ]]; then
    printf >&2 'HF_TOKEN must be a single Hugging Face user access token beginning with hf_.\n'
    exit 3
  fi
  (
    cd "${FE_ROOT}"
    # Pass the exact interactive token directly to the Hub API. Do not rely on
    # a cached `hf auth login` credential and never place the token in argv.
    HF_TOKEN="${HF_TOKEN}" uv run --frozen python -c \
      'import os; from huggingface_hub import HfApi; HfApi().whoami(token=os.environ["HF_TOKEN"]); print("Hugging Face token verified.")'
  )
}

prepare_data() {
  local lift="${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
  local lpd="${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"
  if [[ -f "${lift}" && -f "${lpd}" ]]; then
    return
  fi
  if [[ -n "${M2_DATA_ROOT:-}" ]]; then
    local mounted
    mounted="$(realpath "${M2_DATA_ROOT}")"
    test -f "${mounted}/lift_barrier/training_manifest.json"
    test -f "${mounted}/long_pipeline_delivery/training_manifest.json"
    if [[ -e "${FE_ROOT}/datasets/robofactory_multitask" ]]; then
      printf >&2 'Refusing to replace partial data path: %s\n' \
        "${FE_ROOT}/datasets/robofactory_multitask"
      exit 3
    fi
    mkdir -p "${FE_ROOT}/datasets"
    ln -s "${mounted}" "${FE_ROOT}/datasets/robofactory_multitask"
    test -f "${lift}"
    test -f "${lpd}"
    return
  fi
  local available_gb
  available_gb="$(df --output=avail -BG "${FE_ROOT}" | tail -1 | tr -dc '0-9')"
  if (( available_gb < 550 )); then
    printf >&2 'Need at least 550 GiB free to download the existing M2 corpus.\n'
    exit 3
  fi
  if [[ -n "${HF_M2_DATASET_REPO:-}" ]]; then
    : "${HF_M2_DATASET_REVISION:?set immutable HF_M2_DATASET_REVISION}"
    [[ "${HF_M2_DATASET_REVISION}" =~ ^[0-9a-f]{40}$ ]]
    (
      cd "${FE_ROOT}"
      HF_TOKEN="${HF_TOKEN}" uv run --frozen hf download "${HF_M2_DATASET_REPO}" \
        --repo-type dataset \
        --revision "${HF_M2_DATASET_REVISION}" \
        --local-dir datasets/robofactory_multitask
    )
  else
    [[ "${LIFT_DATASET_REVISION}" =~ ^[0-9a-f]{40}$ ]]
    [[ "${LPD_DATASET_REVISION}" =~ ^[0-9a-f]{40}$ ]]
    if [[ ! -f "${lift}" ]]; then
      (
        cd "${FE_ROOT}"
        HF_TOKEN="${HF_TOKEN}" uv run --frozen hf download "${LIFT_DATASET_REPO}" \
          --repo-type dataset \
          --revision "${LIFT_DATASET_REVISION}" \
          --local-dir datasets/robofactory_multitask/lift_barrier
      )
    fi
    if [[ ! -f "${lpd}" ]]; then
      (
        cd "${FE_ROOT}"
        HF_TOKEN="${HF_TOKEN}" uv run --frozen hf download "${LPD_DATASET_REPO}" \
          --repo-type dataset \
          --revision "${LPD_DATASET_REVISION}" \
          --local-dir datasets/robofactory_multitask/long_pipeline_delivery
      )
    fi
  fi
  test -f "${lift}"
  test -f "${lpd}"
}

prepare_vision() {
  : "${HF_TOKEN:?HF_TOKEN is required for the gated DINOv3 artifact}"
  (
    cd "${FE_ROOT}"
    HF_TOKEN="${HF_TOKEN}" uv run --frozen python scripts/prepare_dinov3_encoder.py \
      --encoder dinov3_vitl16_lvd
  )
}

prepare_robofactory() {
  local asset_root="${ROBOFACTORY_ROOT}/robofactory/assets"
  local asset_archive="${asset_root}/assets.zip"
  local asset_sentinel="${asset_root}/scenes/table/table.glb"
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
  if [[ ! -s "${asset_sentinel}" ]]; then
    (
      cd "${FE_ROOT}"
      HF_TOKEN="${HF_TOKEN}" uv run --frozen hf download sparklexfantasy/RoboFactory_asset \
        --repo-type dataset \
        --revision "${ROBOFACTORY_ASSET_REVISION}" \
        --local-dir "${asset_root}"
    )
    test -s "${asset_archive}"
    "${RF_PYTHON}" "${FE_ROOT}/scripts/extract_robofactory_assets.py" \
      --archive "${asset_archive}" \
      --output-dir "${asset_root}" \
      --require scenes/table/table.glb
  fi
  test -s "${asset_sentinel}"
}

prepare() {
  (
    cd "${FE_ROOT}"
    uv sync --frozen
  )
  verify_hf_token
  # Preserve the tested one-click preparation order from c79ff1e and 859cecd.
  prepare_data
  prepare_vision
  prepare_robofactory
  doctor
}

train() {
  if [[ -e "${LPD_CHECKPOINT}" ]]; then
    printf 'Reusing completed checkpoint %s\n' "${LPD_CHECKPOINT}"
    return
  fi
  case "${LPD_POLICY_KIND}" in
    static_act)
      (
        cd "${FE_ROOT}"
        PYTHONUNBUFFERED=1 uv run --frozen python scripts/train_static_rgb_act_moe.py \
          --config "${LPD_CONFIG}" \
          --device cuda:0 \
          --progress-log "${LPD_PROGRESS_LOG}"
      )
      ;;
    agent_flow)
      (
        cd "${FE_ROOT}"
        PYTHONUNBUFFERED=1 uv run --frozen python \
          scripts/train_agent_factorized_flow_wam.py \
          --config "${LPD_CONFIG}" \
          --device cuda:0 \
          --progress-log "${LPD_PROGRESS_LOG}"
      )
      ;;
    wam)
      (
        cd "${FE_ROOT}"
        PYTHONUNBUFFERED=1 uv run --frozen python scripts/train_robofactory_m2.py \
          --config "${LPD_CONFIG}" \
          --device cuda:0 \
          --progress-log "${LPD_PROGRESS_LOG}"
      )
      ;;
    *)
      printf >&2 'unknown LPD_POLICY_KIND=%q\n' "${LPD_POLICY_KIND}"
      exit 2
      ;;
  esac
  test -e "${LPD_CHECKPOINT}"
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
