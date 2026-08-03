#!/usr/bin/env bash

# Existing-server convenience entry point.  The underlying launcher owns every
# identity, GPU, worktree, tmux, resume and acceptance check.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S4_R8_RUN_ID:-s4-r8-parallel-fast30k-round1}"
PREPARE_ARGS=()

if (( $# )) && [[ "$1" == "--run-id" && -n "${2:-}" ]]; then
  RUN_ID="$2"
  shift 2
fi

ROBOFACTORY_ROOT="$(cd "${FE_ROOT}/.." && pwd)/RoboFactory"
HF_ASSETS_COMPLETE=1
for required in \
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/take_photo/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/three_robots_stack_cube/training_manifest.json" \
  "${FE_ROOT}/datasets/robofactory_multitask/camera_alignment/training_manifest.json" \
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors" \
  "${ROBOFACTORY_ROOT}/robofactory/assets/scenes/table/table.glb" \
  "${ROBOFACTORY_ROOT}/.venv/bin/python"; do
  [[ -f "${required}" ]] || HF_ASSETS_COMPLETE=0
done
if (( HF_ASSETS_COMPLETE == 0 )); then
  PREPARE_ARGS+=(--prepare-from-s0)
  printf 'One or more shared HF assets are absent; enabling the S0 hidden-token FIFO preparation path.\n'
fi

printf 'Existing-server S4-R8: reusing one-copy data/cache and exact accepted ancestors.\n'
exec bash "${FE_ROOT}/scripts/launch_s4_r8_2gpu_tmux.sh" \
  --run-id "${RUN_ID}" "${PREPARE_ARGS[@]}" "$@"
