#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S3_R6_RUN_ID:-s3-r6-round1}"
PREPARE_ARGS=()
if (( $# )); then
  if [[ "$1" == "--run-id" && -n "${2:-}" ]]; then RUN_ID="$2"; shift 2; fi
fi
ROBOFACTORY_ROOT="$(cd "${FE_ROOT}/.." && pwd)/RoboFactory"
if [[ ! -x "${ROBOFACTORY_ROOT}/.venv/bin/python" || \
      ! -s "${ROBOFACTORY_ROOT}/robofactory/assets/scenes/table/table.glb" ]]; then
  PREPARE_ARGS+=(--prepare-from-s0)
  printf 'RoboFactory is absent; enabling the S0 hidden-token FIFO preparation path.\n'
fi
printf 'Existing-server S3-R6: sharing five-task data/S2 parents; freshly training four Flow candidates two-by-two.\n'
exec bash "${FE_ROOT}/scripts/launch_s3_r6_2gpu_tmux.sh" \
  --run-id "${RUN_ID}" "${PREPARE_ARGS[@]}" "$@"
