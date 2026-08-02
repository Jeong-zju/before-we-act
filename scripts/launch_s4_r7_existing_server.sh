#!/usr/bin/env bash

# Existing-server convenience entry point.  The underlying launcher owns every
# identity, GPU, worktree, tmux, resume and acceptance check.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S4_R7_RUN_ID:-s4-r7-round1}"
PREPARE_ARGS=()

if (( $# )) && [[ "$1" == "--run-id" && -n "${2:-}" ]]; then
  RUN_ID="$2"
  shift 2
fi

ROBOFACTORY_ROOT="$(cd "${FE_ROOT}/.." && pwd)/RoboFactory"
if [[ ! -x "${ROBOFACTORY_ROOT}/.venv/bin/python" || \
      ! -s "${ROBOFACTORY_ROOT}/robofactory/assets/scenes/table/table.glb" ]]; then
  PREPARE_ARGS+=(--prepare-from-s0)
  printf 'RoboFactory is absent; enabling the S0 hidden-token FIFO preparation path.\n'
fi

printf 'Existing-server S4-R7: reusing one-copy data/cache and exact accepted ancestors.\n'
exec bash "${FE_ROOT}/scripts/launch_s4_r7_2gpu_tmux.sh" \
  --run-id "${RUN_ID}" "${PREPARE_ARGS[@]}" "$@"
