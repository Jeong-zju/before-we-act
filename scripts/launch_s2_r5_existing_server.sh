#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S2_R5_RUN_ID:-s2-r5-round1}"
if (( $# )); then
  if [[ "$1" == "--run-id" && -n "${2:-}" ]]; then RUN_ID="$2"; shift 2; fi
fi
printf 'Existing-server S2-R5: auto-detecting shared assets, R4-P0, run/resume and monitor.\n'
exec bash "${FE_ROOT}/scripts/launch_s2_r5_2gpu_tmux.sh" \
  --run-id "${RUN_ID}" "$@"
