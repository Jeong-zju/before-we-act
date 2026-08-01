#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S3_R6_RUN_ID:-s3-r6-round1}"
if (( $# )); then
  if [[ "$1" == "--run-id" && -n "${2:-}" ]]; then RUN_ID="$2"; shift 2; fi
fi
printf 'Existing-server S3-R6: reusing data/parents and scheduling four branches two-by-two.\n'
exec bash "${FE_ROOT}/scripts/launch_s3_r6_2gpu_tmux.sh" \
  --run-id "${RUN_ID}" "$@"
