#!/usr/bin/env bash
set -Eeuo pipefail
export OPENPI_RUN_STAGE=smoke
export OPENPI_EXP_NAME=h200_gate
export OPENPI_MAX_STEPS=2
export OPENPI_BATCH_SIZE=${OPENPI_SMOKE_BATCH_SIZE:-2}
export OPENPI_SAVE_INTERVAL=1
export OPENPI_KEEP_PERIOD=1
export OPENPI_FSDP_DEVICES=1
exec /workspace/repos/before-we-act/deployment/vla_baselines/run_pi05.sh
