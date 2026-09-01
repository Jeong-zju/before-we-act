#!/usr/bin/env bash
set -Eeuo pipefail
MARS_OPENVLA_STAGE=smoke MARS_OPENVLA_MAX_STEPS=2 MARS_OPENVLA_SAVE_FREQ=2 MARS_OPENVLA_BATCH_SIZE=1 MARS_OPENVLA_GRAD_ACCUM=1 MARS_OPENVLA_RUN_ID=openvla7b_mars_control_lora_r32_smoke MARS_OPENVLA_NPROC=4 exec "$(dirname "$0")/run_train.sh"
