#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}
PROFILE="$RUN_ROOT/smoke/openvla_oft/resource_profile.json"
PYTHON=${OPENVLA_PYTHON:-/workspace/venvs/openvla/bin/python}
TRAIN=/workspace/repos/before-we-act/deployment/vla_baselines/run_openvla_oft.sh

read -r batch accumulation < <(
  "$PYTHON" - "$PROFILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "complete" and p["world_size"] == 4
print(int(p["per_device_batch"]), int(p["formal_gradient_accumulation"]))
PY
)

export OPENVLA_STAGE=formal
export OPENVLA_RUN_ID=openvla7b_robofactory_lora_r32_formal
export OPENVLA_MAX_STEPS=${OPENVLA_MAX_STEPS:-150000}
export OPENVLA_SAVE_FREQ=${OPENVLA_SAVE_FREQ:-10000}
export OPENVLA_BATCH_SIZE="$batch"
export OPENVLA_GRAD_ACCUM="$accumulation"
export OPENVLA_NPROC=4
export OPENVLA_MERGE_LORA=True
exec "$TRAIN"
