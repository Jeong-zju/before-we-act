#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${FE_ROOT}/datasets/archive/pre_multiview_${RUN_ID}"
LIFT_SOURCE="${FE_ROOT}/../RoboFactory/data/m2_raw/LiftBarrier-rf/motionplanning/LiftBarrier-rf_m2_multiview_150"
LONG_SOURCE="${FE_ROOT}/../RoboFactory/data/m2_raw/LongPipelineDelivery-rf/motionplanning/LongPipelineDelivery-rf_m2_multiview_150"
REQUESTED_WORKERS="${M2_CONVERSION_NUM_WORKERS:-auto}"
MAX_WORKERS="${M2_CONVERSION_MAX_WORKERS:-16}"
MEMORY_FRACTION="${M2_CONVERSION_MEMORY_FRACTION:-0.75}"
LIFT_WORKER_MEMORY_MIB="${M2_LIFT_CONVERSION_WORKER_MEMORY_MIB:-4096}"
LONG_WORKER_MEMORY_MIB="${M2_LONG_CONVERSION_WORKER_MEMORY_MIB:-6144}"
export M2_CONVERSION_THREADS_PER_WORKER="${M2_CONVERSION_THREADS_PER_WORKER:-1}"

test -f "${LIFT_SOURCE}.h5"
test -f "${LIFT_SOURCE}.json"
test -f "${LONG_SOURCE}.h5"
test -f "${LONG_SOURCE}.json"

cd "${FE_ROOT}"

validate_source_resolution() {
  local source_h5="$1"
  shift
  UV_CACHE_DIR=.uv-cache uv run --frozen python - "${source_h5}" "$@" <<'PY'
import sys

import h5py

path, *cameras = sys.argv[1:]
with h5py.File(path, "r") as stream:
    trajectories = [name for name in stream if name.startswith("traj_")]
    if len(trajectories) != 150:
        raise ValueError(f"{path}: expected 150 trajectories, got {len(trajectories)}")
    for trajectory_name in trajectories:
        trajectory = stream[trajectory_name]
        for camera in cameras:
            rgb = trajectory[f"obs/sensor_data/{camera}/rgb"]
            if tuple(rgb.shape[-3:]) != (480, 640, 3) or rgb.dtype.name != "uint8":
                raise ValueError(
                    f"{path}:{trajectory_name}:{camera} is "
                    f"{tuple(rgb.shape[-3:])}/{rgb.dtype}; "
                    "recollect native 480x640 uint8 sensor RGB first"
                )
print(f"validated native 480x640 RGB: {path}")
PY
}

validate_source_resolution \
  "${LIFT_SOURCE}.h5" \
  head_camera_global head_camera_agent0 head_camera_agent1
validate_source_resolution \
  "${LONG_SOURCE}.h5" \
  head_camera_global head_camera_agent0 head_camera_agent1 \
  head_camera_agent2 head_camera_agent3

mkdir -p "${ARCHIVE}"
if [[ -e datasets/robofactory_multitask/lift_barrier ]]; then
  mv datasets/robofactory_multitask/lift_barrier "${ARCHIVE}/"
fi
if [[ -e datasets/robofactory_multitask/long_pipeline_delivery ]]; then
  mv datasets/robofactory_multitask/long_pipeline_delivery "${ARCHIVE}/"
fi

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/convert_robofactory_dataset.py \
  --input "${LIFT_SOURCE}.h5" \
  --metadata-json "${LIFT_SOURCE}.json" \
  --out-dir datasets/robofactory_multitask/lift_barrier \
  --profile m1-scratch \
  --format hdf5 \
  --fps 20 \
  --task "Lift the barrier together" \
  --task-id lift_barrier \
  --camera global \
  --camera agent_0 \
  --camera agent_1 \
  --executed-action-source command-echo \
  --episodes 150 \
  --success-only \
  --num-workers "${M2_LIFT_CONVERSION_NUM_WORKERS:-${REQUESTED_WORKERS}}" \
  --max-workers "${MAX_WORKERS}" \
  --worker-memory-mib "${LIFT_WORKER_MEMORY_MIB}" \
  --memory-fraction "${MEMORY_FRACTION}" \
  --compression gzip

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_multitask/lift_barrier \
  --transition-selection through-first-done-inclusive \
  --split-seed 7 \
  --expected-episodes 150 \
  --expected-state-dim 36 \
  --expected-action-dim 16 \
  --expected-task-id lift_barrier \
  --expected-camera global \
  --expected-camera agent_0 \
  --expected-camera agent_1 \
  --expected-fps 20 \
  --action-codec configs/action_codecs/robofactory_2panda_pd_joint_pos_16d.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/convert_robofactory_dataset.py \
  --input "${LONG_SOURCE}.h5" \
  --metadata-json "${LONG_SOURCE}.json" \
  --out-dir datasets/robofactory_multitask/long_pipeline_delivery \
  --profile m1-scratch \
  --format hdf5 \
  --fps 20 \
  --task "Deliver the long pipeline together" \
  --task-id long_pipeline_delivery \
  --camera global \
  --camera agent_0 \
  --camera agent_1 \
  --camera agent_2 \
  --camera agent_3 \
  --executed-action-source command-echo \
  --episodes 150 \
  --success-only \
  --num-workers "${M2_LONG_CONVERSION_NUM_WORKERS:-${REQUESTED_WORKERS}}" \
  --max-workers "${MAX_WORKERS}" \
  --worker-memory-mib "${LONG_WORKER_MEMORY_MIB}" \
  --memory-fraction "${MEMORY_FRACTION}" \
  --compression gzip

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_multitask/long_pipeline_delivery \
  --transition-selection through-first-done-inclusive \
  --split-seed 7 \
  --expected-episodes 150 \
  --expected-state-dim 72 \
  --expected-action-dim 32 \
  --expected-task-id long_pipeline_delivery \
  --expected-camera global \
  --expected-camera agent_0 \
  --expected-camera agent_1 \
  --expected-camera agent_2 \
  --expected-camera agent_3 \
  --expected-fps 20 \
  --action-codec configs/action_codecs/robofactory_4panda_pd_joint_pos_32d.json

printf 'Archived previous converted data under: %s\n' "${ARCHIVE}"
