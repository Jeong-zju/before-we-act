#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; printf >&2 "Collection script failed at line %d: %s (exit %d)\n" "${LINENO}" "${BASH_COMMAND}" "${status}"; exit "${status}"' ERR

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOFACTORY_ROOT="$(cd "${FE_ROOT}/../RoboFactory" && pwd)"

cd "${ROBOFACTORY_ROOT}"
source ./activate_uv.sh

EXPECTED_TRAJECTORIES=150
EXISTING_MODE="${M2_COLLECTION_EXISTING:-reuse}"
ARCHIVE_STAMP="$(date +%Y%m%d_%H%M%S)_$$"

case "${EXISTING_MODE}" in
  reuse|archive|error) ;;
  *)
    printf >&2 \
      'Invalid M2_COLLECTION_EXISTING=%q; expected reuse, archive, or error.\n' \
      "${EXISTING_MODE}"
    exit 2
    ;;
esac

episode_count() {
  local metadata_path="$1"
  python - "${metadata_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if not isinstance(episodes, list):
        raise TypeError("'episodes' is not a list")
except (OSError, KeyError, TypeError, ValueError):
    print(-1)
else:
    print(len(episodes))
PY
}

collection_contract_ok() {
  local h5_path="$1"
  local camera_csv="$2"
  python - "${h5_path}" "${camera_csv}" <<'PY'
import sys

import h5py

path, camera_csv = sys.argv[1:]
cameras = camera_csv.split(",")
try:
    with h5py.File(path, "r") as stream:
        trajectory_names = sorted(
            (name for name in stream if name.startswith("traj_")),
            key=lambda value: int(value.split("_", 1)[1]),
        )
        if not trajectory_names:
            raise ValueError("no trajectories")
        trajectory = stream[trajectory_names[0]]
        for camera in cameras:
            rgb = trajectory[f"obs/sensor_data/{camera}/rgb"]
            if tuple(rgb.shape[-3:]) != (480, 640, 3) or rgb.dtype.name != "uint8":
                raise ValueError(
                    f"{camera} is {tuple(rgb.shape[-3:])}/{rgb.dtype}, "
                    "expected (480,640,3)/uint8"
                )
except (KeyError, OSError, TypeError, ValueError):
    print("false")
else:
    print("true")
PY
}

archive_task_outputs() {
  local task="$1"
  local prefix="$2"
  local archive_dir="data/m2_raw/_collection_archive/${ARCHIVE_STAMP}/${task}/motionplanning"
  local candidate
  local -a candidates=()

  shopt -s nullglob
  candidates=("${prefix}".*)
  shopt -u nullglob
  if ((${#candidates[@]} == 0)); then
    return
  fi

  mkdir -p "${archive_dir}"
  for candidate in "${candidates[@]}"; do
    mv -- "${candidate}" "${archive_dir}/"
  done
  printf 'Archived existing %s artifacts to %s\n' "${task}" "${archive_dir}"
}

prepare_task() {
  local task="$1"
  local prefix="$2"
  local result_var="$3"
  local archive_var="$4"
  local camera_csv="$5"
  local h5_path="${prefix}.h5"
  local json_path="${prefix}.json"
  local count=-1
  local state="missing"

  if [[ -e "${json_path}" ]]; then
    count="$(episode_count "${json_path}")"
  fi
  local contract_ok="false"
  if [[ -e "${h5_path}" ]]; then
    contract_ok="$(collection_contract_ok "${h5_path}" "${camera_csv}")"
  fi
  if [[ \
    -e "${h5_path}" \
    && -e "${json_path}" \
    && "${count}" -eq "${EXPECTED_TRAJECTORIES}" \
    && "${contract_ok}" == "true" \
  ]]; then
    state="complete"
  elif [[ -e "${h5_path}" || -e "${json_path}" ]]; then
    state="incomplete"
  fi

  case "${EXISTING_MODE}:${state}" in
    reuse:complete)
      printf 'Reusing complete %s collection (%d trajectories): %s.{h5,json}\n' \
        "${task}" "${count}" "${prefix}"
      printf -v "${result_var}" '%s' "false"
      printf -v "${archive_var}" '%s' "false"
      ;;
    reuse:incomplete)
      printf 'Found incomplete or non-640x480 %s collection (%d/%d trajectories); it will be archived after the render-device preflight.\n' \
        "${task}" "${count}" "${EXPECTED_TRAJECTORIES}"
      printf -v "${result_var}" '%s' "true"
      printf -v "${archive_var}" '%s' "true"
      ;;
    archive:complete|archive:incomplete)
      printf -v "${result_var}" '%s' "true"
      printf -v "${archive_var}" '%s' "true"
      ;;
    error:complete|error:incomplete)
      printf >&2 \
        'Existing %s artifacts found at %s.{h5,json}; use M2_COLLECTION_EXISTING=reuse or archive.\n' \
        "${task}" "${prefix}"
      exit 2
      ;;
    *:missing)
      printf -v "${result_var}" '%s' "true"
      printf -v "${archive_var}" '%s' "false"
      ;;
  esac
}

preflight_render_device() {
  local nvidia_output
  local sapien_output

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf >&2 \
      'NVIDIA render-device preflight failed: nvidia-smi is not available. RGB collection requires a working NVIDIA/Vulkan render device.\n'
    exit 3
  fi
  if ! nvidia_output="$(nvidia-smi -L 2>&1)"; then
    printf >&2 'NVIDIA render-device preflight failed:\n%s\n' "${nvidia_output}"
    if [[ "${nvidia_output}" == *"Driver/library version mismatch"* ]]; then
      printf >&2 \
        'The loaded NVIDIA kernel module and user-space libraries have different versions. Reboot the host, then rerun this script.\n'
    else
      printf >&2 \
        'Restore access to the NVIDIA driver/Vulkan device, then rerun this script.\n'
    fi
    exit 3
  fi

  if ! sapien_output="$(
    python - <<'PY' 2>&1
import sapien

device = sapien.Device("cuda:0")
render_system = sapien.render.RenderSystem(device)
del render_system
print("SAPIEN cuda:0 RenderSystem: OK")
PY
  )"; then
    printf >&2 'SAPIEN render-device preflight failed:\n%s\n' "${sapien_output}"
    printf >&2 \
      'Check the NVIDIA driver and Vulkan installation before rerunning collection.\n'
    exit 3
  fi
  printf '%s\n' "${sapien_output}"
}

LIFT_PREFIX="data/m2_raw/LiftBarrier-rf/motionplanning/LiftBarrier-rf_m2_multiview_150"
LONG_PREFIX="data/m2_raw/LongPipelineDelivery-rf/motionplanning/LongPipelineDelivery-rf_m2_multiview_150"
prepare_task \
  "LiftBarrier-rf" "${LIFT_PREFIX}" LIFT_SHOULD_COLLECT LIFT_SHOULD_ARCHIVE \
  "head_camera_global,head_camera_agent0,head_camera_agent1"
prepare_task \
  "LongPipelineDelivery-rf" "${LONG_PREFIX}" LONG_SHOULD_COLLECT LONG_SHOULD_ARCHIVE \
  "head_camera_global,head_camera_agent0,head_camera_agent1,head_camera_agent2,head_camera_agent3"

if [[ "${LIFT_SHOULD_COLLECT}" == "true" || "${LONG_SHOULD_COLLECT}" == "true" ]]; then
  preflight_render_device
fi
if [[ "${LIFT_SHOULD_ARCHIVE}" == "true" ]]; then
  archive_task_outputs "LiftBarrier-rf" "${LIFT_PREFIX}"
fi
if [[ "${LONG_SHOULD_ARCHIVE}" == "true" ]]; then
  archive_task_outputs "LongPipelineDelivery-rf" "${LONG_PREFIX}"
fi

REQUESTED_PROCS="${M2_COLLECTION_NUM_PROCS:-auto}"
MAX_PROCS="${M2_COLLECTION_MAX_PROCS:-16}"
CPU_THREADS_TARGET="${M2_COLLECTION_CPU_THREADS_PER_WORKER:-2}"
MEMORY_FRACTION="${M2_COLLECTION_MEMORY_FRACTION:-0.8}"

select_workers() {
  local task="$1"
  local requested="$2"
  python "${FE_ROOT}/scripts/select_robofactory_collection_workers.py" \
    --task "${task}" \
    --trajectories 150 \
    --requested "${requested}" \
    --max-workers "${MAX_PROCS}" \
    --cpu-threads-per-worker "${CPU_THREADS_TARGET}" \
    --memory-fraction "${MEMORY_FRACTION}"
}

if [[ "${LIFT_SHOULD_COLLECT}" == "true" ]]; then
  read -r LIFT_PROCS LIFT_THREADS < <(
    select_workers \
      lift_barrier \
      "${M2_LIFT_COLLECTION_NUM_PROCS:-${REQUESTED_PROCS}}"
  )
  LIFT_MAX_ATTEMPTS=$(((1500 + LIFT_PROCS - 1) / LIFT_PROCS))

  printf 'Collecting LiftBarrier with %d disjoint seed workers.\n' "${LIFT_PROCS}"
  OMP_NUM_THREADS="${LIFT_THREADS}" \
  MKL_NUM_THREADS="${LIFT_THREADS}" \
  OPENBLAS_NUM_THREADS="${LIFT_THREADS}" \
  NUMEXPR_NUM_THREADS="${LIFT_THREADS}" \
  python -m robofactory.script.generate_data \
    --scene table \
    --task LiftBarrier-rf \
    --num "${EXPECTED_TRAJECTORIES}" \
    --seed 3000 \
    --max-attempts "${LIFT_MAX_ATTEMPTS}" \
    --num-procs "${LIFT_PROCS}" \
    --record-dir data/m2_raw \
    --traj-name LiftBarrier-rf_m2_multiview_150
fi

if [[ "${LONG_SHOULD_COLLECT}" == "true" ]]; then
  read -r LONG_PROCS LONG_THREADS < <(
    select_workers \
      long_pipeline_delivery \
      "${M2_LONG_COLLECTION_NUM_PROCS:-${REQUESTED_PROCS}}"
  )
  LONG_MAX_ATTEMPTS=$(((1500 + LONG_PROCS - 1) / LONG_PROCS))

  printf 'Collecting LongPipelineDelivery with %d disjoint seed workers.\n' \
    "${LONG_PROCS}"
  OMP_NUM_THREADS="${LONG_THREADS}" \
  MKL_NUM_THREADS="${LONG_THREADS}" \
  OPENBLAS_NUM_THREADS="${LONG_THREADS}" \
  NUMEXPR_NUM_THREADS="${LONG_THREADS}" \
  python -m robofactory.script.generate_data \
    --scene table \
    --task LongPipelineDelivery-rf \
    --num "${EXPECTED_TRAJECTORIES}" \
    --seed 3000 \
    --max-attempts "${LONG_MAX_ATTEMPTS}" \
    --num-procs "${LONG_PROCS}" \
    --record-dir data/m2_raw \
    --traj-name LongPipelineDelivery-rf_m2_multiview_150
fi

printf 'Per-task seed-parallel collection completed successfully.\n'
