#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; printf >&2 "M2 training script failed at line %d: %s (exit %d)\n" "${LINENO}" "${BASH_COMMAND}" "${status}"; exit "${status}"' ERR

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${FE_ROOT}/checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101"
RESUME="${FE_ROOT}/checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101_resume"
OUTPUT="${FE_ROOT}/outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480"
REPORT="${OUTPUT}/seed101_training.json"
LOG="${OUTPUT}/seed101_training.log"
PROGRESS_LOG="${OUTPUT}/seed101_progress.jsonl"
VALIDATION="${OUTPUT}/teacher_context_validation.json"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESET="${M2_TRAINING_RESET:-0}"

test -f "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
test -f "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"

mkdir -p "${FE_ROOT}/checkpoints/archive" "${FE_ROOT}/outputs/archive"
case "${RESET}" in
  0|1) ;;
  *)
    printf >&2 'M2_TRAINING_RESET must be 0 or 1, got %q.\n' "${RESET}"
    exit 2
    ;;
esac
if [[ "${RESET}" == "1" ]]; then
  if [[ -e "${CHECKPOINT}" ]]; then
    mv "${CHECKPOINT}" \
      "${FE_ROOT}/checkpoints/archive/phase_m2_multiview_640x480_seed101_${RUN_ID}"
  fi
  if [[ -e "${RESUME}" ]]; then
    mv "${RESUME}" \
      "${FE_ROOT}/checkpoints/archive/phase_m2_multiview_640x480_seed101_resume_${RUN_ID}"
  fi
  if [[ -e "${OUTPUT}" ]]; then
    mv "${OUTPUT}" \
      "${FE_ROOT}/outputs/archive/phase_m2_multiview_640x480_${RUN_ID}"
  fi
fi
mkdir -p "${OUTPUT}"

if [[ -e "${CHECKPOINT}" || -e "${REPORT}" ]]; then
  test -f "${CHECKPOINT}/schema.json"
  test -f "${REPORT}"
  printf 'Reusing completed M2 training artifacts: %s\n' "${CHECKPOINT}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L; then
    printf >&2 \
      'CUDA preflight failed; restore the NVIDIA driver before training. Existing logs and resume snapshots were left untouched.\n'
    exit 3
  fi
  cd "${FE_ROOT}"
  printf '{"event":"launcher_started","recorded_at":"%s","log":"%s","progress_log":"%s"}\n' \
    "$(date --iso-8601=seconds)" "${LOG}" "${PROGRESS_LOG}" >>"${LOG}"
  script \
    --quiet \
    --return \
    --flush \
    --append \
    --command \
    'CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/train_robofactory_m2.py --config configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml --device cuda:0 --torch-threads 8 --seed 101' \
    "${LOG}"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 UV_CACHE_DIR=.uv-cache \
  uv run --frozen python scripts/evaluate_robofactory_m2_teacher_context.py \
    --config configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml \
    --checkpoint "${CHECKPOINT}" \
    --output "${VALIDATION}" \
    --split validation \
    --samples-per-task 64 \
    --batch-size 2 \
    --device cuda:0 \
    --precision bf16

jq -e '
  .format_version == "wam.robofactory.m2.training_report/5" and
  .passed == true and
  .strict_reload_max_abs_difference == 0 and
  .dataset.dataset_protocol == "wam.robofactory.multitask/3" and
  .dataset.camera_order == [
    "global", "agent_0", "agent_1", "agent_2", "agent_3"
  ] and
  .dataset.image_shape_hwc == [480, 640, 3] and
  .model.config.max_cameras == 5 and
  .model.config.visual_grid_height == 2 and
  .model.config.visual_grid_width == 3
' "${REPORT}" >/dev/null

jq -e '
  .format_version == "wam.robofactory.m2.checkpoint/5" and
  .vision_identity.input_height == 480 and
  .vision_identity.input_width == 640
' "${CHECKPOINT}/schema.json" >/dev/null

test "$(jq '[.tasks[].agents[] | select(.active_scalar_count > 0)] | length' "${VALIDATION}")" -eq 6
jq -e '
  .format_version == "wam.robofactory.m2.teacher_context_validation/1" and
  .passed == true and
  .spatial_visual_grid == [2, 3]
' "${VALIDATION}" >/dev/null

printf 'M2 multiview training artifact passed: %s\n' "${CHECKPOINT}"
