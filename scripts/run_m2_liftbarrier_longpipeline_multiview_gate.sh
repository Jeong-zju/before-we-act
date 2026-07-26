#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${FE_ROOT}/.." && pwd)"
ROBOFACTORY_ROOT="${WORKSPACE}/RoboFactory"
CHECKPOINT="${FE_ROOT}/checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101"
CONFIG="${FE_ROOT}/configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml"
PORT=8872
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${FE_ROOT}/outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480_gate_${RUN_ID}"
SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT INT TERM

test -f "${CHECKPOINT}/schema.json"
test -f "${CHECKPOINT}/task_runtime.json"
test -f "${CONFIG}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

jq -e '
  .format_version == "wam.robofactory.m2.checkpoint/5" and
  .action_space == "per_task_zscore_canonical_unit_action" and
  .task_vocabulary == ["lift_barrier", "long_pipeline_delivery"] and
  .model_config.max_cameras == 5 and
  .model_config.visual_grid_height == 2 and
  .model_config.visual_grid_width == 3 and
  .vision_identity.input_height == 480 and
  .vision_identity.input_width == 640 and
  .action_generation == {
    "execution_steps": 2,
    "normalized_action_clip": 10.0,
    "solver": "euler",
    "solver_steps": 1,
    "warm_start": false
  }
' "${CHECKPOINT}/schema.json" >/dev/null

jq -e '
  length == 2 and
  .[0].task_id == "lift_barrier" and
  .[0].camera_order == ["global", "agent_0", "agent_1"] and
  .[0].camera_slot_indices == [0, 1, 2] and
  .[0].camera_agent_indices == [4, 0, 1] and
  .[1].task_id == "long_pipeline_delivery" and
  .[1].camera_order == [
    "global", "agent_0", "agent_1", "agent_2", "agent_3"
  ] and
  .[1].camera_slot_indices == [0, 1, 2, 3, 4] and
  .[1].camera_agent_indices == [4, 0, 1, 2, 3]
' "${CHECKPOINT}/task_runtime.json" >/dev/null

run_case() {
  local task="$1"
  local task_slug="$2"
  local max_steps="$3"
  local label="$4"
  local episodes="$5"
  local seed_start="$6"
  local output_dir="${OUTPUT_ROOT}/${task_slug}_${label}"

  (
    cd "${ROBOFACTORY_ROOT}"
    source ./activate_uv.sh
    python "${FE_ROOT}/scripts/serve_robofactory_m2_rollout.py" \
      --robofactory-root . \
      --task "${task}" \
      --scene table \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --episodes "${episodes}" \
      --seed-start "${seed_start}" \
      --max-steps "${max_steps}" \
      --sim-backend cpu \
      --shader default \
      --video-fps 20 \
      --output-dir "${output_dir}"
  ) &
  SERVER_PID=$!

  if ! (
    cd "${FE_ROOT}"
    CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache \
      uv run --frozen python scripts/run_robofactory_m2_inference.py \
      --checkpoint "${CHECKPOINT}" \
      --config "${CONFIG}" \
      --device cuda:0 \
      --precision bf16 \
      --host 127.0.0.1 \
      --port "${PORT}"
  ); then
    cleanup_server
    SERVER_PID=""
    return 1
  fi
  wait "${SERVER_PID}"
  SERVER_PID=""

  jq -e --argjson episodes "${episodes}" '
    .format_version == "wam.robofactory.m2.rollout_summary/2" and
    .completed == true and
    .fatal_error == null and
    .episodes_completed == $episodes and
    .direct_model_action_coverage == 1 and
    .engineering_smoke_passed == true
  ' "${output_dir}/rollout_summary.json" >/dev/null
}

for task_spec in \
  "LiftBarrier-rf|lift|500" \
  "LongPipelineDelivery-rf|long|1500"
do
  IFS='|' read -r task task_slug max_steps <<<"${task_spec}"
  run_case "${task}" "${task_slug}" "${max_steps}" train_seed3000 1 3000
  run_case "${task}" "${task_slug}" "${max_steps}" validation_seed3099 1 3099
  run_case "${task}" "${task_slug}" "${max_steps}" unseen_seed900_902 3 900
done

GATE_SUMMARY="${OUTPUT_ROOT}/gate_summary.json"
jq -n \
  --slurpfile lift_train "${OUTPUT_ROOT}/lift_train_seed3000/rollout_summary.json" \
  --slurpfile lift_validation "${OUTPUT_ROOT}/lift_validation_seed3099/rollout_summary.json" \
  --slurpfile lift_unseen "${OUTPUT_ROOT}/lift_unseen_seed900_902/rollout_summary.json" \
  --slurpfile long_train "${OUTPUT_ROOT}/long_train_seed3000/rollout_summary.json" \
  --slurpfile long_validation "${OUTPUT_ROOT}/long_validation_seed3099/rollout_summary.json" \
  --slurpfile long_unseen "${OUTPUT_ROOT}/long_unseen_seed900_902/rollout_summary.json" \
  '{
    format_version: "wam.robofactory.m2.multiview_joint_gate/1",
    checkpoint: $lift_train[0].client.checkpoint,
    lift_barrier: {
      train_seed3000: $lift_train[0].successes,
      validation_seed3099: $lift_validation[0].successes,
      unseen_seed900_902: $lift_unseen[0].successes,
      gate_passed: (
        $lift_train[0].successes == 1 and
        $lift_validation[0].successes == 1 and
        $lift_unseen[0].successes >= 2
      )
    },
    long_pipeline_delivery: {
      train_seed3000: $long_train[0].successes,
      validation_seed3099: $long_validation[0].successes,
      unseen_seed900_902: $long_unseen[0].successes,
      gate_passed: (
        $long_train[0].successes == 1 and
        $long_validation[0].successes == 1 and
        $long_unseen[0].successes >= 2
      )
    }
  }
  | .gate_passed = (
      .lift_barrier.gate_passed and
      .long_pipeline_delivery.gate_passed
    )
  ' >"${GATE_SUMMARY}"

jq . "${GATE_SUMMARY}"
jq -e '.gate_passed == true' "${GATE_SUMMARY}" >/dev/null
printf 'Joint multiview gate passed: %s\n' "${OUTPUT_ROOT}"
