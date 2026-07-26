#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${FE_ROOT}/.." && pwd)"
ROBOFACTORY_ROOT="${WORKSPACE}/RoboFactory"
CHECKPOINT="${FE_ROOT}/checkpoints/phase_m2_liftbarrier_tailfixed_seed101"
CONFIG="${FE_ROOT}/configs/wam_multimodal/m2_liftbarrier_single.yaml"
PORT=8872
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${FE_ROOT}/outputs/phase_m2_liftbarrier_tailfixed_gate_${RUN_ID}"
SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT INT TERM

test -f "${CHECKPOINT}/schema.json"
test -f "${CONFIG}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

jq -e '
  .format_version == "wam.robofactory.m2.checkpoint/3" and
  .action_space == "per_task_zscore_canonical_unit_action" and
  .task_vocabulary == ["lift_barrier"] and
  .action_generation == {
    "execution_steps": 2,
    "normalized_action_clip": 10.0,
    "solver": "euler",
    "solver_steps": 1,
    "warm_start": false
  } and
  .action_objective == {
    "executed_prefix_weight": 4.0,
    "tail_windows": "repeat_last_with_validity_masks"
  }
' "${CHECKPOINT}/schema.json" >/dev/null

run_case() {
  local label="$1"
  local episodes="$2"
  local seed_start="$3"
  local output_dir="${OUTPUT_ROOT}/${label}"

  (
    cd "${ROBOFACTORY_ROOT}"
    source ./activate_uv.sh
    python "${FE_ROOT}/scripts/serve_robofactory_m2_rollout.py" \
      --robofactory-root . \
      --task LiftBarrier-rf \
      --scene table \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --episodes "${episodes}" \
      --seed-start "${seed_start}" \
      --max-steps 500 \
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
    .completed == true and
    .fatal_error == null and
    .episodes_completed == $episodes and
    .direct_model_action_coverage == 1
  ' "${output_dir}/rollout_summary.json" >/dev/null
}

run_case train_seed3000 1 3000
run_case validation_seed3003 1 3003
run_case unseen_seed900_902 3 900

GATE_SUMMARY="${OUTPUT_ROOT}/gate_summary.json"
jq -n \
  --slurpfile train "${OUTPUT_ROOT}/train_seed3000/rollout_summary.json" \
  --slurpfile validation "${OUTPUT_ROOT}/validation_seed3003/rollout_summary.json" \
  --slurpfile unseen "${OUTPUT_ROOT}/unseen_seed900_902/rollout_summary.json" \
  '{
    format_version: "wam.robofactory.m2.liftbarrier_gate/1",
    checkpoint: $train[0].client.checkpoint,
    train_seed3000: {
      successes: $train[0].successes,
      episodes: $train[0].episodes_completed
    },
    validation_seed3003: {
      successes: $validation[0].successes,
      episodes: $validation[0].episodes_completed
    },
    unseen_seed900_902: {
      successes: $unseen[0].successes,
      episodes: $unseen[0].episodes_completed
    },
    gate_passed: (
      $train[0].successes == 1 and
      $validation[0].successes == 1 and
      $unseen[0].successes >= 2
    )
  }' >"${GATE_SUMMARY}"

jq -e '.gate_passed == true' "${GATE_SUMMARY}" >/dev/null
mkdir -p "${FE_ROOT}/outputs/phase_m2_liftbarrier_tailfixed"
cp "${GATE_SUMMARY}" \
  "${FE_ROOT}/outputs/phase_m2_liftbarrier_tailfixed/gate_summary.json"
jq . "${GATE_SUMMARY}"
printf 'LiftBarrier gate output: %s\n' "${OUTPUT_ROOT}"
