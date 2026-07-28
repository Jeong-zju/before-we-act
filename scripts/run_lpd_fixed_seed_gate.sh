#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${FE_ROOT}/.." && pwd)"
ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT:-${WORKSPACE}/RoboFactory}"
RF_PYTHON="${RF_PYTHON:-${ROBOFACTORY_ROOT}/.venv/bin/python}"
LPD_CONFIG="${LPD_CONFIG:?set LPD_CONFIG}"
LPD_CHECKPOINT="${LPD_CHECKPOINT:?set LPD_CHECKPOINT}"
LPD_POLICY_KIND="${LPD_POLICY_KIND:?set LPD_POLICY_KIND to wam or static_act}"
LPD_GATE_MODE="${LPD_GATE_MODE:-gate}"
LPD_PORT="${LPD_PORT:-8872}"
LPD_RUN_ID="${LPD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SERVER_PID=""

case "${LPD_GATE_MODE}" in
  gate)
    EPISODES="${LPD_EPISODES:-20}"
    SEED_START="${LPD_SEED_START:-900}"
    ;;
  formal)
    EPISODES="${LPD_EPISODES:-100}"
    SEED_START="${LPD_SEED_START:-920}"
    ;;
  *)
    printf >&2 'LPD_GATE_MODE must be gate or formal.\n'
    exit 2
    ;;
esac

OUTPUT_ROOT="${LPD_OUTPUT_ROOT:-${FE_ROOT}/outputs/${LPD_EXPERIMENT_SLUG:?set LPD_EXPERIMENT_SLUG}/${LPD_GATE_MODE}_${LPD_RUN_ID}}"

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT INT TERM

test -x "${RF_PYTHON}"
test -f "${LPD_CONFIG}"
test -e "${LPD_CHECKPOINT}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_case() {
  local task="$1"
  local slug="$2"
  local max_steps="$3"
  local output="${OUTPUT_ROOT}/${slug}"
  (
    cd "${ROBOFACTORY_ROOT}"
    PYTHONPATH="${ROBOFACTORY_ROOT}" "${RF_PYTHON}" \
      "${FE_ROOT}/scripts/serve_robofactory_m2_rollout.py" \
      --robofactory-root "${ROBOFACTORY_ROOT}" \
      --task "${task}" \
      --scene table \
      --host 127.0.0.1 \
      --port "${LPD_PORT}" \
      --episodes "${EPISODES}" \
      --seed-start "${SEED_START}" \
      --max-steps "${max_steps}" \
      --sim-backend cpu \
      --shader default \
      --video-fps 20 \
      --output-dir "${output}"
  ) &
  SERVER_PID=$!

  case "${LPD_POLICY_KIND}" in
    wam)
      (
        cd "${FE_ROOT}"
        uv run --frozen python scripts/run_robofactory_m2_inference.py \
          --checkpoint "${LPD_CHECKPOINT}" \
          --config "${LPD_CONFIG}" \
          --device cuda:0 \
          --precision bf16 \
          --host 127.0.0.1 \
          --port "${LPD_PORT}"
      )
      ;;
    static_act)
      (
        cd "${FE_ROOT}"
        uv run --frozen python scripts/run_static_rgb_act_moe_inference.py \
          --checkpoint "${LPD_CHECKPOINT}" \
          --config "${LPD_CONFIG}" \
          --device cuda:0 \
          --host 127.0.0.1 \
          --port "${LPD_PORT}"
      )
      ;;
    *)
      printf >&2 'unknown LPD_POLICY_KIND=%q\n' "${LPD_POLICY_KIND}"
      return 2
      ;;
  esac
  wait "${SERVER_PID}"
  SERVER_PID=""
  jq -e --argjson episodes "${EPISODES}" '
    .completed == true and
    .fatal_error == null and
    .episodes_completed == $episodes and
    .direct_model_action_coverage == 1
  ' "${output}/rollout_summary.json" >/dev/null
}

run_case LiftBarrier-rf lift_barrier 500
run_case LongPipelineDelivery-rf long_pipeline_delivery 1500

SUMMARY="${OUTPUT_ROOT}/gate_summary.json"
jq -n \
  --arg mode "${LPD_GATE_MODE}" \
  --arg experiment "${LPD_EXPERIMENT_SLUG}" \
  --arg config "${LPD_CONFIG}" \
  --arg checkpoint "${LPD_CHECKPOINT}" \
  --argjson seed_start "${SEED_START}" \
  --argjson episodes "${EPISODES}" \
  --slurpfile lift "${OUTPUT_ROOT}/lift_barrier/rollout_summary.json" \
  --slurpfile lpd "${OUTPUT_ROOT}/long_pipeline_delivery/rollout_summary.json" \
  '{
    format_version: "wam.robofactory.lpd_fixed_seed_gate/1",
    mode: $mode,
    experiment: $experiment,
    config: $config,
    checkpoint: $checkpoint,
    seed_protocol: {
      seed_start: $seed_start,
      episodes_per_task: $episodes,
      identical_across_tasks: true
    },
    lift_barrier: {
      successes: $lift[0].successes,
      success_rate: $lift[0].success_rate
    },
    long_pipeline_delivery: {
      successes: $lpd[0].successes,
      success_rate: $lpd[0].success_rate
    }
  }
  | .passed = (
      if $mode == "formal"
      then (
        .lift_barrier.success_rate >= 0.90 and
        .long_pipeline_delivery.success_rate >= 0.90
      )
      else (
        .lift_barrier.successes >= 1 and
        .long_pipeline_delivery.successes >= 1
      )
      end
    )
  ' >"${SUMMARY}"

jq . "${SUMMARY}"
printf 'Fixed-seed %s complete: %s\n' "${LPD_GATE_MODE}" "${OUTPUT_ROOT}"
