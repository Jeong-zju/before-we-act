#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_N2_REPO_ROOT:-/workspace/fe-pc-wam-b-core}"
RUN_ROOT="${BWA_N2_RUN_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
SEED_ROOT="${BWA_N2_SEED_ROOT:-/workspace/bwa_runs/w10-six-task-v1/seeds/validation}"
W10_SUMMARY="${BWA_N2_W10_SUMMARY:-/workspace/bwa_runs/w10-six-task-v1/evaluation/validation/summary.json}"
B0H_SUMMARY="${BWA_N2_B0H_SUMMARY:-/workspace/bwa_runs/p1-step2-b0h-v7/hidden_residual/evaluation/validation20/summary.json}"
PYTHON="${BWA_N2_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONCLUSION="${RUN_ROOT}/conclusion.json"
OUTPUT_ROOT="${RUN_ROOT}/validation20"
LOG_ROOT="${RUN_ROOT}/logs"
SUMMARY="${OUTPUT_ROOT}/comparison.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

fail() {
  printf >&2 'B3-N2 Validation20: %s\n' "$*"
  exit 1
}

[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -f "${CONCLUSION}" ]] || fail "N2 conclusion is missing"
[[ "$(jq -r '.status' "${CONCLUSION}")" == POSITIVE_SIGNAL ]] || \
  fail "Validation20 requires POSITIVE_SIGNAL after frozen Validation5"
[[ "$(jq -r '.validation20_candidate.closed_loop_results_used_for_selection' "${CONCLUSION}")" == false ]] || \
  fail "candidate selection must not use closed-loop results"
[[ -f "${W10_SUMMARY}" && -f "${B0H_SUMMARY}" ]] || \
  fail "frozen W10/B0-H summaries are missing"

SEED="$(jq -r '.validation20_candidate.seed' "${CONCLUSION}")"
CHECKPOINT="$(jq -r '.validation20_candidate.deployment_checkpoint' "${CONCLUSION}")"
EXPECTED_SHA="$(jq -r '.validation20_candidate.deployment_checkpoint_sha256' "${CONCLUSION}")"
[[ "${SEED}" =~ ^2026081[567]$ ]] || fail "selected seed is outside the frozen set"
[[ -f "${CHECKPOINT}" ]] || fail "selected deployment checkpoint is missing"
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${ACTUAL_SHA}" == "${EXPECTED_SHA}" ]] || fail "selected checkpoint hash drifted"

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)
for task in "${TASKS[@]}"; do
  [[ -f "${SEED_ROOT}/${task}.json" ]] || fail "seed file is missing: ${task}"
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
CHILDREN=()

stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}
trap 'stop_children; exit 130' INT TERM

is_complete() {
  local output="$1"
  [[ -f "${output}" ]] && "${PYTHON}" - "${output}" "${EXPECTED_SHA}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
ok=(value.get("mode")=="n2" and value.get("episodes")==20
    and len(value.get("rows",[]))==20
    and value.get("checkpoint_sha256")==sys.argv[2])
raise SystemExit(0 if ok else 1)
PY
}

launch_task() {
  local task="$1" gpu="$2"
  local output="${OUTPUT_ROOT}/${task}.json"
  local log="${LOG_ROOT}/validation20_seed_${SEED}_${task}.log"
  if is_complete "${output}"; then
    printf '[%s] preserve N2 Validation20 task=%s\n' "$(date -u +%FT%TZ)" "${task}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_b3_n2 \
    --checkpoint "${CHECKPOINT}" --mode n2 --task "${task}" \
    --seed-file "${SEED_ROOT}/${task}.json" --episodes 20 \
    --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
    --resume-log "${log}" --output "${output}" >>"${log}" 2>&1 &
  CHILDREN+=("$!")
}

run_wave() {
  CHILDREN=()
  local item task gpu pid code=0 child_code
  for item in "$@"; do
    task="${item%%:*}"
    gpu="${item##*:}"
    launch_task "${task}" "${gpu}"
  done
  for pid in "${CHILDREN[@]:-}"; do
    set +e
    wait "${pid}"
    child_code=$?
    set -e
    ((child_code == 0)) || code="${child_code}"
  done
  CHILDREN=()
  ((code == 0)) || return "${code}"
}

run_wave camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave pass_shoe:0 place_food:1

"${PYTHON}" scripts/before_we_act/summarize_b3_n2_validation20.py \
  --conclusion "${CONCLUSION}" --validation-root "${OUTPUT_ROOT}" \
  --seed-root "${SEED_ROOT}" --w10-summary "${W10_SUMMARY}" \
  --b0h-summary "${B0H_SUMMARY}" --output "${SUMMARY}"
printf 'B3_N2_VALIDATION20_COMPLETED summary=%s\n' "${SUMMARY}"
