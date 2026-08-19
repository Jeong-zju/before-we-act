#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_ATTRIBUTION_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_ATTRIBUTION_RUN_ROOT:-/workspace/bwa_runs/b-core/n3-minimal-attribution-v1}"
CONTRACT="${BWA_ATTRIBUTION_CONTRACT:-${ROOT}/docs/plans/contracts/b_core/n3_minimal_attribution/20260818/b_core_direct_closed_loop_contract.json}"
PYTHON="${BWA_ATTRIBUTION_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
SEED_ROOT="/workspace/bwa_runs/w10-six-task-v1/seeds/validation"
BCORE_ROOT="/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1/validation20"
DIRECT_ROOT="${RUN_ROOT}/validation20/direct_reactive"
STATUS="${RUN_ROOT}/status.json"
SUMMARY="${RUN_ROOT}/validation20/comparison.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
cd "${ROOT}"

write_status() {
  local status="$1" stage="$2" detail="$3"
  "${PYTHON}" - "${STATUS}" "${status}" "${stage}" "${detail}" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
value={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
tmp=p.with_name(f".{p.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
os.replace(tmp,p)
PY
}

PIDS=()
stop_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}
on_error() {
  local code=$?
  write_status FAILED error "minimal attribution exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'stop_children; write_status STOPPED interrupted "minimal attribution interrupted"; exit 130' INT TERM

mkdir -p "${RUN_ROOT}/logs" "${DIRECT_ROOT}"
[[ -f "${CONTRACT}" ]] || { echo "missing frozen contract" >&2; exit 2; }

B0H_CHECKPOINT="$(jq -r '.immutable_inputs.b0h_checkpoint.path' "${CONTRACT}")"
TRAINING_CHECKPOINT="$(jq -r '.immutable_inputs.training_checkpoint.path' "${CONTRACT}")"
for spec in b0h_checkpoint training_checkpoint bcore_checkpoint; do
  path="$(jq -r --arg spec "${spec}" '.immutable_inputs[$spec].path' "${CONTRACT}")"
  expected="$(jq -r --arg spec "${spec}" '.immutable_inputs[$spec].sha256' "${CONTRACT}")"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "immutable input drifted: ${spec}" >&2
    exit 2
  }
done
for path in \
  before_we_act/direct_reactive_policy.py \
  before_we_act/evaluate_direct_reactive.py \
  scripts/before_we_act/summarize_minimal_bcore_attribution.py \
  scripts/before_we_act/run_minimal_bcore_attribution.sh; do
  expected="$(jq -r --arg path "${path}" '.code_sha256[$path]' "${CONTRACT}")"
  [[ "$(sha256sum "${ROOT}/${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "frozen code drifted: ${path}" >&2
    exit 2
  }
done

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)
for task in "${TASKS[@]}"; do
  expected="$(jq -r --arg task "${task}" '.immutable_inputs.validation_seeds[$task]' "${CONTRACT}")"
  [[ -f "${SEED_ROOT}/${task}.json" && "$(sha256sum "${SEED_ROOT}/${task}.json" | awk '{print $1}')" == "${expected}" ]] || {
    echo "validation seed file drifted: ${task}" >&2
    exit 2
  }
  expected="$(jq -r --arg task "${task}" '.immutable_inputs.historical_bcore_results.task_sha256[$task]' "${CONTRACT}")"
  [[ -f "${BCORE_ROOT}/${task}.json" && "$(sha256sum "${BCORE_ROOT}/${task}.json" | awk '{print $1}')" == "${expected}" ]] || {
    echo "historical B-core result drifted: ${task}" >&2
    exit 2
  }
done

is_complete() {
  local output="$1" episodes="$2"
  [[ -f "${output}" ]] && "${PYTHON}" - "${output}" "${episodes}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
episodes=int(sys.argv[2])
ok=(value.get("mode")=="direct_reactive" and value.get("episodes")==episodes
    and len(value.get("rows",[]))==episodes)
raise SystemExit(0 if ok else 1)
PY
}

launch_task() {
  local task="$1" gpu="$2" episodes="$3" phase="$4"
  local output="${DIRECT_ROOT}/${task}.json"
  local log="${RUN_ROOT}/logs/validation20_direct_${task}.log"
  if is_complete "${output}" "${episodes}"; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_direct_reactive \
    --b0h-checkpoint "${B0H_CHECKPOINT}" \
    --training-checkpoint "${TRAINING_CHECKPOINT}" \
    --task "${task}" --seed-file "${SEED_ROOT}/${task}.json" \
    --episodes "${episodes}" --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
    --resume-log "${log}" --output "${output}" >>"${log}" 2>&1 &
  PIDS+=("$!")
  printf '[%s] phase=%s task=%s gpu=%s episodes=%s pid=%s\n' \
    "$(date -u +%FT%TZ)" "${phase}" "${task}" "${gpu}" "${episodes}" "$!"
}

run_wave() {
  local episodes="$1" phase="$2"
  shift 2
  PIDS=()
  local item task gpu pid code=0 child_code
  for item in "$@"; do
    task="${item%%:*}"; gpu="${item##*:}"
    launch_task "${task}" "${gpu}" "${episodes}" "${phase}"
  done
  for pid in "${PIDS[@]:-}"; do
    set +e
    wait "${pid}"
    child_code=$?
    set -e
    ((child_code == 0)) || code="${child_code}"
  done
  PIDS=()
  ((code == 0))
}

write_status RUNNING smoke "one frozen episode per task; results will be reused"
run_wave 1 smoke camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave 1 smoke pass_shoe:0 place_food:1
write_status RUNNING validation20 "running 120 paired direct-reactive episodes"
run_wave 20 validation20 camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave 20 validation20 pass_shoe:0 place_food:1

"${PYTHON}" scripts/before_we_act/summarize_minimal_bcore_attribution.py \
  --contract "${CONTRACT}" --bcore-root "${BCORE_ROOT}" \
  --direct-root "${DIRECT_ROOT}" --output "${SUMMARY}" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
DECISION="$(jq -r '.status' "${SUMMARY}")"
write_status COMPLETED decision "${DECISION}"
trap - ERR
printf 'B3_N3_MINIMAL_ATTRIBUTION_COMPLETED summary=%s decision=%s\n' "${SUMMARY}" "${DECISION}"
