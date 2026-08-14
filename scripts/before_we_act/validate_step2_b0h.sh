#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${STEP2_RUN_ROOT:?STEP2_RUN_ROOT is required}"
CHECKPOINT="${STEP2_CHECKPOINT:?STEP2_CHECKPOINT is required}"
EPISODES="${STEP2_VALIDATION_EPISODES:?STEP2_VALIDATION_EPISODES is required}"
SEED_ROOT="${STEP2_SEED_ROOT:-/workspace/bwa_runs/w10-six-task-v1/seeds/validation}"
PYTHON_BIN="${STEP2_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${STEP2_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
EVAL_ROOT="${RUN_ROOT}/evaluation/validation${EPISODES}"
LOG_ROOT="${RUN_ROOT}/logs"
STATUS="${RUN_ROOT}/validation${EPISODES}_status.json"
EVALUATOR="${ROOT}/before_we_act/evaluate_step2_b0h.py"

fail() {
  printf >&2 'Step-2 B0-H validation: %s\n' "$*"
  exit 1
}

[[ "${EPISODES}" == 5 || "${EPISODES}" == 20 ]] || fail "episodes must be 5 or 20"
[[ -x "${PYTHON_BIN}" ]] || fail "Python is missing: ${PYTHON_BIN}"
[[ -f "${CHECKPOINT}" ]] || fail "checkpoint is missing: ${CHECKPOINT}"
[[ -f "${EVALUATOR}" ]] || fail "evaluator is missing: ${EVALUATOR}"
[[ -d "${ROBOFACTORY_ROOT}" ]] || fail "RoboFactory root is missing"

TASKS=(
  lift_barrier
  camera_alignment
  long_pipeline_delivery
  take_photo
  pass_shoe
  place_food
)
declare -A MAX_STEPS=(
  [lift_barrier]=500
  [camera_alignment]=1500
  [long_pipeline_delivery]=1500
  [take_photo]=1500
  [pass_shoe]=500
  [place_food]=500
)
for task in "${TASKS[@]}"; do
  [[ -f "${SEED_ROOT}/${task}.json" ]] || fail "missing seed file: ${task}"
done

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}"
CHILDREN=()

write_status() {
  local status="$1" wave="$2" detail="$3"
  "${PYTHON_BIN}" - "${STATUS}" "${status}" "${wave}" "${detail}" "${EPISODES}" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1])
value={"status":sys.argv[2],"stage":"validation","wave":sys.argv[3],
       "detail":sys.argv[4],"episodes_per_task":int(sys.argv[5]),
       "updated_at_epoch":time.time()}
tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,sort_keys=True)+"\n")
os.replace(tmp,path)
PY
}

stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}

on_error() {
  local code=$?
  write_status FAILED error "validation command exited with code ${code}" || true
  stop_children
  exit "${code}"
}
trap on_error ERR
trap 'write_status STOPPED interrupted "signal"; stop_children; exit 130' INT TERM

is_complete() {
  local output="$1"
  [[ -f "${output}" ]] && "${PYTHON_BIN}" - "${output}" "${EPISODES}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
n=int(sys.argv[2])
raise SystemExit(0 if value.get("episodes")==n and len(value.get("rows",[]))==n else 1)
PY
}

launch_task() {
  local task="$1" gpu="$2"
  local output="${EVAL_ROOT}/${task}.json"
  local log="${LOG_ROOT}/validation${EPISODES}_${task}.log"
  if is_complete "${output}"; then
    printf '[%s] preserve completed task=%s episodes=%s\n' "$(date -u +%FT%TZ)" "${task}" "${EPISODES}"
    return
  fi
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/vendor/stereo-core/stereo_core:${ROOT}:${ROBOFACTORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -u "${EVALUATOR}" \
      --checkpoint "${CHECKPOINT}" \
      --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" \
      --episodes "${EPISODES}" \
      --max-steps "${MAX_STEPS[$task]}" \
      --device cuda:0 \
      --resume-log "${log}" \
      --output "${output}" >>"${log}" 2>&1 &
  CHILDREN+=("$!")
}

run_wave() {
  local wave="$1"
  shift
  CHILDREN=()
  write_status RUNNING "${wave}" "launching $*"
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

run_wave wave1 camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave wave2 pass_shoe:0 place_food:1

"${PYTHON_BIN}" - "${EVAL_ROOT}" "${CHECKPOINT}" "${EPISODES}" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); checkpoint=Path(sys.argv[2]); episodes=int(sys.argv[3])
tasks=("lift_barrier","camera_alignment","long_pipeline_delivery","take_photo","pass_shoe","place_food")
results={task:json.loads((root/f"{task}.json").read_text()) for task in tasks}
counts={task:int(results[task]["successes"]) for task in tasks}
total=sum(counts.values())
acceptance=None
if episodes==20:
    stable=("lift_barrier","long_pipeline_delivery","take_photo","pass_shoe")
    checks={
      "total_ge_80":total>=80,
      "stable_sum_ge_72":sum(counts[t] for t in stable)>=72,
      "stable_each_ge_16":all(counts[t]>=16 for t in stable),
      "camera_ge_6":counts["camera_alignment"]>=6,
      "camera_plus_food_ge_8":counts["camera_alignment"]+counts["place_food"]>=8,
    }
    acceptance={"status":"PASSED" if all(checks.values()) else "FAILED",
                "checks":checks,"matches_w10_88":total>=88}
payload={
  "status":"PASSED","stage":"validation","checkpoint":str(checkpoint),
  "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
  "episodes":sum(int(results[t]["episodes"]) for t in tasks),"successes":total,
  "tasks":{t:{"episodes":int(results[t]["episodes"]),"successes":counts[t],
              "success_rate":float(results[t]["success_rate"])} for t in tasks},
  "acceptance":acceptance,"completed_at_epoch":time.time(),
}
(root/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,sort_keys=True))
PY
write_status PASSED complete "all validation episodes completed"
trap - ERR
