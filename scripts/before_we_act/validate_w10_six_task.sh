#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${W10_SIX_RUN_ROOT:-/workspace/bwa_runs/w10-six-task-v1}"
CHECKPOINT="${W10_SIX_CHECKPOINT:-${RUN_ROOT}/train/formal/checkpoint_120000.pt}"
EVAL_ROOT="${W10_SIX_EVAL_ROOT:-${RUN_ROOT}/evaluation/validation}"
LOG_ROOT="${W10_SIX_LOG_ROOT:-${RUN_ROOT}/logs}"
SEED_ROOT="${W10_SIX_SEED_ROOT:-${RUN_ROOT}/seeds/validation}"
PYTHON_BIN="${W10_SIX_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${W10_SIX_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
EVALUATOR="${ROOT}/stereo_core/evaluate_no_wrist_pair.py"
STATUS="${RUN_ROOT}/validation_status.json"
DRY_RUN=0

fail() {
  printf >&2 'W10 six-task validation: %s\n' "$*"
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/before_we_act/validate_w10_six_task.sh [--dry-run]

Environment overrides:
  W10_SIX_RUN_ROOT, W10_SIX_CHECKPOINT, W10_SIX_EVAL_ROOT,
  W10_SIX_LOG_ROOT, W10_SIX_SEED_ROOT, W10_SIX_PYTHON,
  W10_SIX_ROBOFACTORY_ROOT
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

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

[[ -x "${PYTHON_BIN}" ]] || fail "Python is missing: ${PYTHON_BIN}"
[[ -f "${CHECKPOINT}" ]] || fail "checkpoint is missing: ${CHECKPOINT}"
[[ -f "${EVALUATOR}" ]] || fail "evaluator is missing: ${EVALUATOR}"
[[ -d "${ROBOFACTORY_ROOT}" ]] || fail "RoboFactory root is missing: ${ROBOFACTORY_ROOT}"
for task in "${TASKS[@]}"; do
  [[ -f "${SEED_ROOT}/${task}.json" ]] || fail "validation seeds are missing: ${SEED_ROOT}/${task}.json"
done

if ((DRY_RUN)); then
  printf 'W10 six-task validation dry-run passed\n'
  printf 'checkpoint=%s\nseed_root=%s\neval_root=%s\nlog_root=%s\n' \
    "${CHECKPOINT}" "${SEED_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}"
  printf 'wave1: camera_alignment=gpu0 long_pipeline_delivery=gpu1 take_photo=gpu2 lift_barrier=gpu3\n'
  printf 'wave2: pass_shoe=gpu0 place_food=gpu1\n'
  exit 0
fi

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}"
CHILDREN=()

write_status() {
  local status="$1" wave="$2" detail="$3"
  "${PYTHON_BIN}" - "${STATUS}" "${status}" "${wave}" "${detail}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": "validation",
    "wave": sys.argv[3],
    "detail": sys.argv[4],
    "tasks": [
        "lift_barrier",
        "camera_alignment",
        "long_pipeline_delivery",
        "take_photo",
        "pass_shoe",
        "place_food",
    ],
    "episodes_per_task": 20,
    "total_episodes": 120,
    "updated_at_epoch": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}

on_signal() {
  trap - ERR
  write_status STOPPED interrupted "stopping active validation children"
  stop_children
  exit 130
}

on_error() {
  local code=$?
  write_status FAILED error "validation command exited with code ${code}" || true
  stop_children
  exit "${code}"
}

trap on_signal INT TERM
trap on_error ERR

is_complete() {
  local output="$1"
  [[ -f "${output}" ]] && "${PYTHON_BIN}" - "${output}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
complete = payload.get("episodes") == 20 and len(payload.get("rows", [])) == 20
raise SystemExit(0 if complete else 1)
PY
}

launch_task() {
  local task="$1" gpu="$2"
  local output="${EVAL_ROOT}/${task}.json"
  local log="${LOG_ROOT}/validation_${task}.log"
  if is_complete "${output}"; then
    printf '[%s] preserve completed validation: %s\n' "$(date -u +%FT%TZ)" "${task}"
    return 0
  fi
  printf '[%s] launch validation task=%s gpu=%s\n' "$(date -u +%FT%TZ)" "${task}" "${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/stereo_core:${ROOT}:${ROBOFACTORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -u "${EVALUATOR}" \
      --checkpoint "${CHECKPOINT}" \
      --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" \
      --episodes 20 \
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
  local item task gpu
  for item in "$@"; do
    task="${item%%:*}"
    gpu="${item##*:}"
    launch_task "${task}" "${gpu}"
  done
  local code=0 child_code pid
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

write_status STARTING prepare "validating checkpoint and fixed seed protocol"
run_wave wave1 camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave wave2 pass_shoe:0 place_food:1

"${PYTHON_BIN}" - "${EVAL_ROOT}" "${CHECKPOINT}" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
tasks = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
results = {task: json.loads((root / f"{task}.json").read_text()) for task in tasks}
payload = {
    # PASSED means that the fixed 120-episode protocol completed. The measured
    # success counts below, not this field, define model quality.
    "status": "PASSED",
    "stage": "validation",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "episodes": sum(results[task]["episodes"] for task in tasks),
    "successes": sum(results[task]["successes"] for task in tasks),
    "tasks": {
        task: {
            "episodes": results[task]["episodes"],
            "successes": results[task]["successes"],
            "success_rate": results[task]["success_rate"],
        }
        for task in tasks
    },
    "completed_at_epoch": time.time(),
}
(root / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True), flush=True)
PY
write_status PASSED complete "all 120 validation episodes completed"
trap - ERR
