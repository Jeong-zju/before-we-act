#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r10-20260804
CANDIDATE=p1
PYTHON=/venv/robofactory-act/bin/python
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
PARENT_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$CANDIDATE" =~ ^p[0-3]$ ]] || { printf 'candidate p0..p3 required\n' >&2; exit 2; }
GPU_INDEX="${CANDIDATE#p}"
MANIFEST="$RUN_ROOT/run_manifest.json"
[[ -f "$MANIFEST" ]] || { printf 'missing run manifest\n' >&2; exit 3; }
WORKTREE="$(jq -r --arg candidate "$CANDIDATE" '.worktrees[$candidate]' "$MANIFEST")"
BRANCH="$(jq -r --arg candidate "$CANDIDATE" '.branches[$candidate]' "$MANIFEST")"
COMMIT="$(jq -r --arg candidate "$CANDIDATE" '.commits[$candidate]' "$MANIFEST")"
[[ "$(git -C "$WORKTREE" branch --show-current)" == "$BRANCH" && \
   "$(git -C "$WORKTREE" rev-parse HEAD)" == "$COMMIT" ]] || {
  printf 'candidate worktree identity differs\n' >&2; exit 3;
}
OFFICIAL_STATUS="$RUN_ROOT/candidates/$CANDIDATE/status.json"
OFFICIAL_GATE="$RUN_ROOT/candidates/$CANDIDATE/validation/screen/gate_zero_latency.json"
CHECKPOINT="$RUN_ROOT/candidates/$CANDIDATE/train/screen/checkpoints/checkpoint_010000.pt"
[[ "$(jq -r '.state' "$OFFICIAL_STATUS")" == FAILED ]] || {
  printf 'diagnostic only accepts an officially failed candidate\n' >&2; exit 3;
}
[[ "$(jq -r '.gate_zero_passed' "$OFFICIAL_GATE")" == true && \
   "$(jq -r '.latency_passed' "$OFFICIAL_GATE")" == false ]] || {
  printf 'candidate did not fail only after an exact gate-zero latency audit\n' >&2; exit 3;
}
[[ -f "$CHECKPOINT" ]] || { printf 'missing 10k checkpoint\n' >&2; exit 3; }

CONFIG="$WORKTREE/configs/before_we_act/r10_perception/$CANDIDATE.yaml"
EVALUATOR="$WORKTREE/stereo_core/evaluate_bwa_perception.py"
ACCEPTOR="$WORKTREE/scripts/before_we_act/accept_r10.py"
INTERVENTION="$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["intervention"]["name"])
PY
)"
DIAG_ROOT="$RUN_ROOT/candidates/$CANDIDATE/diagnostics/latency_waived/screen"
STATE="$DIAG_ROOT/status.json"
HEARTBEAT="$DIAG_ROOT/heartbeat"
LOG_ROOT="$DIAG_ROOT/logs"
mkdir -p "$DIAG_ROOT/normal" "$DIAG_ROOT/intervention" "$LOG_ROOT"
[[ ! -f "$DIAG_ROOT/performance_summary.json" ]] || {
  printf 'diagnostic already complete: %s\n' "$DIAG_ROOT"; exit 3;
}

STARTED_AT="$(date -u +%FT%TZ)"
CHILD_PID=0
HEARTBEAT_PID=0
write_state() {
  local status="$1" stage="$2" detail="$3" exit_code="${4:-0}"
  "$PYTHON" - "$STATE.tmp" "$STATE" "$status" "$stage" "$detail" \
    "$STARTED_AT" "$CHILD_PID" "$$" "$exit_code" "$LOG_ROOT" <<'PY'
import json, os, sys
from datetime import datetime, timezone
temporary, destination, status, stage, detail, started, child, pid, code, logs = sys.argv[1:]
payload = {
    "schema_version": 1,
    "diagnostic": "latency-waived-closed-loop",
    "official_status_unchanged": True,
    "status": status,
    "stage": stage,
    "detail": detail,
    "started_at": started,
    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "pid": int(pid),
    "child_pid": int(child),
    "exit_code": int(code),
    "log_root": logs,
}
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(temporary, destination)
PY
}
heartbeat_loop() { while kill -0 "$$" 2>/dev/null; do touch "$HEARTBEAT"; sleep 20; done; }
cleanup() {
  local code=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((code != 0)); then write_state FAILED failed "diagnostic exited; official result remains FAILED" "$code" || true; fi
}
trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!
write_state RUNNING preparing "official latency failure retained; starting closed-loop diagnostic"

for condition in normal "$INTERVENTION"; do
  condition_dir=normal
  [[ "$condition" == normal ]] || condition_dir=intervention
  for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
    output="$DIAG_ROOT/$condition_dir/$task.json"
    eval_log="$LOG_ROOT/${condition_dir}_${task}.log"
    if [[ -f "$output" ]]; then continue; fi
    write_state RUNNING "${condition_dir}_${task}" "condition=$condition task=$task"
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
      "$PYTHON" "$EVALUATOR" \
        --parent-checkpoint "$PARENT_CHECKPOINT" \
        --extension-checkpoint "$CHECKPOINT" \
        --task "$task" --seed-file "$PROTOCOL_ROOT/seeds/$task.json" \
        --episodes 20 --max-steps 1500 --device cuda:0 \
        --intervention "$condition" --output "$output" \
        --resume-log "$eval_log" >>"$eval_log" 2>&1 &
    CHILD_PID=$!
    write_state RUNNING "${condition_dir}_${task}" "condition=$condition task=$task"
    wait "$CHILD_PID"
    CHILD_PID=0
  done
done

MAPPINGS=()
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  MAPPINGS+=(--baseline "$task=/workspace/bwa_runs/shared/frozen100/$task.json")
  MAPPINGS+=(--normal "$task=$DIAG_ROOT/normal/$task.json")
  MAPPINGS+=(--intervention "$task=$DIAG_ROOT/intervention/$task.json")
done
set +e
"$PYTHON" "$ACCEPTOR" --candidate-id "$CANDIDATE" --branch "$BRANCH" \
  --commit "$COMMIT" --checkpoint "$CHECKPOINT" --gate-audit "$OFFICIAL_GATE" \
  "${MAPPINGS[@]}" --mode formal --output "$DIAG_ROOT/official_acceptance.json" \
  >"$LOG_ROOT/acceptance.log" 2>&1
ACCEPT_CODE=$?
set -e
"$PYTHON" - "$DIAG_ROOT/official_acceptance.json" "$DIAG_ROOT/performance_summary.json.tmp" \
  "$DIAG_ROOT/performance_summary.json" "$ACCEPT_CODE" <<'PY'
import json, os, sys
source, temporary, output, accept_code = sys.argv[1:]
payload = json.load(open(source))
closed_loop_ids = {"gate_zero_exact", "paired_gate20", "camera_stack_and_other_tasks", "causal_intervention"}
checks = {row["id"]: bool(row["passed"]) for row in payload["acceptance"]}
summary = {
    "schema_version": 1,
    "diagnostic": "latency-waived-closed-loop",
    "official_status": "FAILED",
    "official_acceptance_exit_code": int(accept_code),
    "official_latency_result_retained": next(
        row for row in payload["acceptance"] if row["id"] == "latency_and_inputs"
    ),
    "closed_loop_gate_ids": sorted(closed_loop_ids),
    "closed_loop_passed_ignoring_latency": all(checks.get(key, False) for key in closed_loop_ids),
    "gate20": payload["gate20"],
    "causal_intervention": payload["causal_intervention"],
    "source": source,
}
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
PY
write_state COMPLETED complete "closed-loop diagnostic complete; official latency failure retained" 0
printf 'completed diagnostic: %s\n' "$DIAG_ROOT/performance_summary.json"
