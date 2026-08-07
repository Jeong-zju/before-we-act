#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""; CANDIDATE=""; GPU_INDEX=""
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
ACTION_CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
WORLD_CHECKPOINT=/workspace/bwa_runs/shared/w13/checkpoint_010000.pt
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
W12_RUN=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2
PYTHON=/venv/robofactory-act/bin/python
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift 2 ;;
    --action-checkpoint) ACTION_CHECKPOINT="$2"; shift 2 ;;
    --world-checkpoint) WORLD_CHECKPOINT="$2"; shift 2 ;;
    --vision-artifact) VISION_ARTIFACT="$2"; shift 2 ;;
    --protocol-root) PROTOCOL_ROOT="$2"; shift 2 ;;
    --w12-run) W12_RUN="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2; exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/scripts/before_we_act/r14_runtime.py"
CONFIG="$ROOT/configs/before_we_act/r14_decision/$CANDIDATE.yaml"
LOCK="$ROOT/experiments/before_we_act/r14/$CANDIDATE/component_lock.yaml"
PARITY="$ROOT/experiments/before_we_act/r14/$CANDIDATE/parity.py"
BELIEF_CONFIG="$ROOT/configs/before_we_act/r11_belief/p0.yaml"
ACTION_CONFIG="$ROOT/configs/before_we_act/r12_action/e1_p2.yaml"
WORLD_CONFIG="$ROOT/configs/before_we_act/r13_world/p0.yaml"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"; MAIN_LOG="$LOG_ROOT/candidate.log"
RECEIPTS="$CANDIDATE_ROOT/receipts"; GATE_ROOT="$CANDIDATE_ROOT/validation/gate20"
mkdir -p "$LOG_ROOT" "$RECEIPTS" "$CANDIDATE_ROOT/preflight" "$GATE_ROOT"
exec > >(tee -a "$MAIN_LOG") 2>&1

EXPECTED_BRANCH="$($PYTHON - "$MANIFEST" "$CANDIDATE" <<'PY'
import json,sys; print(json.load(open(sys.argv[1]))["branches"][sys.argv[2]])
PY
)"
EXPECTED_COMMIT="$($PYTHON - "$MANIFEST" "$CANDIDATE" <<'PY'
import json,sys; print(json.load(open(sys.argv[1]))["commits"][sys.argv[2]])
PY
)"
PARENT_COMMIT="$($PYTHON - "$MANIFEST" <<'PY'
import json,sys; print(json.load(open(sys.argv[1]))["parent_commit"])
PY
)"
[[ "$(git -C "$ROOT" branch --show-current)" == "$EXPECTED_BRANCH" && "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { printf 'R14 branch/commit differs from manifest\n' >&2; exit 3; }
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$LOCK" "$PARITY" "$BELIEF_CONFIG" "$ACTION_CONFIG" "$WORLD_CONFIG" "$BELIEF_CHECKPOINT" "$ACTION_CHECKPOINT" "$WORLD_CHECKPOINT" "$VISION_ARTIFACT/config.json" "$VISION_ARTIFACT/model.safetensors"; do
  [[ -e "$path" ]] || { printf 'missing R14 path: %s\n' "$path" >&2; exit 3; }
done

CHILD_PID=0; HEARTBEAT_PID=0; STOP_REQUESTED=0; TERMINAL_WRITTEN=0
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" \
    --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" \
    --child-pid "$CHILD_PID" --total-steps 100 --log "$MAIN_LOG"
  case "$1" in PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$CHILD_PID" >/dev/null 2>&1 || true
    sleep 20
  done
}
on_signal() { STOP_REQUESTED=1; [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; }
cleanup() {
  local code=$?; kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then status STOPPED stopped run_r14_candidate.sh "graceful stop; outputs preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then status FAILED failed run_r14_candidate.sh "pipeline exit=$code; inspect receipts/log" || true; fi
}
trap on_signal INT TERM; trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!
run_child() {
  local state="$1" stage="$2" program="$3" detail="$4"; shift 4
  status "$state" "$stage" "$program" "$detail"
  "$@" & CHILD_PID=$!; status "$state" "$stage" "$program" "$detail"
  local code=0; wait "$CHILD_PID" || code=$?; CHILD_PID=0; return "$code"
}

UPSTREAM="/workspace/bwa_upstream/r14/$CANDIDATE"
REPO="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["official_repo"])' "$LOCK")"
UPSTREAM_COMMIT="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["upstream_commit_sha"])' "$LOCK")"
run_child DOWNLOADING source_fetch fetch_upstream_readonly.py "fetch/verify pinned official source" \
  "$PYTHON" "$ROOT/scripts/before_we_act/fetch_upstream_readonly.py" --repo "$REPO" --commit "$UPSTREAM_COMMIT" --destination "$UPSTREAM"
run_child PREPARING source_verify verify_upstream_source.py "official repo, exact commit, clean checkout" \
  "$PYTHON" "$ROOT/scripts/before_we_act/verify_upstream_source.py" --lock "$LOCK" --upstream "$UPSTREAM" --output "$RECEIPTS/source.json"
run_child PREPARING license verify_component_license.py "preserved MIT license" \
  "$PYTHON" "$ROOT/scripts/before_we_act/verify_component_license.py" --lock "$LOCK" --project-root "$ROOT" --output "$RECEIPTS/license.json"
run_child PREPARING patch audit_component_patch.py "algorithmic lines unchanged" \
  "$PYTHON" "$ROOT/scripts/before_we_act/audit_component_patch.py" --lock "$LOCK" --upstream "$UPSTREAM" --project-root "$ROOT" --patch-output "$RECEIPTS/upstream_adaptation.patch" --report-output "$RECEIPTS/patch.json"
run_child PREPARING dependency audit_no_full_repo_dependency.py "no full upstream runtime dependency" \
  "$PYTHON" "$ROOT/scripts/before_we_act/audit_no_full_repo_dependency.py" --project-root "$ROOT" --output "$RECEIPTS/dependency.json"
run_child PREPARING action_effect classify_r14_action_effect.py "R14 is action-affecting; Gate20 mandatory" \
  "$PYTHON" "$ROOT/scripts/before_we_act/classify_r14_action_effect.py" --parent "$PARENT_COMMIT" --head HEAD --output "$RECEIPTS/action_effect.json"
run_child PREPARING parity parity.py "official/local decision component parity" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$PARITY" --upstream "$UPSTREAM" --output "$RECEIPTS/parity.json" --device cuda:0
run_child PREPARING tests pytest "common and candidate R14 tests" \
  env PYTHONPATH="$ROOT" "$PYTHON" -m pytest -q "$ROOT/tests/before_we_act/test_r14_common.py" "$ROOT/tests/before_we_act/test_r14_$CANDIDATE.py"
run_child PREPARING preflight verify_r14_preflight.py "finite/effective/trust-region smoke" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/before_we_act/verify_r14_preflight.py" --config "$CONFIG" --device cuda:0 --output "$RECEIPTS/preflight.json"
run_child PREPARING separation audit_r14_method_separation.py "CoRE and full repo absent from runtime" \
  "$PYTHON" "$ROOT/scripts/before_we_act/audit_r14_method_separation.py" --project-root "$ROOT" --dependency "$RECEIPTS/dependency.json" --patch "$RECEIPTS/patch.json" --output "$RECEIPTS/separation.json"

TASKS=(lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo)
for task in "${TASKS[@]}"; do
  seed_file="$PROTOCOL_ROOT/seeds/$task.json"; w12_report="$W12_RUN/validation/gate20/$task.json"
  [[ -f "$seed_file" && -f "$w12_report" ]] || { printf 'missing paired protocol/W12 report for %s\n' "$task" >&2; exit 3; }
  if [[ "$task" != three_robots_stack_cube ]]; then
    run_child VALIDATING gate20 materialize_r14_w12_fallback.py "$task exact W12 protected route" \
      env PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/before_we_act/materialize_r14_w12_fallback.py" --config "$CONFIG" --task "$task" --w12-report "$w12_report" --seed-file "$seed_file" --output "$GATE_ROOT/$task.json"
  else
    eval_log="$LOG_ROOT/gate20_${task}.log"
    run_child VALIDATING gate20 evaluate_world_guided_decision.py "$task paired Gate20; 20 live episodes" \
      env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" -m before_we_act.evaluate_world_guided_decision --config "$CONFIG" --action-config "$ACTION_CONFIG" --action-checkpoint "$ACTION_CHECKPOINT" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --world-config "$WORLD_CONFIG" --world-checkpoint "$WORLD_CHECKPOINT" --vision-artifact "$VISION_ARTIFACT" --vision-batch-size 5 --task "$task" --seed-file "$seed_file" --episodes 20 --max-steps 1500 --device cuda:0 --output "$GATE_ROOT/$task.json" --resume-log "$eval_log" >>"$eval_log" 2>&1
  fi
done

GATE_ARGS=(); BASE_ARGS=()
for task in "${TASKS[@]}"; do
  GATE_ARGS+=(--gate20 "$task=$GATE_ROOT/$task.json")
  BASE_ARGS+=(--baseline "$task=$W12_RUN/validation/gate20/$task.json")
done
set +e
run_child ACCEPTING acceptance accept_r14.py "provenance/separation plus complete Gate20 > W12 77" \
  "$PYTHON" "$ROOT/scripts/before_we_act/accept_r14.py" --candidate "$CANDIDATE" --branch "$EXPECTED_BRANCH" --commit "$EXPECTED_COMMIT" --source "$RECEIPTS/source.json" --license "$RECEIPTS/license.json" --patch "$RECEIPTS/patch.json" --dependency "$RECEIPTS/dependency.json" --action-effect "$RECEIPTS/action_effect.json" --parity "$RECEIPTS/parity.json" --preflight "$RECEIPTS/preflight.json" --separation "$RECEIPTS/separation.json" "${GATE_ARGS[@]}" "${BASE_ARGS[@]}" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
TOTAL="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate20"]["candidate_total_successes"])' "$CANDIDATE_ROOT/acceptance.json")"
if ((ACCEPT_CODE == 0)); then status PASSED complete accept_r14.py "R14 qualified; Gate20=$TOTAL/100 >77"
else status FAILED complete accept_r14.py "R14 not qualified; Gate20=$TOTAL/100 or engineering gate failed"; fi
if [[ -f "$RUN_ROOT/candidates/p0/acceptance.json" && -f "$RUN_ROOT/candidates/p1/acceptance.json" && -f "$RUN_ROOT/candidates/p2/acceptance.json" && -f "$RUN_ROOT/candidates/p3/acceptance.json" ]]; then
  ( flock -x 9; "$PYTHON" "$ROOT/scripts/before_we_act/decide_r14_winner.py" --run-root "$RUN_ROOT" --output "$RUN_ROOT/round_decision.json" || true ) 9>"$RUN_ROOT/.decision.lock"
fi
exit "$ACCEPT_CODE"
