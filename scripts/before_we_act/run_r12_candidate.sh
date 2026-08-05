#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
NORMALIZATION_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
ACTION_CACHE=/workspace/bwa_runs/shared/r12_causal_coldstart_action_cache.pt
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
DATA_ROOT=/workspace/datasets/robofactory_multitask
PYTHON=/venv/robofactory-act/bin/python

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift 2 ;;
    --normalization-checkpoint) NORMALIZATION_CHECKPOINT="$2"; shift 2 ;;
    --action-cache) ACTION_CACHE="$2"; shift 2 ;;
    --protocol-root) PROTOCOL_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2
  exit 2
fi

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$FE_ROOT/scripts/before_we_act/r12_runtime.py"
CONFIG="$FE_ROOT/configs/before_we_act/r12_action/$CANDIDATE.yaml"
BELIEF_CONFIG="$FE_ROOT/configs/before_we_act/r11_belief/p0.yaml"
LOCK="$FE_ROOT/experiments/before_we_act/r12/$CANDIDATE/component_lock.yaml"
PARITY="$FE_ROOT/experiments/before_we_act/r12/$CANDIDATE/parity.py"
MODULE="$($PYTHON - "$CANDIDATE" <<'PY'
import sys
print({"p0":"before_we_act/action_generator/openpi_flow.py","p1":"before_we_act/action_generator/smolvla_flow.py","p2":"before_we_act/action_generator/rdt_diffusion.py","p3":"before_we_act/action_generator/consistency_policy.py"}[sys.argv[1]])
PY
)"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"
RECEIPTS="$CANDIDATE_ROOT/receipts"
UPSTREAM="/workspace/bwa_upstream/r12/$CANDIDATE"
TARGET_UPDATES="$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["training"]["updates"])
PY
)"
[[ "$TARGET_UPDATES" == 20000 ]] || { printf 'R12-R1 requires exactly 20000 updates\n' >&2; exit 3; }
CHECKPOINT_NAME="$(printf 'checkpoint_%06d.pt' "$TARGET_UPDATES")"
mkdir -p "$LOG_ROOT" "$RECEIPTS" "$CANDIDATE_ROOT/preflight" "$CANDIDATE_ROOT/train/formal" "$CANDIDATE_ROOT/validation/gate20"
exec > >(tee -a "$MAIN_LOG") 2>&1

EXPECTED_BRANCH="$($PYTHON - "$MANIFEST" "$CANDIDATE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["branches"][sys.argv[2]])
PY
)"
EXPECTED_COMMIT="$($PYTHON - "$MANIFEST" "$CANDIDATE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["commits"][sys.argv[2]])
PY
)"
if [[ "$(git -C "$FE_ROOT" branch --show-current)" != "$EXPECTED_BRANCH" || "$(git -C "$FE_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  printf 'R12 candidate worktree branch/commit differs from manifest\n' >&2
  exit 3
fi
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$BELIEF_CONFIG" "$LOCK" "$PARITY" "$BELIEF_CHECKPOINT" "$NORMALIZATION_CHECKPOINT" "$DATA_ROOT" "$PROTOCOL_ROOT/baseline_gate20.json"; do
  [[ -e "$path" ]] || { printf 'missing required R12 path: %s\n' "$path" >&2; exit 3; }
done

CHILD_PID=0
HEARTBEAT_PID=0
STOP_REQUESTED=0
TERMINAL_WRITTEN=0
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" \
    --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" \
    --child-pid "$CHILD_PID" --log "$MAIN_LOG" ${5:+--total-updates "$5"}
  case "$1" in PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$CHILD_PID" >/dev/null 2>&1 || true
    sleep 20
  done
}
on_signal() {
  STOP_REQUESTED=1
  if [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT "$CHILD_PID" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then
    status STOPPED stopped run_r12_candidate.sh "graceful stop requested; latest checkpoint preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then
    status FAILED failed run_r12_candidate.sh "pipeline exited with code $code; inspect receipts/log" || true
  fi
}
trap on_signal INT TERM
trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!

run_child() {
  local state="$1" stage="$2" program="$3" detail="$4" total="$5"
  shift 5
  status "$state" "$stage" "$program" "$detail" "$total"
  "$@" & CHILD_PID=$!
  status "$state" "$stage" "$program" "$detail" "$total"
  local code=0
  wait "$CHILD_PID" || code=$?
  CHILD_PID=0
  return "$code"
}

status PREPARING cache_wait run_r12_candidate.sh "waiting for shared causal/cold-start R12 action cache" "$TARGET_UPDATES"
for _ in $(seq 1 3600); do
  [[ -f "$ACTION_CACHE" ]] && break
  sleep 10
done
[[ -f "$ACTION_CACHE" ]] || { printf 'shared R12 action cache did not appear\n' >&2; exit 3; }

REPO="$($PYTHON - "$LOCK" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["official_repo"])
PY
)"
UPSTREAM_COMMIT="$($PYTHON - "$LOCK" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["upstream_commit_sha"])
PY
)"
run_child DOWNLOADING source_fetch fetch_upstream_readonly.py "verify official pinned source" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/fetch_upstream_readonly.py" --repo "$REPO" --commit "$UPSTREAM_COMMIT" --destination "$UPSTREAM"
run_child PREPARING source_verify verify_upstream_source.py "official repo/commit/clean checkout" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_upstream_source.py" --lock "$LOCK" --upstream "$UPSTREAM" --output "$RECEIPTS/source.json"
run_child PREPARING license verify_component_license.py "license preserved" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_component_license.py" --lock "$LOCK" --project-root "$FE_ROOT" --output "$RECEIPTS/license.json"
run_child PREPARING patch audit_component_patch.py "minimal transplant patch audit" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_component_patch.py" --lock "$LOCK" --upstream "$UPSTREAM" --project-root "$FE_ROOT" --patch-output "$RECEIPTS/upstream_adaptation.patch" --report-output "$RECEIPTS/patch.json"
run_child PREPARING dependency audit_no_full_repo_dependency.py "no full upstream runtime import" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_no_full_repo_dependency.py" --project-root "$FE_ROOT" --output "$RECEIPTS/dependency.json"
run_child PREPARING core_free audit_r12_core_free.py "physical CoRE/runtime separation" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_r12_core_free.py" --project-root "$FE_ROOT" --candidate-module "$MODULE" --output "$RECEIPTS/core_free.json"
run_child PREPARING parity parity.py "official/local component numerical parity" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$PARITY" --upstream "$UPSTREAM" --output "$RECEIPTS/parity.json" --device cuda:0

run_child TRAINING preflight train_action_generator.py "two-update train/save test" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_action_generator --config "$CONFIG" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --cache "$ACTION_CACHE" --output "$CANDIDATE_ROOT/preflight" --device cuda:0 --updates 2
run_child VALIDATING preflight_restore verify_r12_preflight.py "strict restore, normalization, finite/range/mask" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_r12_preflight.py" --config "$CONFIG" --checkpoint "$CANDIDATE_ROOT/preflight/checkpoints/checkpoint_000002.pt" --device cuda:0 --output "$RECEIPTS/preflight.json"

FORMAL="$CANDIDATE_ROOT/train/formal"
run_child TRAINING formal train_action_generator.py "causal/cold-start 20000-update action component training" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_action_generator --config "$CONFIG" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --cache "$ACTION_CACHE" --output "$FORMAL" --device cuda:0
CHECKPOINT="$FORMAL/checkpoints/$CHECKPOINT_NAME"
run_child VALIDATING offline_validation evaluate_action_generator_offline.py "held-out finite/control-cycle smoke; no quality threshold" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.evaluate_action_generator_offline --config "$CONFIG" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --cache "$ACTION_CACHE" --checkpoint "$CHECKPOINT" --output "$CANDIDATE_ROOT/validation/offline.json" --device cuda:0

for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  eval_log="$LOG_ROOT/gate20_${task}.log"
  run_child VALIDATING gate20 evaluate_action_generator.py "$task paired Gate20; exactly 20 episodes" "$TARGET_UPDATES" \
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" BWA_R12_RUN_ROOT="$RUN_ROOT" BWA_R12_CANDIDATE="$CANDIDATE" \
    "$PYTHON" -m before_we_act.evaluate_action_generator --config "$CONFIG" --checkpoint "$CHECKPOINT" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --task "$task" --seed-file "$PROTOCOL_ROOT/seeds/$task.json" --episodes 20 --max-steps 1500 --device cuda:0 --output "$CANDIDATE_ROOT/validation/gate20/$task.json" --resume-log "$eval_log" >>"$eval_log" 2>&1
done

GATE_ARGS=()
BASE_ARGS=()
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  GATE_ARGS+=(--gate20 "$task=$CANDIDATE_ROOT/validation/gate20/$task.json")
  BASE_ARGS+=(--baseline "$task=/workspace/bwa_runs/shared/frozen100/$task.json")
done
set +e
run_child ACCEPTING acceptance accept_r12.py "engineering hard gates plus complete Gate20 > W10 74/100" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/accept_r12.py" --candidate "$CANDIDATE" --branch "$EXPECTED_BRANCH" --commit "$EXPECTED_COMMIT" --checkpoint "$CHECKPOINT" --expected-updates "$TARGET_UPDATES" --source "$RECEIPTS/source.json" --license "$RECEIPTS/license.json" --patch "$RECEIPTS/patch.json" --dependency "$RECEIPTS/dependency.json" --parity "$RECEIPTS/parity.json" --preflight "$RECEIPTS/preflight.json" --offline "$CANDIDATE_ROOT/validation/offline.json" --core-free "$RECEIPTS/core_free.json" --baseline-summary "$PROTOCOL_ROOT/baseline_gate20.json" "${BASE_ARGS[@]}" "${GATE_ARGS[@]}" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
TOTAL="$($PYTHON - "$CANDIDATE_ROOT/acceptance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["gate20"]["candidate_total_successes"])
PY
)"
if ((ACCEPT_CODE == 0)); then
  status PASSED complete accept_r12.py "qualified R12 component; Gate20=$TOTAL/100 strictly >74" "$TARGET_UPDATES"
else
  status FAILED complete accept_r12.py "R12 not qualified; Gate20=$TOTAL/100 or engineering gate failed" "$TARGET_UPDATES"
fi
exit "$ACCEPT_CODE"
