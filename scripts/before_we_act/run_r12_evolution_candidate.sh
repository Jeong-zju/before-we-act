#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
NORMALIZATION_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
FULL_INDEX=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2/index.json
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
PYTHON=/venv/robofactory-act/bin/python

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift 2 ;;
    --normalization-checkpoint) NORMALIZATION_CHECKPOINT="$2"; shift 2 ;;
    --full-index) FULL_INDEX="$2"; shift 2 ;;
    --vision-artifact) VISION_ARTIFACT="$2"; shift 2 ;;
    --protocol-root) PROTOCOL_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2; exit 2
fi

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$FE_ROOT/scripts/before_we_act/r12_runtime.py"
CONFIG="$FE_ROOT/configs/before_we_act/r12_action/e1_$CANDIDATE.yaml"
BELIEF_CONFIG="$FE_ROOT/configs/before_we_act/r11_belief/p0.yaml"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"
RECEIPTS="$CANDIDATE_ROOT/receipts"
FORMAL="$CANDIDATE_ROOT/train/formal"
TARGET_UPDATES=130000
mkdir -p "$LOG_ROOT" "$RECEIPTS" "$CANDIDATE_ROOT/preflight" "$FORMAL" "$CANDIDATE_ROOT/validation/gate20"
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
  printf 'R12-E1 candidate worktree branch/commit differs from manifest\n' >&2; exit 3
fi
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$BELIEF_CONFIG" "$BELIEF_CHECKPOINT" "$NORMALIZATION_CHECKPOINT" "$PROTOCOL_ROOT/baseline_gate20.json" "$VISION_ARTIFACT/config.json" "$VISION_ARTIFACT/model.safetensors"; do
  [[ -e "$path" ]] || { printf 'missing required R12-E1 path: %s\n' "$path" >&2; exit 3; }
done

CHILD_PID=0
HEARTBEAT_PID=0
STOP_REQUESTED=0
TERMINAL_WRITTEN=0
CHILD_FILE="$CANDIDATE_ROOT/child.pid"
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" \
    --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" \
    --child-pid "$CHILD_PID" --log "$MAIN_LOG" ${5:+--total-updates "$5"}
  case "$1" in PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    local observed=0
    [[ -f "$CHILD_FILE" ]] && observed="$(<"$CHILD_FILE")"
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$observed" >/dev/null 2>&1 || true
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
    status STOPPED stopped run_r12_evolution_candidate.sh "graceful stop; latest checkpoint preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then
    status FAILED failed run_r12_evolution_candidate.sh "pipeline exited with code $code; inspect log" || true
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
  printf '%s\n' "$CHILD_PID" >"$CHILD_FILE"
  status "$state" "$stage" "$program" "$detail" "$total"
  local code=0
  wait "$CHILD_PID" || code=$?
  CHILD_PID=0
  printf '0\n' >"$CHILD_FILE"
  return "$code"
}

status PREPARING cache_wait run_r12_evolution_candidate.sh "waiting for complete high-resolution full-data index" "$TARGET_UPDATES"
for _ in $(seq 1 4320); do [[ -f "$FULL_INDEX" ]] && break; sleep 10; done
[[ -f "$FULL_INDEX" ]] || { printf 'R12-E1 full index did not appear\n' >&2; exit 3; }
status PREPARING gpu_wait run_r12_evolution_candidate.sh "waiting for exclusive physical GPU $GPU_INDEX" "$TARGET_UPDATES"
for _ in $(seq 1 4320); do
  if ! nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then break; fi
  sleep 10
done
if nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  printf 'R12-E1 GPU %s never became exclusive\n' "$GPU_INDEX" >&2; exit 3
fi

MODULE="$($PYTHON - "$CANDIDATE" <<'PY'
import sys
print({"p0":"before_we_act/action_generator/openpi_flow.py","p1":"before_we_act/action_generator/smolvla_flow.py","p2":"before_we_act/action_generator/act_chunk.py","p3":"before_we_act/action_generator/diffusion_policy_transformer.py"}[sys.argv[1]])
PY
)"
run_child PREPARING tests pytest "candidate and common evolution tests" "$TARGET_UPDATES" \
  "$PYTHON" -m pytest -q "$FE_ROOT/tests/before_we_act/test_r12_$CANDIDATE.py" "$FE_ROOT/tests/before_we_act/test_r12_full_episode_windows.py" "$FE_ROOT/tests/before_we_act/test_r12_common.py"
run_child PREPARING core_free audit_r12_core_free.py "core-free specialist runtime audit" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_r12_core_free.py" --round R12-E1 --project-root "$FE_ROOT" --candidate-module "$MODULE" --output "$RECEIPTS/core_free.json"

run_child TRAINING preflight train_action_generator_evolution.py "two-update full-data train/save smoke" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_action_generator_evolution --config "$CONFIG" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --full-index "$FULL_INDEX" --normalization-checkpoint "$NORMALIZATION_CHECKPOINT" --output "$CANDIDATE_ROOT/preflight" --device cuda:0 --updates 2 --workers 2
run_child VALIDATING preflight_restore verify_r12_evolution_preflight.py "strict restore, image and task effects" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_r12_evolution_preflight.py" --config "$CONFIG" --checkpoint "$CANDIDATE_ROOT/preflight/checkpoints/checkpoint_000002.pt" --device cuda:0 --output "$RECEIPTS/preflight.json"

RESUME_ARGS=()
[[ -f "$FORMAL/checkpoints/checkpoint_latest.pt" ]] && RESUME_ARGS=(--resume "$FORMAL/checkpoints/checkpoint_latest.pt")
run_child TRAINING formal train_action_generator_evolution.py "10k image/task alignment plus 120k full-data joint training" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_action_generator_evolution --config "$CONFIG" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --full-index "$FULL_INDEX" --normalization-checkpoint "$NORMALIZATION_CHECKPOINT" --output "$FORMAL" --device cuda:0 --workers 4 --heartbeat "$CANDIDATE_ROOT/training_heartbeat.json" "${RESUME_ARGS[@]}"
CHECKPOINT="$FORMAL/checkpoints/checkpoint_130000.pt"
BRIDGE_CHECKPOINT="$FORMAL/checkpoints/checkpoint_010000.pt"
run_child VALIDATING offline_validation evaluate_action_generator_evolution_offline.py "all 22475 held-out timesteps" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.evaluate_action_generator_evolution_offline --config "$CONFIG" --checkpoint "$CHECKPOINT" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --full-index "$FULL_INDEX" --output "$CANDIDATE_ROOT/validation/offline.json" --heartbeat "$CANDIDATE_ROOT/offline_heartbeat.json" --device cuda:0 --batch-size 10 --workers 4

for task in lift_barrier camera_alignment long_pipeline_delivery take_photo; do
  run_child VALIDATING exact_fallback materialize_r12_evolution_fallback.py "$task immutable exact-W10 paired result" "$TARGET_UPDATES" \
    env PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/materialize_r12_evolution_fallback.py" --config "$CONFIG" --task "$task" --baseline "/workspace/bwa_runs/shared/frozen100/$task.json" --seed-file "$PROTOCOL_ROOT/seeds/$task.json" --output "$CANDIDATE_ROOT/validation/gate20/$task.json"
done
task=three_robots_stack_cube
eval_log="$LOG_ROOT/gate20_${task}.log"
run_child VALIDATING gate20 evaluate_action_generator_evolution.py "Stack paired Gate20; exactly 20 episodes" "$TARGET_UPDATES" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" BWA_R12_RUN_ROOT="$RUN_ROOT" BWA_R12_CANDIDATE="$CANDIDATE" \
  "$PYTHON" -m before_we_act.evaluate_action_generator_evolution --config "$CONFIG" --checkpoint "$CHECKPOINT" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --vision-artifact "$VISION_ARTIFACT" --vision-batch-size 5 --task "$task" --seed-file "$PROTOCOL_ROOT/seeds/$task.json" --episodes 20 --max-steps 1500 --device cuda:0 --output "$CANDIDATE_ROOT/validation/gate20/$task.json" --resume-log "$eval_log" >>"$eval_log" 2>&1

GATE_ARGS=()
BASE_ARGS=()
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  GATE_ARGS+=(--gate20 "$task=$CANDIDATE_ROOT/validation/gate20/$task.json")
  BASE_ARGS+=(--baseline "$task=/workspace/bwa_runs/shared/frozen100/$task.json")
done
set +e
run_child ACCEPTING acceptance accept_r12_evolution.py "hybrid R11+R12 mean strictly above W10" "$TARGET_UPDATES" \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/accept_r12_evolution.py" --candidate "$CANDIDATE" --branch "$EXPECTED_BRANCH" --commit "$EXPECTED_COMMIT" --checkpoint "$CHECKPOINT" --bridge-checkpoint "$BRIDGE_CHECKPOINT" --preflight "$RECEIPTS/preflight.json" --offline "$CANDIDATE_ROOT/validation/offline.json" --core-free "$RECEIPTS/core_free.json" --training-identity "$FORMAL/training_identity.json" --full-index "$FULL_INDEX" --baseline-summary "$PROTOCOL_ROOT/baseline_gate20.json" "${BASE_ARGS[@]}" "${GATE_ARGS[@]}" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
TOTAL="$($PYTHON - "$CANDIDATE_ROOT/acceptance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["gate20"]["candidate_total_successes"])
PY
)"
if ((ACCEPT_CODE == 0)); then
  status PASSED complete accept_r12_evolution.py "R12-E1 qualified; hybrid Gate20=$TOTAL/100 >74" "$TARGET_UPDATES"
else
  status FAILED complete accept_r12_evolution.py "R12-E1 not qualified; hybrid Gate20=$TOTAL/100" "$TARGET_UPDATES"
fi
exit "$ACCEPT_CODE"
