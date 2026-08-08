#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""; CANDIDATE=p1; GPU_INDEX=""; EXPERT_INDEX=""; PHASE_MANIFEST=""
UPDATES=10000; BATCH_SIZE=12; EXPERT_ROWS=6; LEARNING_RATE=2e-5; WARMUP=500
PYTHON=/venv/robofactory-act/bin/python
PARENT_CHECKPOINT=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2/train/formal/checkpoints/checkpoint_130000.pt
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --expert-index) EXPERT_INDEX="$2"; shift 2 ;;
    --phase-manifest) PHASE_MANIFEST="$2"; shift 2 ;;
    --updates) UPDATES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --expert-rows) EXPERT_ROWS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ROOT" && "$CANDIDATE" =~ ^p[1-3]$ && "$GPU_INDEX" =~ ^[0-3]$ && -n "$EXPERT_INDEX" && "$UPDATES" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$EXPERT_ROWS" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[1-9][0-9]*$ && "$LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]] || { printf 'valid run/candidate/GPU/expert training budget required\n' >&2; exit 2; }
((BATCH_SIZE >= 2 && EXPERT_ROWS < BATCH_SIZE && WARMUP <= UPDATES)) || { printf 'invalid expert sampling or warmup budget\n' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/scripts/before_we_act/r15_runtime.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
CONFIG="$ROOT/configs/before_we_act/r12_action/e1_p2.yaml"
BELIEF_CONFIG="$ROOT/configs/before_we_act/r11_belief/p0.yaml"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
TRAIN_ROOT="$CANDIDATE_ROOT/train/stack_expert"
CHECKPOINT="$TRAIN_ROOT/checkpoints/checkpoint_$(printf '%06d' "$UPDATES").pt"
MAIN_LOG="$CANDIDATE_ROOT/logs/candidate.log"
TRAIN_LOG="$CANDIDATE_ROOT/logs/expert_finetune.log"
HEARTBEAT="$CANDIDATE_ROOT/training_heartbeat.json"
CHILD_FILE="$CANDIDATE_ROOT/child.pid"
mkdir -p "$CANDIDATE_ROOT/logs" "$TRAIN_ROOT"
exec > >(tee -a "$MAIN_LOG") 2>&1

readarray -t IDENTITY < <("$PYTHON" - "$MANIFEST" "$CANDIDATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); c=d["candidates"][sys.argv[2]]
for key in ("worktree","branch","commit","checkpoint"):
    print(c[key])
PY
)
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$CURRENT_BRANCH" =~ ^bwa/r15-(closed-loop-evolution|expert-evolution|phase-balanced-expert|role-query-specialist|role-query-view-dedup)$ && "${IDENTITY[0]}" == "$ROOT" && "${IDENTITY[1]}" == "$CURRENT_BRANCH" && "${IDENTITY[2]}" == "$(git -C "$ROOT" rev-parse HEAD)" && "${IDENTITY[3]}" == "$CHECKPOINT" && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'R15 expert run identity differs\n' >&2; exit 3; }
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$BELIEF_CONFIG" "$BELIEF_CHECKPOINT" "$PARENT_CHECKPOINT" "$EXPERT_INDEX"; do
  [[ -e "$path" ]] || { printf 'missing R15 expert input: %s\n' "$path" >&2; exit 3; }
done
PHASE_ARGS=()
DETAIL="source-aware Stack continuation"
if [[ -n "$PHASE_MANIFEST" ]]; then
  [[ -f "$PHASE_MANIFEST" && $((EXPERT_ROWS % 3)) -eq 0 ]] || { printf 'phase-balanced manifest/rows differ\n' >&2; exit 3; }
  PHASE_ARGS=(--phase-manifest "$PHASE_MANIFEST")
  DETAIL="three-phase-balanced source-aware Stack continuation"
fi

CHILD_PID=0; HEARTBEAT_PID=0; STOP_REQUESTED=0; TERMINAL_WRITTEN=0
printf '0\n' >"$CHILD_FILE"
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" --child-pid "$CHILD_PID" --log "$MAIN_LOG" ${5:+--exit-code "$5"}
  case "$1" in PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    local observed=0; [[ -f "$CHILD_FILE" ]] && observed="$(<"$CHILD_FILE")"
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$observed" >/dev/null 2>&1 || true
    sleep 20
  done
}
on_signal() { STOP_REQUESTED=1; [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; }
cleanup() {
  local code=$?; kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then status STOPPED stopped run_r15_expert_finetune_candidate.sh "graceful stop; latest checkpoint and logs preserved" 130 || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then status FAILED failed run_r15_expert_finetune_candidate.sh "pipeline exit=$code; inspect log" "$code" || true; fi
}
trap on_signal INT TERM; trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!

RESUME_ARGS=()
[[ -f "$TRAIN_ROOT/checkpoints/checkpoint_latest.pt" ]] && RESUME_ARGS=(--resume "$TRAIN_ROOT/checkpoints/checkpoint_latest.pt")
status TRAINING stack_expert_finetune train_r15_stack_expert.py "$DETAIL; batch=$BATCH_SIZE expert_rows=$EXPERT_ROWS lr=$LEARNING_RATE"
(
  cd "$ROOT"
  exec env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" -m before_we_act.train_r15_stack_expert \
    --config "$CONFIG" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" \
    --expert-index "$EXPERT_INDEX" --output "$TRAIN_ROOT" --device cuda:0 \
    "${PHASE_ARGS[@]}" \
    --updates "$UPDATES" --batch-size "$BATCH_SIZE" --expert-rows "$EXPERT_ROWS" \
    --learning-rate "$LEARNING_RATE" --warmup "$WARMUP" \
    --heartbeat "$HEARTBEAT" "${RESUME_ARGS[@]}"
) >>"$TRAIN_LOG" 2>&1 &
CHILD_PID=$!; printf '%s\n' "$CHILD_PID" >"$CHILD_FILE"; wait "$CHILD_PID"
CHILD_PID=0; printf '0\n' >"$CHILD_FILE"
[[ -f "$CHECKPOINT" ]] || { printf 'R15 expert final checkpoint missing: %s\n' "$CHECKPOINT" >&2; exit 3; }

status VALIDATING closed_loop evaluate_action_generator_evolution.py "expert fine-tune complete; entering frozen paired Stack screen"
kill "$HEARTBEAT_PID" 2>/dev/null || true
wait "$HEARTBEAT_PID" 2>/dev/null || true
trap - EXIT INT TERM
exec "$ROOT/scripts/before_we_act/run_r15_stack_screen.sh" --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --gpu-index "$GPU_INDEX" --python "$PYTHON"
