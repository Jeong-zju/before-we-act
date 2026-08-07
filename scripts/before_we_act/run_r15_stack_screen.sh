#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""; CANDIDATE=""; GPU_INDEX=""; PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$CANDIDATE" =~ ^p[0-3]$ && "$GPU_INDEX" =~ ^[0-3]$ ]] || { printf 'valid candidate/GPU required\n' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/scripts/before_we_act/r15_runtime.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
readarray -t IDENTITY < <("$PYTHON" - "$MANIFEST" "$CANDIDATE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); c=d["candidates"][sys.argv[2]]
for key in ("worktree","branch","commit","config","checkpoint","label"):
    print(c[key])
print("1" if c["reference"] else "0")
print(d["split"]); print(d["seed_file"])
PY
)
WORKTREE="${IDENTITY[0]}"; BRANCH="${IDENTITY[1]}"; COMMIT="${IDENTITY[2]}"
CONFIG="${IDENTITY[3]}"; CHECKPOINT="${IDENTITY[4]}"; LABEL="${IDENTITY[5]}"; REFERENCE="${IDENTITY[6]}"
SPLIT="${IDENTITY[7]}"; SEED_FILE="${IDENTITY[8]}"
BELIEF_CONFIG="$WORKTREE/configs/before_we_act/r11_belief/p0.yaml"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"; LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"; EVAL_LOG="$LOG_ROOT/$SPLIT.log"
OUTPUT="$CANDIDATE_ROOT/validation/$SPLIT.json"
mkdir -p "$LOG_ROOT" "$(dirname "$OUTPUT")"
exec > >(tee -a "$MAIN_LOG") 2>&1
[[ "$(git -C "$WORKTREE" branch --show-current)" == "$BRANCH" && "$(git -C "$WORKTREE" rev-parse HEAD)" == "$COMMIT" && -z "$(git -C "$WORKTREE" status --porcelain)" ]] || { printf 'candidate worktree identity differs\n' >&2; exit 3; }
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$CHECKPOINT" "$BELIEF_CONFIG" "$BELIEF_CHECKPOINT" "$SEED_FILE" "$VISION_ARTIFACT/config.json" "$VISION_ARTIFACT/model.safetensors" "$WORKTREE/before_we_act/evaluate_action_generator_evolution.py"; do
  [[ -e "$path" ]] || { printf 'missing R15 screen input: %s\n' "$path" >&2; exit 3; }
done

CHILD_PID=0; HEARTBEAT_PID=0; STOP_REQUESTED=0; TERMINAL_WRITTEN=0
CHILD_FILE="$CANDIDATE_ROOT/child.pid"
printf '0\n' >"$CHILD_FILE"
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" --child-pid "$CHILD_PID" --log "$MAIN_LOG" ${5:+--exit-code "$5"}
  case "$1" in REFERENCE|PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    local observed=0
    [[ -f "$CHILD_FILE" ]] && observed="$(<"$CHILD_FILE")"
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$observed" >/dev/null 2>&1 || true
    sleep 20
  done
}
on_signal() { STOP_REQUESTED=1; [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; }
cleanup() {
  local code=$?; kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then status STOPPED stopped run_r15_stack_screen.sh "graceful stop; outputs preserved" 130 || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then status FAILED failed run_r15_stack_screen.sh "pipeline exit=$code; inspect log" "$code" || true; fi
}
trap on_signal INT TERM; trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!
EXECUTION_ARGS=()
EXECUTION_DETAIL="W10 temporal ensemble decay=0.01"
case "$LABEL" in
  w12_balanced_decay_0p05)
    EXECUTION_ARGS=(--execution-mode balanced_temporal_ensemble)
    EXECUTION_DETAIL="balanced temporal ensemble decay=0.05"
    ;;
  w12_recent_decay_0p10)
    EXECUTION_ARGS=(--execution-mode recent_temporal_ensemble)
    EXECUTION_DETAIL="recent-weighted temporal ensemble decay=0.10"
    ;;
  w12_latest_chunk)
    EXECUTION_ARGS=(--execution-mode latest_chunk)
    EXECUTION_DETAIL="latest chunk first action; replan each step"
    ;;
esac
status VALIDATING closed_loop evaluate_action_generator_evolution.py "$SPLIT paired Stack screen; $EXECUTION_DETAIL"
(
  cd "$WORKTREE"
  exec env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$WORKTREE" BWA_R15_RUN_ROOT="$RUN_ROOT" BWA_R15_CANDIDATE="$CANDIDATE" \
    "$PYTHON" -m before_we_act.evaluate_action_generator_evolution --config "$CONFIG" --checkpoint "$CHECKPOINT" --belief-config "$BELIEF_CONFIG" --belief-checkpoint "$BELIEF_CHECKPOINT" --vision-artifact "$VISION_ARTIFACT" --vision-batch-size 5 --task three_robots_stack_cube --seed-file "$SEED_FILE" --episodes 20 --max-steps 1500 --device cuda:0 --output "$OUTPUT" --resume-log "$EVAL_LOG" "${EXECUTION_ARGS[@]}"
) >>"$EVAL_LOG" 2>&1 &
CHILD_PID=$!; printf '%s\n' "$CHILD_PID" >"$CHILD_FILE"
status VALIDATING closed_loop evaluate_action_generator_evolution.py "$SPLIT paired Stack screen; $EXECUTION_DETAIL"
wait "$CHILD_PID"; CHILD_PID=0; printf '0\n' >"$CHILD_FILE"
if ((REFERENCE == 0)); then
  status ACCEPTING paired_screen r15_runtime.py "waiting for identical-seed W12 reference"
  for _ in $(seq 1 4320); do [[ -f "$RUN_ROOT/candidates/p0/validation/$SPLIT.json" ]] && break; sleep 20; done
  [[ -f "$RUN_ROOT/candidates/p0/validation/$SPLIT.json" ]] || { printf 'W12 reference did not complete\n' >&2; exit 3; }
fi
set +e
"$PYTHON" "$RUNTIME" accept --run-root "$RUN_ROOT" --candidate "$CANDIDATE"
ACCEPT_CODE=$?
set -e
ACCEPT_STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CANDIDATE_ROOT/acceptance.json")"
if [[ "$ACCEPT_STATUS" == REFERENCE ]]; then
  status REFERENCE complete r15_runtime.py "W12 identical-seed reference complete; screen-only"
elif ((ACCEPT_CODE == 0)); then
  status PASSED complete r15_runtime.py "strict paired $SPLIT gain over W12"
else
  status FAILED complete r15_runtime.py "no strict paired $SPLIT gain over W12" "$ACCEPT_CODE"
fi
exit "$ACCEPT_CODE"
