#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
WORKTREE=""
ATTEMPT=1
ACTION_CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
CACHE=/workspace/bwa_runs/shared/r13/world_cache_v1.pt
PYTHON=/venv/robofactory-act/bin/python
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --worktree) WORKTREE="$2"; shift 2 ;;
    --attempt) ATTEMPT="$2"; shift 2 ;;
    --action-checkpoint) ACTION_CHECKPOINT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2; exit 2
fi
[[ "$ATTEMPT" =~ ^[1-9][0-9]*$ ]] || { printf 'attempt must be a positive integer\n' >&2; exit 2; }
[[ -n "$RUN_ROOT" && -n "$WORKTREE" ]] || { printf 'run root and worktree are required\n' >&2; exit 2; }
BASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$BASE_ROOT/scripts/before_we_act/r13_runtime.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
CONFIG="$WORKTREE/configs/before_we_act/r13_world/$CANDIDATE.yaml"
CHECKPOINT="$CANDIDATE_ROOT/train/formal/checkpoints/checkpoint_010000.pt"
SCREEN="$CANDIDATE_ROOT/validation/world_screen.json"
ACTION_HASH="$CANDIDATE_ROOT/validation/action_hash.json"
ACCEPTANCE="$CANDIDATE_ROOT/acceptance.json"
if ((ATTEMPT == 1)); then
  RECOVERY="$CANDIDATE_ROOT/receipts/validation_recovery.json"
  LOG="$CANDIDATE_ROOT/logs/validation_recovery.log"
else
  RECOVERY="$CANDIDATE_ROOT/receipts/validation_recovery_v${ATTEMPT}.json"
  LOG="$CANDIDATE_ROOT/logs/validation_recovery_v${ATTEMPT}.log"
fi
for path in "$PYTHON" "$MANIFEST" "$WORKTREE" "$CONFIG" "$CHECKPOINT" "$ACTION_CHECKPOINT" "$CACHE"; do
  [[ -e "$path" ]] || { printf 'missing recovery input: %s\n' "$path" >&2; exit 3; }
done
for path in "$SCREEN" "$ACTION_HASH" "$ACCEPTANCE" "$RECOVERY"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite recovery output: %s\n' "$path" >&2; exit 3; }
done
EXPECTED_BRANCH="$(jq -r --arg c "$CANDIDATE" '.branches[$c]' "$MANIFEST")"
TRAINING_COMMIT="$(jq -r --arg c "$CANDIDATE" '.commits[$c]' "$MANIFEST")"
[[ "$(git -C "$WORKTREE" branch --show-current)" == "$EXPECTED_BRANCH" ]] || { printf 'recovery branch differs\n' >&2; exit 3; }
git -C "$WORKTREE" merge-base --is-ancestor "$TRAINING_COMMIT" HEAD || { printf 'validation fix does not descend from training commit\n' >&2; exit 3; }
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || { printf 'recovery worktree is dirty\n' >&2; exit 3; }
mkdir -p "$(dirname "$SCREEN")" "$(dirname "$RECOVERY")" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

CHILD_PID=0
HEARTBEAT_PID=0
TERMINAL_WRITTEN=0
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" \
    --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" \
    --child-pid "$CHILD_PID" --total-updates 10000 --checkpoint "$CHECKPOINT" \
    --best-checkpoint "$CHECKPOINT" --log "$LOG"
  case "$1" in PASSED|FAILED|STOPPED) TERMINAL_WRITTEN=1 ;; esac
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --pid "$$" --child-pid "$CHILD_PID" >/dev/null 2>&1 || true
    sleep 20
  done
}
cleanup() {
  code=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((code != 0 && TERMINAL_WRITTEN == 0)); then
    status FAILED validation_recovery recover_r13_validation.sh "validation recovery exited with code $code" || true
  fi
}
trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!

run_child() {
  local state="$1" stage="$2" program="$3" detail="$4"
  status "$state" "$stage" "$program" "$detail"
  shift 4
  "$@" & CHILD_PID=$!
  status "$state" "$stage" "$program" "$detail"
  wait "$CHILD_PID"
  CHILD_PID=0
}

VALIDATION_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
"$PYTHON" - "$RECOVERY" "$TRAINING_COMMIT" "$VALIDATION_COMMIT" "$ATTEMPT" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "round": "R13",
    "reason": "shared evaluator latent persistence baseline omitted the target-token axis",
    "training_commit": sys.argv[2],
    "validation_fix_commit": sys.argv[3],
    "attempt": int(sys.argv[4]),
    "training_reused": True,
    "checkpoint_overwritten": False,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
# Python prepends the current working directory to sys.path.  Entering the
# candidate worktree is therefore required in addition to PYTHONPATH; otherwise
# the common base branch's intentionally empty candidate registry wins.
cd "$WORKTREE"
run_child VALIDATING validation_recovery evaluate_team_world.py "evaluate completed 10000-update checkpoint with axis fix" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$WORKTREE" "$PYTHON" -m before_we_act.evaluate_team_world --config "$CONFIG" --cache "$CACHE" --checkpoint "$CHECKPOINT" --output "$SCREEN" --device cuda:0
run_child ACCEPTING action_hash audit_r13_action_hash.py "frozen W12 proposal/checkpoint exact hash" \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$WORKTREE" "$PYTHON" "$WORKTREE/scripts/before_we_act/audit_r13_action_hash.py" --config "$CONFIG" --cache "$CACHE" --checkpoint "$CHECKPOINT" --action-checkpoint "$ACTION_CHECKPOINT" --device cuda:0 --output "$ACTION_HASH"
set +e
run_child ACCEPTING special_acceptance accept_r13.py "authoritative R13 hard gates" \
  "$PYTHON" "$WORKTREE/scripts/before_we_act/accept_r13.py" --candidate "$CANDIDATE" --source "$CANDIDATE_ROOT/receipts/source.json" --license "$CANDIDATE_ROOT/receipts/license.json" --patch "$CANDIDATE_ROOT/receipts/patch.json" --dependency "$CANDIDATE_ROOT/receipts/dependency.json" --action-effect "$CANDIDATE_ROOT/receipts/action_effect.json" --parity "$CANDIDATE_ROOT/receipts/parity.json" --preflight "$CANDIDATE_ROOT/receipts/preflight.json" --screen "$SCREEN" --action-hash "$ACTION_HASH" --cache "$RUN_ROOT/shared/cache.json" --output "$ACCEPTANCE"
code=$?
set -e
if ((code == 0)); then
  score="$(jq -r '.world_screen_score' "$ACCEPTANCE")"
  status PASSED complete accept_r13.py "validation recovery passed all hard gates; world score=$score"
else
  status FAILED complete accept_r13.py "validation recovery completed; one or more hard gates failed"
fi
exit "$code"
