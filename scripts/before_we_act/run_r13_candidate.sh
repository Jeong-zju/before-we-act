#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
BELIEF_CHECKPOINT=""
ACTION_CHECKPOINT=""
CACHE=/workspace/bwa_runs/shared/r13_world_cache.pt
PYTHON=/venv/robofactory-act/bin/python
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift 2 ;;
    --action-checkpoint) ACTION_CHECKPOINT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/scripts/before_we_act/r13_runtime.py"
CONFIG="$ROOT/configs/before_we_act/r13_world/${CANDIDATE}.yaml"
LOCK="$ROOT/experiments/before_we_act/r13/${CANDIDATE}/component_lock.yaml"
PARITY="$ROOT/experiments/before_we_act/r13/${CANDIDATE}/parity.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"
RECEIPTS="$CANDIDATE_ROOT/receipts"
CACHE_RECEIPT="$RUN_ROOT/shared/cache.json"
UPSTREAM="/workspace/bwa_upstream/r13/$CANDIDATE"
mkdir -p "$LOG_ROOT" "$RECEIPTS" "$CANDIDATE_ROOT/preflight" "$CANDIDATE_ROOT/train/formal" "$CANDIDATE_ROOT/validation"
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
PARENT_COMMIT="$($PYTHON - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["parent_commit"])
PY
)"
if [[ "$(git -C "$ROOT" branch --show-current)" != "$EXPECTED_BRANCH" || "$(git -C "$ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  printf 'candidate worktree branch/commit differs from run manifest\n' >&2
  exit 3
fi
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$LOCK" "$PARITY" "$BELIEF_CHECKPOINT" "$ACTION_CHECKPOINT"; do
  [[ -e "$path" ]] || { printf 'missing required path: %s\n' "$path" >&2; exit 3; }
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
    status STOPPED stopped run_r13_candidate.sh "graceful stop requested; latest checkpoint preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then
    status FAILED failed run_r13_candidate.sh "pipeline exited with code $code; inspect receipts and log" || true
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

REQ="$ROOT/requirements/r13-${CANDIDATE}.txt"
if [[ -s "$REQ" ]]; then
  run_child PREPARING dependencies uv "install candidate-pinned runtime dependencies" 10000 \
    uv pip install --python "$PYTHON" -r "$REQ"
fi

status PREPARING cache_wait run_r13_candidate.sh "waiting for shared W11+W12 legal-input cache" 10000
for _ in $(seq 1 3600); do
  [[ -f "$CACHE" && -f "$CACHE_RECEIPT" ]] && break
  sleep 10
done
[[ -f "$CACHE" && -f "$CACHE_RECEIPT" ]] || { printf 'shared R13 cache did not appear\n' >&2; exit 3; }

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
run_child DOWNLOADING source_fetch fetch_upstream_readonly.py "verify official pinned source" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/fetch_upstream_readonly.py" --repo "$REPO" --commit "$UPSTREAM_COMMIT" --destination "$UPSTREAM"
run_child PREPARING source_verify verify_upstream_source.py "official repo/commit/clean checkout" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/verify_upstream_source.py" --lock "$LOCK" --upstream "$UPSTREAM" --output "$RECEIPTS/source.json"
run_child PREPARING license verify_component_license.py "license preserved" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/verify_component_license.py" --lock "$LOCK" --project-root "$ROOT" --output "$RECEIPTS/license.json"
run_child PREPARING patch audit_component_patch.py "minimal transplant patch audit" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/audit_component_patch.py" --lock "$LOCK" --upstream "$UPSTREAM" --project-root "$ROOT" --patch-output "$RECEIPTS/upstream_adaptation.patch" --report-output "$RECEIPTS/patch.json"
run_child PREPARING dependency audit_no_full_repo_dependency.py "no full upstream runtime import" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/audit_no_full_repo_dependency.py" --project-root "$ROOT" --output "$RECEIPTS/dependency.json"
run_child PREPARING off_path classify_r13_action_effect.py "prove world path cannot affect W12 action" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/classify_r13_action_effect.py" --parent "$PARENT_COMMIT" --head HEAD --output "$RECEIPTS/action_effect.json"
run_child PREPARING parity parity.py "official/local component numerical parity" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$PARITY" --upstream "$UPSTREAM" --output "$RECEIPTS/parity.json" --device cuda:0

run_child TRAINING preflight train_team_world.py "two-update train/save test" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" -m before_we_act.train_team_world --config "$CONFIG" --cache "$CACHE" --output "$CANDIDATE_ROOT/preflight" --device cuda:0 --updates 2 --heartbeat "$CANDIDATE_ROOT/heartbeat.json"
run_child VALIDATING preflight_restore verify_r13_preflight.py "strict restore, action effect and future-input rejection" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/before_we_act/verify_r13_preflight.py" --config "$CONFIG" --cache "$CACHE" --checkpoint "$CANDIDATE_ROOT/preflight/checkpoints/checkpoint_000002.pt" --device cuda:0 --output "$RECEIPTS/preflight.json"

FORMAL="$CANDIDATE_ROOT/train/formal"
RESUME_ARGS=()
[[ -f "$FORMAL/checkpoints/checkpoint_latest.pt" ]] && RESUME_ARGS=(--resume "$FORMAL/checkpoints/checkpoint_latest.pt")
run_child TRAINING formal train_team_world.py "frozen common 10000-update world screen" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" -m before_we_act.train_team_world --config "$CONFIG" --cache "$CACHE" --output "$FORMAL" --device cuda:0 --heartbeat "$CANDIDATE_ROOT/heartbeat.json" "${RESUME_ARGS[@]}"
CHECKPOINT="$FORMAL/checkpoints/checkpoint_010000.pt"
run_child VALIDATING world_screen evaluate_team_world.py "frozen validation windows and world score" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" -m before_we_act.evaluate_team_world --config "$CONFIG" --cache "$CACHE" --checkpoint "$CHECKPOINT" --output "$CANDIDATE_ROOT/validation/world_screen.json" --device cuda:0
run_child ACCEPTING action_hash audit_r13_action_hash.py "frozen W12 proposal/checkpoint exact hash" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/before_we_act/audit_r13_action_hash.py" --config "$CONFIG" --cache "$CACHE" --checkpoint "$CHECKPOINT" --action-checkpoint "$ACTION_CHECKPOINT" --device cuda:0 --output "$CANDIDATE_ROOT/validation/action_hash.json"

set +e
run_child ACCEPTING special_acceptance accept_r13.py "R13 validity hard gates; quality has no threshold" 10000 \
  "$PYTHON" "$ROOT/scripts/before_we_act/accept_r13.py" --candidate "$CANDIDATE" --source "$RECEIPTS/source.json" --license "$RECEIPTS/license.json" --patch "$RECEIPTS/patch.json" --dependency "$RECEIPTS/dependency.json" --action-effect "$RECEIPTS/action_effect.json" --parity "$RECEIPTS/parity.json" --preflight "$RECEIPTS/preflight.json" --screen "$CANDIDATE_ROOT/validation/world_screen.json" --action-hash "$CANDIDATE_ROOT/validation/action_hash.json" --cache "$CACHE_RECEIPT" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
if ((ACCEPT_CODE == 0)); then
  SCORE="$($PYTHON - "$CANDIDATE_ROOT/acceptance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["world_screen_score"])
PY
)"
  status PASSED complete accept_r13.py "all R13 validity gates passed; world score=$SCORE" 10000
else
  status FAILED complete accept_r13.py "one or more R13 validity gates failed" 10000
fi
if [[ -f "$RUN_ROOT/candidates/p0/acceptance.json" && -f "$RUN_ROOT/candidates/p1/acceptance.json" && -f "$RUN_ROOT/candidates/p2/acceptance.json" && -f "$RUN_ROOT/candidates/p3/acceptance.json" ]]; then
  (
    flock -x 9
    "$PYTHON" "$ROOT/scripts/before_we_act/decide_r13_winner.py" \
      --run-root "$RUN_ROOT" --output "$RUN_ROOT/winner_pack.json" || true
  ) 9>"$RUN_ROOT/.winner.lock"
fi
exit "$ACCEPT_CODE"
