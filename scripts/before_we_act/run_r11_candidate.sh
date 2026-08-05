#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
PARENT_CHECKPOINT=""
CACHE=/workspace/bwa_runs/shared/r11_observation_cache.pt
PYTHON=/venv/robofactory-act/bin/python
DATA_ROOT=/workspace/datasets/robofactory_multitask

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --parent-checkpoint) PARENT_CHECKPOINT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ || "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU must be p0/0 through p3/3\n' >&2
  exit 2
fi

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$FE_ROOT/scripts/before_we_act/r11_runtime.py"
CONFIG="$FE_ROOT/configs/before_we_act/r11_belief/${CANDIDATE}.yaml"
LOCK="$FE_ROOT/experiments/before_we_act/r11/${CANDIDATE}/component_lock.yaml"
PARITY="$FE_ROOT/experiments/before_we_act/r11/${CANDIDATE}/parity.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"
RECEIPTS="$CANDIDATE_ROOT/receipts"
UPSTREAM="/workspace/bwa_upstream/r11/$CANDIDATE"
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
if [[ "$(git -C "$FE_ROOT" branch --show-current)" != "$EXPECTED_BRANCH" || "$(git -C "$FE_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  printf 'candidate worktree branch/commit differs from run manifest\n' >&2
  exit 3
fi
for path in "$PYTHON" "$RUNTIME" "$CONFIG" "$LOCK" "$PARITY" "$PARENT_CHECKPOINT" "$DATA_ROOT"; do
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
    status STOPPED stopped run_r11_candidate.sh "graceful stop requested; latest checkpoint preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then
    status FAILED failed run_r11_candidate.sh "pipeline exited with code $code; inspect receipts and log" || true
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

if ! "$PYTHON" -c 'import timm' >/dev/null 2>&1; then
  run_child PREPARING dependencies uv "install pinned V-JEPA2 dependency into the existing venv" 10000 \
    uv pip install --python "$PYTHON" -r "$FE_ROOT/requirements/r11-p0.txt"
fi

status PREPARING cache_wait run_r11_candidate.sh "waiting for shared legal-input cache" 10000
for _ in $(seq 1 3600); do
  [[ -f "$CACHE" ]] && break
  sleep 10
done
[[ -f "$CACHE" ]] || { printf 'shared R11 cache did not appear\n' >&2; exit 3; }

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
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/fetch_upstream_readonly.py" --repo "$REPO" --commit "$UPSTREAM_COMMIT" --destination "$UPSTREAM"
run_child PREPARING source_verify verify_upstream_source.py "official repo/commit/clean checkout" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_upstream_source.py" --lock "$LOCK" --upstream "$UPSTREAM" --output "$RECEIPTS/source.json"
run_child PREPARING license verify_component_license.py "license preserved" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_component_license.py" --lock "$LOCK" --project-root "$FE_ROOT" --output "$RECEIPTS/license.json"
run_child PREPARING patch audit_component_patch.py "minimal transplant patch audit" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_component_patch.py" --lock "$LOCK" --upstream "$UPSTREAM" --project-root "$FE_ROOT" --patch-output "$RECEIPTS/upstream_adaptation.patch" --report-output "$RECEIPTS/patch.json"
run_child PREPARING dependency audit_no_full_repo_dependency.py "no full upstream runtime import" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_no_full_repo_dependency.py" --project-root "$FE_ROOT" --output "$RECEIPTS/dependency.json"
run_child PREPARING off_path classify_action_effect.py "classify branch strictly off-path" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/classify_action_effect.py" --parent "$PARENT_COMMIT" --head HEAD --output "$RECEIPTS/action_effect.json"
run_child PREPARING parity parity.py "official/local component numerical parity" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$PARITY" --upstream "$UPSTREAM" --output "$RECEIPTS/parity.json" --device cuda:0

run_child TRAINING preflight train_team_belief.py "two-update train/save test" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_team_belief --config "$CONFIG" --cache "$CACHE" --output "$CANDIDATE_ROOT/preflight" --device cuda:0 --updates 2
run_child VALIDATING preflight_restore verify_r11_preflight.py "strict restore and finite replay" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_r11_preflight.py" --config "$CONFIG" --cache "$CACHE" --checkpoint "$CANDIDATE_ROOT/preflight/checkpoints/checkpoint_000002.pt" --device cuda:0 --output "$RECEIPTS/preflight.json"

FORMAL="$CANDIDATE_ROOT/train/formal"
run_child TRAINING formal train_team_belief.py "frozen common 10000-update representation screen" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_team_belief --config "$CONFIG" --cache "$CACHE" --output "$FORMAL" --device cuda:0
CHECKPOINT="$FORMAL/checkpoints/checkpoint_010000.pt"
run_child VALIDATING representation_screen evaluate_team_belief.py "frozen validation windows and score" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.evaluate_team_belief --config "$CONFIG" --cache "$CACHE" --checkpoint "$CHECKPOINT" --output "$CANDIDATE_ROOT/validation/representation_screen.json" --device cuda:0
run_child ACCEPTING action_hash audit_r11_action_hash.py "five-task W10 exact action hash" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/audit_r11_action_hash.py" --config "$CONFIG" --belief-checkpoint "$CHECKPOINT" --parent-checkpoint "$PARENT_CHECKPOINT" --data-root "$DATA_ROOT" --device cuda:0 --output "$CANDIDATE_ROOT/validation/action_hash.json"

set +e
run_child ACCEPTING special_acceptance accept_r11.py "nine validity hard gates; quality has no threshold" 10000 \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/accept_r11.py" --candidate "$CANDIDATE" --source "$RECEIPTS/source.json" --license "$RECEIPTS/license.json" --patch "$RECEIPTS/patch.json" --dependency "$RECEIPTS/dependency.json" --parity "$RECEIPTS/parity.json" --preflight "$RECEIPTS/preflight.json" --screen "$CANDIDATE_ROOT/validation/representation_screen.json" --action-hash "$CANDIDATE_ROOT/validation/action_hash.json" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
if ((ACCEPT_CODE == 0)); then
  SCORE="$($PYTHON - "$CANDIDATE_ROOT/acceptance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["representation_screen_score"])
PY
)"
  status PASSED complete accept_r11.py "all R11 validity gates passed; screen score=$SCORE" 10000
else
  status FAILED complete accept_r11.py "one or more R11 validity gates failed" 10000
fi
exit "$ACCEPT_CODE"
