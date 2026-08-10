#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r11-four-way-v1
CANDIDATE=""
GPU_INDEX=""
BASE_PYTHON=/venv/robofactory-act/bin/python
WORKERS=8

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --base-python) BASE_PYTHON="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$CANDIDATE" =~ ^[A-D]$ ]] || { printf 'candidate A, B, C, or D is required\n' >&2; exit 2; }
[[ "$GPU_INDEX" =~ ^[0-3]$ ]] || { printf 'GPU index 0..3 is required\n' >&2; exit 2; }
[[ $(( $(printf '%d' "'$CANDIDATE") - 65 )) == "$GPU_INDEX" ]] || {
  printf 'candidate/GPU mapping differs: %s/%s\n' "$CANDIDATE" "$GPU_INDEX" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/scripts/before_we_act/r11_runtime.py"
TRAINER="$ROOT/before_we_act/train_r11_candidate.py"
CAUSAL="$ROOT/before_we_act/evaluate_r11_causality.py"
SUITE="$ROOT/scripts/before_we_act/evaluate_r11_suite.py"
SUMMARIZE="$ROOT/scripts/before_we_act/summarize_r11_validation.py"
HARD_SUMMARIZE="$ROOT/scripts/before_we_act/summarize_r11_hard_task_gate.py"
DISCOVERY_GATE="$ROOT/scripts/before_we_act/check_r11_discovery_gate.py"
SELECTION_GATE="$ROOT/scripts/before_we_act/check_r11_selection_gate.py"
ACCEPTOR="$ROOT/scripts/before_we_act/accept_r11_candidate.py"
FAIL_WRITER="$ROOT/scripts/before_we_act/fail_r11_candidate.py"
PREFLIGHT="$ROOT/scripts/before_we_act/preflight_r11_candidate.py"
MANIFEST="$RUN_ROOT/run_manifest.json"
BASELINE_PROVENANCE="$RUN_ROOT/preflight/baseline_provenance.json"
BASELINE_CHECKPOINT=/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt
SEED_ROOT=/workspace/bwa_runs/w10-six-task-v1/seeds/validation
W10_LATENCY="$RUN_ROOT/preflight/w10_latency.json"
CANDIDATE_ROOT="$RUN_ROOT/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/pipeline.log"
VENV_ROOT="$CANDIDATE_ROOT/venv"

declare -A CONFIGS=(
  [A]=configs/before_we_act/r11/a-vjepa21-ac-refine.json
  [B]=configs/before_we_act/r11/b-dreamzero-wan22-wam.json
  [C]=configs/before_we_act/r11/c-cosmos-policy-latent.json
  [D]=configs/before_we_act/r11/d-lawam-latent-subgoal.json
)
declare -A ASSET_SCRIPTS=(
  [A]=scripts/before_we_act/prepare_r11_a_assets.sh
  [B]=scripts/before_we_act/prepare_r11_b_assets.sh
  [C]=scripts/before_we_act/prepare_r11_c_assets.sh
  [D]=scripts/before_we_act/prepare_r11_d_assets.sh
)
declare -A BRANCHES=(
  [A]=feat/r11-vjepa21-ac-refine
  [B]=feat/r11-dreamzero-wan22-wam
  [C]=feat/r11-cosmos-policy-latent
  [D]=feat/r11-lawam-latent-subgoal
)
declare -A UPSTREAMS=(
  [A]=204698b45b3712590f06245fbfba32d3be539812
  [B]=ab790c198fbce33503358efbbd4187ce9a89adf3
  [C]=a2c298b0a3df3778b973fe65e9e58877b292d8a7
  [D]=4ea6fdadce6c9b8746028307a246b79ee2c4fd55
)

CONFIG="$ROOT/${CONFIGS[$CANDIDATE]}"
ASSET_SCRIPT="$ROOT/${ASSET_SCRIPTS[$CANDIDATE]}"
BRANCH="${BRANCHES[$CANDIDATE]}"
UPSTREAM="${UPSTREAMS[$CANDIDATE]}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
WRAPPER_PID="$$"
WRAPPER_START="$(awk '{print $22}' "/proc/$$/stat")"
CHILD_PID=0
CHILD_START=0
CHILD_KIND=other
WATCHDOG_PID=0
STOP_REQUESTED=0
CURRENT_STAGE=setup
TERMINAL_WRITTEN=0

mkdir -p "$LOG_ROOT" "$CANDIDATE_ROOT/status" "$CANDIDATE_ROOT/train" \
  "$CANDIDATE_ROOT/validation" "$CANDIDATE_ROOT/causal" "$CANDIDATE_ROOT/preflight"
exec > >(tee -a "$MAIN_LOG") 2>&1

status() {
  "$BASE_PYTHON" "$RUNTIME" status \
    --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --state "$1" --stage "$2" \
    --program "$3" --detail "$4" --branch "$BRANCH" --commit "$COMMIT" \
    --upstream-commit "$UPSTREAM" --pid "$WRAPPER_PID" \
    --pid-start-time-ticks "$WRAPPER_START" --child-pid "$CHILD_PID" \
    --child-pid-start-time-ticks "$CHILD_START" --log "$MAIN_LOG" ${5:+--exit-code "$5"}
}

identity_alive() {
  local pid="$1" expected="$2" observed
  [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/stat" ]] || return 1
  observed="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$observed" == "$expected" ]]
}

watch_child() {
  local pid="$1" start="$2" stage="$3"
  while identity_alive "$pid" "$start"; do
    "$BASE_PYTHON" "$RUNTIME" watchdog --run-root "$RUN_ROOT" \
      --candidate "$CANDIDATE" --stage "$stage" --pid "$pid" \
      --pid-start-time-ticks "$start" >/dev/null 2>&1 || break
    sleep 20
  done
}

on_signal() {
  STOP_REQUESTED=1
  if identity_alive "$CHILD_PID" "$CHILD_START"; then
    if [[ "$CHILD_KIND" == trainer ]]; then
      kill -USR1 "$CHILD_PID" 2>/dev/null || true
    else
      kill -INT "$CHILD_PID" 2>/dev/null || true
    fi
  fi
}

cleanup() {
  local code=$?
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then
    status STOPPED "$CURRENT_STAGE" run_r11_candidate.sh \
      "graceful stop completed; all produced artifacts are preserved" 130 || true
    return
  fi
  if ((code != 0 && TERMINAL_WRITTEN == 0)); then
    local state=FAILED
    if grep -Fqi 'CUDA out of memory' "$MAIN_LOG"; then state=FAILED_FIT; fi
    "$BASE_PYTHON" "$FAIL_WRITER" --candidate "$CANDIDATE" --branch "$BRANCH" \
      --commit "$COMMIT" --failed-stage "$CURRENT_STAGE" \
      --reason "pipeline exited nonzero at $CURRENT_STAGE; inspect the preserved candidate log" \
      --exit-code "$code" --output "$CANDIDATE_ROOT/acceptance.json" || true
    status "$state" "$CURRENT_STAGE" run_r11_candidate.sh \
      "pipeline exited with code $code; see structured acceptance and log" "$code" || true
  fi
}
trap on_signal USR1 INT TERM
trap cleanup EXIT

run_child() {
  local state="$1" stage="$2" program="$3" detail="$4" kind="$5"
  shift 5
  CURRENT_STAGE="$stage"
  CHILD_KIND="$kind"
  CHILD_PID=0
  CHILD_START=0
  status "$state" "$stage" "$program" "$detail"
  "$@" &
  CHILD_PID=$!
  CHILD_START="$(awk '{print $22}' "/proc/$CHILD_PID/stat")"
  status "$state" "$stage" "$program" "$detail"
  watch_child "$CHILD_PID" "$CHILD_START" "$stage" &
  WATCHDOG_PID=$!
  local code=0
  wait "$CHILD_PID" || code=$?
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=0
  CHILD_PID=0
  CHILD_START=0
  CHILD_KIND=other
  if ((STOP_REQUESTED)); then return 130; fi
  return "$code"
}

[[ -x "$BASE_PYTHON" && -f "$MANIFEST" && -f "$BASELINE_PROVENANCE" ]] || {
  printf 'missing frozen Python/run-manifest/baseline provenance\n' >&2
  exit 3
}
[[ "$(git -C "$ROOT" branch --show-current)" == "$BRANCH" && -z "$(git -C "$ROOT" status --porcelain)" ]] || {
  printf 'candidate worktree branch/clean identity differs\n' >&2
  exit 3
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$GPU_INDEX" ]] || {
  printf 'worker CUDA_VISIBLE_DEVICES differs from physical assignment\n' >&2
  exit 3
}
[[ -f "$CONFIG" && -x "$ASSET_SCRIPT" ]] || {
  printf 'candidate config or asset script is missing\n' >&2
  exit 3
}
mapfile -t MANIFESTS < <(find /workspace/datasets/robofactory_multitask -mindepth 2 -maxdepth 2 -name training_manifest.json -type f | sort)
[[ ${#MANIFESTS[@]} -eq 6 ]] || { printf 'expected six training manifests\n' >&2; exit 3; }

status PREPARING dependencies run_r11_candidate.sh "creating isolated candidate environment"
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  run_child PREPARING dependencies python-venv "isolated venv with inherited CUDA runtime" other \
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_ROOT"
fi
PYTHON="$VENV_ROOT/bin/python"
mapfile -t REQUIREMENTS < <(find "$ROOT/requirements/r11" -maxdepth 1 -type f -name "${CANDIDATE,,}-*.txt" -print)
[[ ${#REQUIREMENTS[@]} -eq 1 ]] || { printf 'expected one candidate requirements file\n' >&2; exit 3; }
run_child PREPARING dependencies pip "installing branch-local upstream dependency closure" other \
  "$PYTHON" -m pip install --disable-pip-version-check --cache-dir /workspace/.cache/pip \
    -r "${REQUIREMENTS[0]}"

"$PYTHON" -m pip freeze >"$CANDIDATE_ROOT/status/pip_freeze.txt"
status DOWNLOADING foundation "$(basename "$ASSET_SCRIPT")" \
  "verifying or resuming only this candidate's pinned foundation files"
run_child DOWNLOADING foundation "$(basename "$ASSET_SCRIPT")" \
  "pinned revision download and immutable hash receipt" other \
  env HF_HOME=/workspace/.cache/huggingface \
    R11_HF_TOKEN_FILE="$RUN_ROOT/secrets/hf_token" \
    PYTHON_BIN="$BASE_PYTHON" "$ASSET_SCRIPT"

run_child PREFLIGHT F0 preflight_r11_candidate.py \
  "branch/source/license/foundation hashes and cross-candidate diff" other \
  env PYTHONPATH="$ROOT" "$PYTHON" "$PREFLIGHT" --candidate "$CANDIDATE" \
    --worktree "$ROOT" --config "$CONFIG" --run-manifest "$MANIFEST" \
    --output "$CANDIDATE_ROOT/preflight/candidate.json"
run_child PREFLIGHT F0 pytest \
  "official fixed-tensor parity plus RoboFactory adapter/common runtime tests" other \
  env PYTHONPATH="$ROOT" "$PYTHON" -m pytest -q \
    "$ROOT"/tests/before_we_act/test_r11_*.py

train_stage() {
  local stage="$1" target="$2" destination="$3" resume="$4" smoke="$5"
  local arguments=(
    "$PYTHON" "$TRAINER" --config "$CONFIG" --manifests "${MANIFESTS[@]}"
    --run-manifest "$MANIFEST" --baseline-provenance "$BASELINE_PROVENANCE"
    --baseline-checkpoint "$BASELINE_CHECKPOINT" --output "$destination"
    --stage "$stage" --updates "$target" --workers "$WORKERS"
    --device cuda:0 --heartbeat "$CANDIDATE_ROOT/status/worker.json"
    --status "$CANDIDATE_ROOT/status/worker_status.json"
  )
  [[ -z "$resume" ]] || arguments+=(--resume "$resume")
  [[ "$smoke" == 1 ]] && arguments+=(--smoke-modes)
  run_child TRAINING "$stage" train_r11_candidate.py \
    "exact six-task update target=$target effective_batch=48" trainer \
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R11_RUN_ROOT="$RUN_ROOT" \
      BWA_R11_CANDIDATE="$CANDIDATE" PYTHONPATH="$ROOT" "${arguments[@]}" \
      >>"$LOG_ROOT/train_${stage}_${target}.log" 2>&1
}

checkpoint_sha() {
  "$BASE_PYTHON" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sha256"])
PY
}

run_suite() {
  local phase="$1" mode="$2" checkpoint="$3" digest="$4" episodes="$5"
  shift 5
  local tasks=("$@")
  local output="$CANDIDATE_ROOT/validation/$phase/$mode"
  local log="$LOG_ROOT/validation_${phase}_${mode}.log"
  mkdir -p "$output"
  run_child VALIDATING "$phase-$mode" evaluate_r11_suite.py \
    "fixed seeds tasks=${tasks[*]} episodes=$episodes" other \
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R11_RUN_ROOT="$RUN_ROOT" \
      BWA_R11_CANDIDATE="$CANDIDATE" PYTHONPATH="$ROOT:/workspace/RoboFactory" \
      "$PYTHON" "$SUITE" --checkpoint "$checkpoint" --checkpoint-sha256 "$digest" \
      --seed-root "$SEED_ROOT" --tasks "${tasks[@]}" --episodes "$episodes" \
      --mode "$mode" --device cuda:0 --resume-log "$log" --output-root "$output" \
      >>"$log" 2>&1
}

summarize_suite() {
  local phase="$1" mode="$2" digest="$3" episodes="$4"
  shift 4
  local tasks=("$@")
  run_child VALIDATING "$phase-summary" summarize_r11_validation.py \
    "fail-closed closed-loop aggregation" other \
    env PYTHONPATH="$ROOT" "$PYTHON" "$SUMMARIZE" \
      --input-root "$CANDIDATE_ROOT/validation/$phase/$mode" \
      --tasks "${tasks[@]}" --episodes "$episodes" --mode "$mode" \
      --checkpoint-sha256 "$digest" \
      --output "$CANDIDATE_ROOT/validation/$phase/${mode}_summary.json"
}

run_causal() {
  local phase="$1" checkpoint="$2" digest="$3"
  mkdir -p "$CANDIDATE_ROOT/causal/$phase"
  run_child ACCEPTING "$phase-causal" evaluate_r11_causality.py \
    "32 validation samples/task; persistence/action-shuffle/prediction interventions" other \
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R11_RUN_ROOT="$RUN_ROOT" \
      BWA_R11_CANDIDATE="$CANDIDATE" PYTHONPATH="$ROOT" \
      "$PYTHON" "$CAUSAL" --checkpoint "$checkpoint" --checkpoint-sha256 "$digest" \
      --manifests "${MANIFESTS[@]}" --device cuda:0 --workers "$WORKERS" \
      --output "$CANDIDATE_ROOT/causal/$phase/summary.json" \
      >>"$LOG_ROOT/causal_${phase}.log" 2>&1
}

run_hard_gate() {
  local phase="$1" checkpoint="$2" digest="$3"
  local hard_tasks=(camera_alignment place_food)
  run_suite "$phase-hard" normal "$checkpoint" "$digest" 5 "${hard_tasks[@]}"
  run_suite "$phase-hard" prediction_off "$checkpoint" "$digest" 5 "${hard_tasks[@]}"
  run_suite "$phase-hard" prediction_shuffled "$checkpoint" "$digest" 5 "${hard_tasks[@]}"
  run_child ACCEPTING "$phase-hard-gate" summarize_r11_hard_task_gate.py \
    "same checkpoint/text/current observations and first five frozen seeds" other \
    env PYTHONPATH="$ROOT" "$PYTHON" "$HARD_SUMMARIZE" \
      --normal-root "$CANDIDATE_ROOT/validation/$phase-hard/normal" \
      --prediction-off-root "$CANDIDATE_ROOT/validation/$phase-hard/prediction_off" \
      --prediction-shuffled-root "$CANDIDATE_ROOT/validation/$phase-hard/prediction_shuffled" \
      --checkpoint-sha256 "$digest" \
      --output "$CANDIDATE_ROOT/causal/$phase/hard_task_gate.json"
}

ALL_TASKS=(
  lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food
)

F1_FRESH="$CANDIDATE_ROOT/preflight/f1_fresh"
F1_RESUME="$CANDIDATE_ROOT/preflight/f1_resume"
train_stage f1 1 "$F1_FRESH" "" 0
F1_ONE="$F1_FRESH/checkpoints/checkpoint_latest.pt"
train_stage f1 2 "$F1_RESUME" "$F1_ONE" 1
F1_TWO="$F1_RESUME/checkpoints/checkpoint_latest.pt"

DISCOVERY="$CANDIDATE_ROOT/train/discovery"
train_stage discovery 1000 "$DISCOVERY" "$F1_TWO" 0
DISCOVERY_CHECKPOINT="$DISCOVERY/checkpoints/checkpoint_001000.pt"
DISCOVERY_SHA="$(checkpoint_sha "$DISCOVERY/checkpoints/checkpoint_latest.receipt.json")"
run_causal discovery "$DISCOVERY_CHECKPOINT" "$DISCOVERY_SHA"
run_child ACCEPTING discovery-gate check_r11_discovery_gate.py \
  "action loss decreases and both causal directions are positive" other \
  env PYTHONPATH="$ROOT" "$PYTHON" "$DISCOVERY_GATE" \
    --progress "$F1_FRESH/progress.jsonl" "$F1_RESUME/progress.jsonl" "$DISCOVERY/progress.jsonl" \
    --causal "$CANDIDATE_ROOT/causal/discovery/summary.json" \
    --output "$CANDIDATE_ROOT/causal/discovery/gate.json"
run_suite discovery normal "$DISCOVERY_CHECKPOINT" "$DISCOVERY_SHA" 5 "${ALL_TASKS[@]}"
summarize_suite discovery normal "$DISCOVERY_SHA" 5 "${ALL_TASKS[@]}"

SELECTION="$CANDIDATE_ROOT/train/selection"
train_stage selection 20000 "$SELECTION" "$DISCOVERY_CHECKPOINT" 0
SELECTION_CHECKPOINT="$SELECTION/checkpoints/checkpoint_020000.pt"
SELECTION_SHA="$(checkpoint_sha "$SELECTION/checkpoints/checkpoint_latest.receipt.json")"
run_causal selection "$SELECTION_CHECKPOINT" "$SELECTION_SHA"
OFFLINE_PASS="$($PYTHON - "$CANDIDATE_ROOT/causal/selection/summary.json" <<'PY'
import json, sys
checks = {row["id"]: row for row in json.load(open(sys.argv[1]))["checks"]}
print(1 if checks["prediction_to_action_offline"]["passed"] else 0)
PY
)"
HARD_ARGUMENTS=()
if [[ "$OFFLINE_PASS" != 1 ]]; then
  run_hard_gate selection "$SELECTION_CHECKPOINT" "$SELECTION_SHA"
  HARD_ARGUMENTS=(--hard-task-gate "$CANDIDATE_ROOT/causal/selection/hard_task_gate.json")
fi
run_child ACCEPTING selection-gate check_r11_selection_gate.py \
  "all dynamics/action causal gates before formal budget" other \
  env PYTHONPATH="$ROOT" "$PYTHON" "$SELECTION_GATE" \
    --causal "$CANDIDATE_ROOT/causal/selection/summary.json" "${HARD_ARGUMENTS[@]}" \
    --output "$CANDIDATE_ROOT/causal/selection/gate.json"

FORMAL="$CANDIDATE_ROOT/train/formal"
train_stage formal 120000 "$FORMAL" "$SELECTION_CHECKPOINT" 0
FORMAL_CHECKPOINT="$FORMAL/checkpoints/checkpoint_120000.pt"
FORMAL_SHA="$(checkpoint_sha "$FORMAL/checkpoints/checkpoint_latest.receipt.json")"
run_suite formal normal "$FORMAL_CHECKPOINT" "$FORMAL_SHA" 20 "${ALL_TASKS[@]}"
summarize_suite formal normal "$FORMAL_SHA" 20 "${ALL_TASKS[@]}"
run_causal formal "$FORMAL_CHECKPOINT" "$FORMAL_SHA"
FORMAL_OFFLINE_PASS="$($PYTHON - "$CANDIDATE_ROOT/causal/formal/summary.json" <<'PY'
import json, sys
checks = {row["id"]: row for row in json.load(open(sys.argv[1]))["checks"]}
print(1 if checks["prediction_to_action_offline"]["passed"] else 0)
PY
)"
FORMAL_HARD_ARGUMENTS=()
if [[ "$FORMAL_OFFLINE_PASS" != 1 ]]; then
  run_hard_gate formal "$FORMAL_CHECKPOINT" "$FORMAL_SHA"
  FORMAL_HARD_ARGUMENTS=(--hard-task-gate "$CANDIDATE_ROOT/causal/formal/hard_task_gate.json")
fi
[[ -f "$W10_LATENCY" ]] || { printf 'missing frozen W10 latency receipt: %s\n' "$W10_LATENCY" >&2; exit 3; }
set +e
run_child ACCEPTING formal-acceptance accept_r11_candidate.py \
  "seven immutable section-11 qualification checks and frozen score" other \
  env PYTHONPATH="$ROOT" "$PYTHON" "$ACCEPTOR" --candidate "$CANDIDATE" \
    --branch "$BRANCH" --commit "$COMMIT" --checkpoint "$FORMAL_CHECKPOINT" \
    --checkpoint-sha256 "$FORMAL_SHA" --train-status "$CANDIDATE_ROOT/status/worker_status.json" \
    --validation "$CANDIDATE_ROOT/validation/formal/normal_summary.json" \
    --causal "$CANDIDATE_ROOT/causal/formal/summary.json" "${FORMAL_HARD_ARGUMENTS[@]}" \
    --w10-latency "$W10_LATENCY" --output "$CANDIDATE_ROOT/acceptance.json"
ACCEPT_CODE=$?
set -e
TERMINAL_WRITTEN=1
if ((ACCEPT_CODE == 0)); then
  status PASSED complete accept_r11_candidate.py "candidate qualifies for winner ranking" 0
else
  status FAILED complete accept_r11_candidate.py "candidate failed one or more immutable gates" "$ACCEPT_CODE"
fi
exit "$ACCEPT_CODE"
