#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=""
CANDIDATE=""
GPU_INDEX=""
PARENT_CHECKPOINT=""
PROTOCOL_ROOT=""
PYTHON=/venv/robofactory-act/bin/python
WORKERS=8

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --parent-checkpoint) PARENT_CHECKPOINT="$2"; shift 2 ;;
    --protocol-root) PROTOCOL_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$CANDIDATE" =~ ^p[0-3]$ || ! "$GPU_INDEX" =~ ^[0-3]$ ]]; then
  printf 'candidate p0..p3 and GPU 0..3 are required\n' >&2
  exit 2
fi
if [[ "${CANDIDATE#p}" != "$GPU_INDEX" ]]; then
  printf 'candidate/GPU mapping differs: %s/%s\n' "$CANDIDATE" "$GPU_INDEX" >&2
  exit 2
fi

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$FE_ROOT/scripts/before_we_act/r10_runtime.py"
TRAINER="$FE_ROOT/stereo_core/train_bwa_perception.py"
EVALUATOR="$FE_ROOT/stereo_core/evaluate_bwa_perception.py"
GATE_AUDIT="$FE_ROOT/scripts/before_we_act/audit_r10_gate_zero.py"
ACCEPTOR="$FE_ROOT/scripts/before_we_act/accept_r10.py"
CONFIG="$FE_ROOT/configs/before_we_act/r10_perception/${CANDIDATE}.yaml"
MANIFEST="$RUN_ROOT/run_manifest.json"
CANDIDATE_ROOT="$RUN_ROOT/candidates/$CANDIDATE"
LOG_ROOT="$CANDIDATE_ROOT/logs"
MAIN_LOG="$LOG_ROOT/candidate.log"
mkdir -p "$LOG_ROOT" "$CANDIDATE_ROOT/train" "$CANDIDATE_ROOT/validation"
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
if [[ "$(git -C "$FE_ROOT" branch --show-current)" != "$EXPECTED_BRANCH" || \
      "$(git -C "$FE_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  printf 'candidate worktree branch/commit identity differs\n' >&2
  exit 3
fi
for path in "$PYTHON" "$RUNTIME" "$TRAINER" "$EVALUATOR" "$GATE_AUDIT" \
  "$ACCEPTOR" "$CONFIG" "$PARENT_CHECKPOINT" "$PROTOCOL_ROOT/baseline_gate20.json"; do
  [[ -e "$path" ]] || { printf 'missing required path: %s\n' "$path" >&2; exit 3; }
done
mapfile -t MANIFESTS < <(find /workspace/datasets/robofactory_multitask -mindepth 2 -maxdepth 2 -name training_manifest.json -type f | sort)
if ((${#MANIFESTS[@]} != 5)); then
  printf 'expected five training manifests, got %s\n' "${#MANIFESTS[@]}" >&2
  exit 3
fi

CHILD_PID=0
HEARTBEAT_PID=0
STOP_REQUESTED=0
status() {
  "$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate "$CANDIDATE" \
    --state "$1" --stage "$2" --program "$3" --detail "$4" --pid "$$" \
    --child-pid "$CHILD_PID" --log "$MAIN_LOG" ${5:+--total-updates "$5"}
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" "$RUNTIME" heartbeat --run-root "$RUN_ROOT" \
      --candidate "$CANDIDATE" --pid "$$" >/dev/null 2>&1 || true
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
    status STOPPED stopped run_r10_candidate.sh "graceful stop requested; artifacts preserved" || true
  elif ((code != 0)); then
    status FAILED failed run_r10_candidate.sh "candidate pipeline exited with code $code" || true
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
  set +e
  wait "$CHILD_PID"
  local code=$?
  set -e
  CHILD_PID=0
  if ((code != 0)); then
    return "$code"
  fi
}

evaluate_phase() {
  local phase="$1" checkpoint="$2"
  local validation="$CANDIDATE_ROOT/validation/$phase"
  local intervention
  intervention="$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["intervention"]["name"])
PY
)"
  mkdir -p "$validation/normal" "$validation/intervention"
  local completed=0
  for condition in normal "$intervention"; do
    local condition_dir=normal
    [[ "$condition" == normal ]] || condition_dir=intervention
    for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
      completed=$((completed + 1))
      local eval_log="$LOG_ROOT/${phase}_${condition_dir}_${task}.log"
      run_child VALIDATING "${phase}_${condition_dir}" evaluate_bwa_perception.py \
        "$task condition=$condition ($completed/10)" 30000 \
        env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
        "$PYTHON" "$EVALUATOR" \
          --parent-checkpoint "$PARENT_CHECKPOINT" \
          --extension-checkpoint "$checkpoint" \
          --task "$task" --seed-file "$PROTOCOL_ROOT/seeds/$task.json" \
          --episodes 20 --max-steps 1500 --device cuda:0 \
          --intervention "$condition" \
          --output "$validation/$condition_dir/$task.json" \
          --resume-log "$eval_log" >>"$eval_log" 2>&1
    done
  done
}

accept_phase() {
  local phase="$1" checkpoint="$2" mode="$3"
  local validation="$CANDIDATE_ROOT/validation/$phase"
  local gate="$validation/gate_zero_latency.json"
  local output="$validation/acceptance.json"
  local mappings=()
  for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
    mappings+=(--baseline "$task=/workspace/bwa_runs/shared/frozen100/$task.json")
    mappings+=(--normal "$task=$validation/normal/$task.json")
    mappings+=(--intervention "$task=$validation/intervention/$task.json")
  done
  run_child ACCEPTING "${phase}_acceptance" accept_r10.py "five hard gates" 30000 \
    "$PYTHON" "$ACCEPTOR" --candidate-id "$CANDIDATE" \
      --branch "$EXPECTED_BRANCH" --commit "$EXPECTED_COMMIT" \
      --checkpoint "$checkpoint" --gate-audit "$gate" \
      "${mappings[@]}" --mode "$mode" --output "$output"
}

status PREPARING preflight run_r10_candidate.sh "validating exact parent, common sampler and candidate contract" 30000
PREFLIGHT="$CANDIDATE_ROOT/preflight"
run_child TRAINING preflight train_bwa_perception.py "two-update full-path preflight" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$TRAINER" --config "$CONFIG" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --manifests "${MANIFESTS[@]}" --output "$PREFLIGHT" --phase preflight --workers "$WORKERS"
run_child ACCEPTING preflight_gate_zero audit_r10_gate_zero.py "real-parent exact fallback and latency" 2 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$GATE_AUDIT" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --extension-checkpoint "$PREFLIGHT/checkpoints/checkpoint_latest.pt" \
    --device cuda:0 --latency-repeats 1000 --output "$PREFLIGHT/gate_zero_latency.json"

SCREEN="$CANDIDATE_ROOT/train/screen"
run_child TRAINING screen train_bwa_perception.py "locked 10k screen" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$TRAINER" --config "$CONFIG" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --manifests "${MANIFESTS[@]}" --output "$SCREEN" --phase screen --workers "$WORKERS"
SCREEN_CKPT="$SCREEN/checkpoints/checkpoint_010000.pt"
mkdir -p "$CANDIDATE_ROOT/validation/screen"
run_child ACCEPTING screen_gate_zero audit_r10_gate_zero.py "trained gate-zero and latency audit" 10000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$GATE_AUDIT" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --extension-checkpoint "$SCREEN_CKPT" --device cuda:0 --latency-repeats 1000 \
    --output "$CANDIDATE_ROOT/validation/screen/gate_zero_latency.json"
evaluate_phase screen "$SCREEN_CKPT"
set +e
accept_phase screen "$SCREEN_CKPT" screen
SCREEN_CODE=$?
set -e
if ((SCREEN_CODE == 10)); then
  cp "$CANDIDATE_ROOT/validation/screen/acceptance.json" "$CANDIDATE_ROOT/acceptance.json"
  status FAILED screen_complete accept_r10.py "10k screen rejected selection; no acceptance bypass" 10000
  exit 10
fi
if "$PYTHON" - "$CANDIDATE_ROOT/validation/screen/acceptance.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 1)
PY
then
  cp "$CANDIDATE_ROOT/validation/screen/acceptance.json" "$CANDIDATE_ROOT/acceptance.json"
  status PASSED complete accept_r10.py "all five R10 gates passed at 10k" 10000
  exit 0
fi

SELECTION="$CANDIDATE_ROOT/train/selection"
run_child TRAINING selection train_bwa_perception.py "continue same run to locked 30k cutoff" 30000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$TRAINER" --config "$CONFIG" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --manifests "${MANIFESTS[@]}" --output "$SELECTION" --phase selection \
    --resume "$SCREEN_CKPT" --workers "$WORKERS"
FINAL_CKPT="$SELECTION/checkpoints/checkpoint_030000.pt"
mkdir -p "$CANDIDATE_ROOT/validation/formal"
run_child ACCEPTING formal_gate_zero audit_r10_gate_zero.py "formal trained gate-zero and latency audit" 30000 \
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" BWA_R10_RUN_ROOT="$RUN_ROOT" \
  "$PYTHON" "$GATE_AUDIT" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --extension-checkpoint "$FINAL_CKPT" --device cuda:0 --latency-repeats 1000 \
    --output "$CANDIDATE_ROOT/validation/formal/gate_zero_latency.json"
evaluate_phase formal "$FINAL_CKPT"
set +e
accept_phase formal "$FINAL_CKPT" formal
FORMAL_CODE=$?
set -e
cp "$CANDIDATE_ROOT/validation/formal/acceptance.json" "$CANDIDATE_ROOT/acceptance.json"
if ((FORMAL_CODE == 0)); then
  status PASSED complete accept_r10.py "all five formal R10 gates passed" 30000
else
  status FAILED complete accept_r10.py "formal R10 acceptance failed; see per-gate reasons" 30000
fi
exit "$FORMAL_CODE"
