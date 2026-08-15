#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_N2_REPO_ROOT:-/workspace/fe-pc-wam-b-core}"
RUN_ROOT="${BWA_N2_RUN_ROOT:-/workspace/bwa_runs/b-core/n2-r1-discrete-belief-stabilization-v2}"
FAILED_N2_ROOT="${BWA_FAILED_N2_RUN_ROOT:-/workspace/bwa_runs/b-core/n2-predictive-team-belief-v1}"
N1_CACHE="${BWA_N1_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
ACTION_CACHE="${BWA_N2_ACTION_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n2-action-context-v1}"
STEP2_ROOT="${BWA_STEP2_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
R1_ROOT="${BWA_R1_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief}"
R1_REVISION_ROOT="${BWA_R1_REVISION_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-owner-revision-teacher-student}"
DATA_ROOT="${BWA_DATA_ROOT:-/workspace/datasets/robofactory_multitask}"
VISUAL_CACHE="${BWA_STEP2_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-step2-dino-history-cache-v2}"
PYTHON="${BWA_N2_PYTHON:-/venv/robofactory-act/bin/python}"
TORCHRUN="${BWA_N2_TORCHRUN:-/venv/robofactory-act/bin/torchrun}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
ROADMAP="${ROOT}/docs/plans/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_ZH.md"
CONTRACT="${RUN_ROOT}/contract/n2_contract.json"
STATUS="${RUN_ROOT}/pipeline_status.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs"
cd "${ROOT}"

write_status() {
  local status="$1" stage="$2" detail="$3"
  "${PYTHON}" - "${STATUS}" "${status}" "${stage}" "${detail}" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,p)
PY
}

on_error() {
  local code=$?
  write_status FAILED error "pipeline exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'write_status STOPPED interrupted signal; exit 130' INT TERM

[[ -x "${PYTHON}" && -x "${TORCHRUN}" ]] || { echo "N2 Python/torchrun missing" >&2; exit 1; }
[[ -z "$(git -c safe.directory="${ROOT}" -C "${ROOT}" status --short)" ]] || { echo "N2 worktree must be clean" >&2; exit 1; }
SOURCE_COMMIT="$(git -c safe.directory="${ROOT}" -C "${ROOT}" rev-parse HEAD)"
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "N2 source commit is invalid" >&2; exit 1; }
[[ "$(git -C "${ROBOFACTORY_ROOT}" rev-parse HEAD)" == 5868242322414a91454e22f1dd9641f613ba1bcf ]] || { echo "RoboFactory commit drift" >&2; exit 1; }

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
MANIFESTS=()
for task in "${TASKS[@]}"; do
  MANIFESTS+=("${DATA_ROOT}/${task}/training_manifest.json")
done
B0H_CHECKPOINT="${STEP2_ROOT}/hidden_residual/formal/checkpoint_120000.pt"
NORMALIZATION="${STEP2_ROOT}/contract/normalization.pt"
SCENARIO_SPLIT="${R1_ROOT}/contract/scenario_split.json"

if [[ ! -f "${ACTION_CACHE}/cache_receipt.json" ]]; then
  write_status RUNNING action_context_cache "caching frozen B0-H decoded contexts on four GPUs"
  "${TORCHRUN}" --standalone --nproc_per_node=4 \
    scripts/before_we_act/build_b3_n2_action_context_cache.py \
    --manifests "${MANIFESTS[@]}" \
    --normalization "${NORMALIZATION}" \
    --visual-cache "${VISUAL_CACHE}" \
    --n1-cache "${N1_CACHE}" \
    --b0h-checkpoint "${B0H_CHECKPOINT}" \
    --output "${ACTION_CACHE}" --batch-size 16 \
    >"${RUN_ROOT}/logs/action_context_cache.log" 2>&1
fi

if [[ ! -f "${CONTRACT}" ]]; then
  write_status RUNNING contract "freezing N2 architecture, budget and classification before F0"
  "${PYTHON}" scripts/before_we_act/prepare_b3_n2.py \
    --roadmap "${ROADMAP}" \
    --r1-contract "${R1_ROOT}/contract/r1_contract.json" \
    --student-contract "${R1_REVISION_ROOT}/contract/r1_5_student_contract.json" \
    --student-conclusion "${R1_REVISION_ROOT}/r1_5_student/conclusion.json" \
    --student-diagnostic "${R1_REVISION_ROOT}/r1_5_student/validation_diagnostic.json" \
    --step2-contract "${STEP2_ROOT}/contract/step2_contract.json" \
    --b0h-checkpoint "${B0H_CHECKPOINT}" \
    --scenario-split "${SCENARIO_SPLIT}" \
    --n1-cache "${N1_CACHE}" \
    --action-context-cache "${ACTION_CACHE}" \
    --failed-n2-run "${FAILED_N2_ROOT}" \
    --source-commit "${SOURCE_COMMIT}" \
    --output "${CONTRACT}" \
    >"${RUN_ROOT}/logs/contract.log" 2>&1
fi

if [[ ! -f "${RUN_ROOT}/contract/f0_receipt.json" ]]; then
  write_status RUNNING f0 "auditing legal inputs, paired exchange, zero-init and finite backward"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/before_we_act/verify_b3_n2.py f0 \
    --cache "${N1_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${RUN_ROOT}/contract/f0_receipt.json" \
    >"${RUN_ROOT}/logs/f0.log" 2>&1
fi

F1_REFERENCE="${RUN_ROOT}/f1/reference"
F1_RESUMED="${RUN_ROOT}/f1/resumed"
if [[ ! -f "${RUN_ROOT}/contract/f1_receipt.json" ]]; then
  write_status RUNNING f1 "checking four-update fresh versus 2+2 resume equivalence"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_b3_n2 \
    --cache "${N1_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${F1_REFERENCE}" --seed 20260815 --updates 4 --workers 0 --save-every 2 \
    >"${RUN_ROOT}/logs/f1_reference.log" 2>&1
  mkdir -p "${F1_RESUMED}"
  cp "${F1_REFERENCE}/checkpoint_000002.pt" "${F1_RESUMED}/checkpoint_latest.pt"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_b3_n2 \
    --cache "${N1_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${F1_RESUMED}" --seed 20260815 --updates 4 --workers 0 --save-every 2 \
    >"${RUN_ROOT}/logs/f1_resumed.log" 2>&1
  "${PYTHON}" scripts/before_we_act/verify_b3_n2.py f1 \
    --reference "${F1_REFERENCE}/checkpoint_000004.pt" \
    --resumed "${F1_RESUMED}/checkpoint_000004.pt" \
    --output "${RUN_ROOT}/contract/f1_receipt.json" \
    >"${RUN_ROOT}/logs/f1_verify.log" 2>&1
fi

PILOT_ROOT="${RUN_ROOT}/repair_pilot/seed_20260815"
if [[ ! -f "${RUN_ROOT}/repair_pilot_conclusion.json" ]]; then
  write_status RUNNING repair_pilot "running one 2000-update seed; no formal training is authorized"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m before_we_act.train_b3_n2 \
    --cache "${N1_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${PILOT_ROOT}" --seed 20260815 --updates 2000 --workers 2 \
    --save-every 500 --log-every 100 --evaluate-at-end \
    >"${RUN_ROOT}/logs/repair_pilot.log" 2>&1
  "${PYTHON}" scripts/before_we_act/analyze_b3_n2_repair_pilot.py \
    --contract "${CONTRACT}" --pilot "${PILOT_ROOT}" \
    --output "${RUN_ROOT}/repair_pilot_conclusion.json" \
    >"${RUN_ROOT}/logs/repair_pilot_analysis.log" 2>&1
fi

decision="$(jq -r '.status' "${RUN_ROOT}/repair_pilot_conclusion.json")"
if [[ "${BWA_N2_AUTHORIZE_FORMAL:-0}" != 1 ]]; then
  write_status PASSED repair_pilot_complete "${decision}; formal training was not started"
  printf 'B3_N2_REPAIR_PILOT_COMPLETED status=%s formal_training_started=false\n' "${decision}"
  exit 0
fi
if [[ "${decision}" != PASSED_REPAIR_GATES_FORMAL_TRAINING_REQUIRES_OWNER_DECISION ]]; then
  write_status FAILED repair_gate "${decision}; formal training is forbidden"
  echo "repair gates forbid formal N2 training: ${decision}" >&2
  exit 1
fi

write_status RUNNING training "training three frozen N2 seeds to 120000 updates"
SEEDS=(20260815 20260816 20260817)
PIDS=()
for gpu in 0 1 2; do
  seed="${SEEDS[$gpu]}"
  seed_root="${RUN_ROOT}/training/seed_${seed}"
  terminal="$(jq -r '.status // empty' "${seed_root}/status.json" 2>/dev/null || true)"
  if [[ "${terminal}" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT|INCONCLUSIVE_TRAINING_NOT_CONVERGED)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_b3_n2 \
    --cache "${N1_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --output "${seed_root}" --seed "${seed}" --updates 120000 --workers 2 --save-every 5000 \
    >"${RUN_ROOT}/logs/train_seed_${seed}.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]:-}"; do
  [[ -n "${pid}" ]] && wait "${pid}"
done

write_status RUNNING offline_analysis "issuing the training-sufficient offline classification"
"${PYTHON}" scripts/before_we_act/analyze_b3_n2.py \
  --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/offline_conclusion.json" \
  >"${RUN_ROOT}/logs/offline_analysis.log" 2>&1

if [[ "$(jq -r '.validation5_authorized' "${RUN_ROOT}/offline_conclusion.json")" != true ]]; then
  write_status PASSED complete "$(jq -r '.status' "${RUN_ROOT}/offline_conclusion.json")"
  cp "${RUN_ROOT}/offline_conclusion.json" "${RUN_ROOT}/conclusion.json"
  exit 0
fi

write_status RUNNING validation5 "running paired B0-H and three-seed N2 Validation5"
SEED_ROOT="/workspace/bwa_runs/w10-six-task-v1/seeds/validation"
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)

run_model_validation() {
  local label="$1" mode="$2" checkpoint="$3" gpu="$4"
  local output_root="${RUN_ROOT}/validation5/${label}"
  mkdir -p "${output_root}"
  for task in "${TASKS[@]}"; do
    local output="${output_root}/${task}.json"
    local log="${RUN_ROOT}/logs/validation5_${label}_${task}.log"
    if [[ "$(jq -r '.episodes // 0' "${output}" 2>/dev/null || true)" == 5 ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_b3_n2 \
      --checkpoint "${checkpoint}" --mode "${mode}" --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" --episodes 5 \
      --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
      --resume-log "${log}" --output "${output}" >>"${log}" 2>&1
  done
}

PIDS=()
run_model_validation b0h b0h "${B0H_CHECKPOINT}" 3 & PIDS+=("$!")
for gpu in 0 1 2; do
  seed="${SEEDS[$gpu]}"
  run_model_validation "seed_${seed}" n2 "${RUN_ROOT}/training/seed_${seed}/deployment_checkpoint.pt" "${gpu}" &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

"${PYTHON}" scripts/before_we_act/analyze_b3_n2.py \
  --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
  --validation-root "${RUN_ROOT}/validation5" \
  --output "${RUN_ROOT}/conclusion.json" \
  >"${RUN_ROOT}/logs/final_analysis.log" 2>&1
write_status PASSED complete "$(jq -r '.status' "${RUN_ROOT}/conclusion.json")"
trap - ERR
printf 'B3_N2_PIPELINE_COMPLETED status=%s\n' "$(jq -r '.status' "${RUN_ROOT}/conclusion.json")"
