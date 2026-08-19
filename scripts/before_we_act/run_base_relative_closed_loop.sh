#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_A4_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_A4_RUN_ROOT:-/workspace/bwa_runs/a4-base-relative-v1}"
BCORE_ROOT="${BWA_BCORE_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
ACTION_CACHE="${BWA_ACTION_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n2-action-context-v1}"
SCENARIO_SPLIT="${BWA_SCENARIO_SPLIT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief/contract/scenario_split.json}"
SEED_ROOT="${BWA_A4_SEED_ROOT:-/workspace/bwa_runs/w10-six-task-v1/seeds/validation}"
PYTHON="${BWA_A4_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
CONTRACT="${RUN_ROOT}/contract/formal_contract.json"
OFFLINE="${RUN_ROOT}/offline_acceptance.json"
CANDIDATE="${RUN_ROOT}/contract/closed_loop_candidate.json"
STATUS="${RUN_ROOT}/closed_loop_status.json"
V5_ROOT="${RUN_ROOT}/validation5"
V20_ROOT="${RUN_ROOT}/validation20"
SUFFICIENCY="${RUN_ROOT}/control_sufficiency.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
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
  write_status FAILED error "closed-loop pipeline exited with code ${code}" || true
  exit "${code}"
}
PIDS=()
stop_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}
trap on_error ERR
trap 'stop_children; write_status STOPPED interrupted "closed-loop pipeline interrupted"; exit 130' INT TERM

[[ -f "${CONTRACT}" && -f "${OFFLINE}" ]] || {
  echo "formal contract or offline acceptance is missing" >&2
  exit 2
}
[[ "$(jq -r '.status' "${OFFLINE}")" == PASSED_OFFLINE_AWAITING_CLOSED_LOOP ]] || {
  echo "hard offline gates did not authorize closed loop" >&2
  exit 2
}

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/contract"
if [[ ! -f "${CANDIDATE}" ]]; then
  [[ ! -d "${V5_ROOT}" && ! -d "${V20_ROOT}" ]] || {
    echo "closed-loop output exists before candidate selection receipt" >&2
    exit 2
  }
  write_status RUNNING candidate_selection "freezing the lowest held-out-MSE full-arm candidate before closed loop"
  "${PYTHON}" - "${OFFLINE}" "${CANDIDATE}" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
source=Path(sys.argv[1]); output=Path(sys.argv[2])
value=json.loads(source.read_text())
rows=value["training"]["a4_full"]
seed,row=min(rows.items(),key=lambda item:(item[1]["action_mse"],item[1]["conditional_kl_nats"],int(item[0])))
root=Path(row["root"])
selected_update=int(row["selected_update"])
training_checkpoint=root/f"checkpoint_{selected_update:06d}.pt"
deployment=Path(row["deployment_checkpoint"])
for path in (training_checkpoint,deployment):
    if not path.is_file(): raise FileNotFoundError(path)
sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload={
  "format_version":"before-we-act.a4-closed-loop-candidate/1",
  "status":"FROZEN_BEFORE_CLOSED_LOOP",
  "selection_rule":"minimum held-out action MSE; conditional KL then seed only break exact ties",
  "closed_loop_results_used_for_selection":False,
  "seed":int(seed),"selected_update":selected_update,
  "held_out_action_mse":row["action_mse"],"conditional_kl_nats":row["conditional_kl_nats"],
  "training_checkpoint":str(training_checkpoint.resolve()),
  "training_checkpoint_sha256":sha(training_checkpoint),
  "deployment_checkpoint":str(deployment.resolve()),
  "deployment_checkpoint_sha256":sha(deployment),
  "offline_acceptance":str(source.resolve()),"offline_acceptance_sha256":sha(source),
  "frozen_at_epoch":time.time(),
}
output.parent.mkdir(parents=True,exist_ok=True)
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(temporary,output)
PY
fi

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
SEEDS=(20260815 20260816 20260817)
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)
for task in "${TASKS[@]}"; do
  [[ -f "${SEED_ROOT}/${task}.json" ]] || {
    echo "seed file is missing: ${task}" >&2
    exit 2
  }
done

is_complete() {
  local output="$1" expected_episodes="$2" expected_sha="$3"
  [[ -f "${output}" ]] && "${PYTHON}" - "${output}" "${expected_episodes}" "${expected_sha}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
episodes=int(sys.argv[2])
ok=(value.get("mode")=="n2" and value.get("episodes")==episodes
    and len(value.get("rows",[]))==episodes
    and value.get("checkpoint_sha256")==sys.argv[3])
raise SystemExit(0 if ok else 1)
PY
}

run_validation() {
  local label="$1" checkpoint="$2" expected_sha="$3" episodes="$4" gpu="$5" output_root="$6"
  mkdir -p "${output_root}/${label}"
  for task in "${TASKS[@]}"; do
    local output="${output_root}/${label}/${task}.json"
    local log="${RUN_ROOT}/logs/${output_root##*/}_${label}_${task}.log"
    if is_complete "${output}" "${episodes}" "${expected_sha}"; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_predictive_team_belief \
      --checkpoint "${checkpoint}" --mode n2 --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" --episodes "${episodes}" \
      --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
      --resume-log "${log}" --output "${output}" >>"${log}" 2>&1
  done
}

write_status RUNNING validation5 "running all three frozen full-arm seeds and the matched-capacity sufficiency probe"
PIDS=()
for gpu in 0 1 2; do
  seed="${SEEDS[$gpu]}"
  checkpoint="$(jq -r --arg seed "${seed}" '.training.a4_full[$seed].deployment_checkpoint' "${OFFLINE}")"
  checkpoint_sha="$(jq -r --arg seed "${seed}" '.training.a4_full[$seed].deployment_checkpoint_sha256' "${OFFLINE}")"
  run_validation "seed_${seed}" "${checkpoint}" "${checkpoint_sha}" 5 "${gpu}" "${V5_ROOT}" & PIDS+=("$!")
done
if [[ ! -f "${SUFFICIENCY}" ]]; then
  CUDA_VISIBLE_DEVICES=3 "${PYTHON}" scripts/before_we_act/evaluate_base_relative_sufficiency.py \
    --cache "${SIGNAL_CACHE}" --action-context-cache "${ACTION_CACHE}" \
    --contract "${CONTRACT}" --scenario-split "${SCENARIO_SPLIT}" \
    --checkpoint "$(jq -r '.training_checkpoint' "${CANDIDATE}")" \
    --output "${SUFFICIENCY}" --updates 10000 --workers 2 \
    >"${RUN_ROOT}/logs/control_sufficiency.log" 2>&1 & PIDS+=("$!")
fi
child_code=0
for pid in "${PIDS[@]}"; do
  set +e
  wait "${pid}"
  code=$?
  set -e
  ((code == 0)) || child_code="${code}"
done
((child_code == 0)) || {
  echo "Validation5 or sufficiency worker failed with code ${child_code}" >&2
  exit "${child_code}"
}

"${PYTHON}" scripts/before_we_act/analyze_base_relative_belief.py \
  --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
  --bcore-training-root "${BCORE_ROOT}/training" \
  --sufficiency "${SUFFICIENCY}" --validation5-root "${V5_ROOT}" \
  --output "${RUN_ROOT}/validation5_acceptance.json" \
  >"${RUN_ROOT}/logs/validation5_analysis.log" 2>&1
if [[ "$(jq -r '.validation5_gate.passed' "${RUN_ROOT}/validation5_acceptance.json")" != true ]]; then
  write_status FAILED validation5_gate "Validation5 did not meet the frozen mean/single-seed gate; Validation20 not run"
  trap - ERR
  exit 0
fi

SELECTED="$(jq -r '.deployment_checkpoint' "${CANDIDATE}")"
SELECTED_SHA="$(jq -r '.deployment_checkpoint_sha256' "${CANDIDATE}")"
write_status RUNNING validation20 "running the pre-selected candidate on the frozen six-task Validation20"
mkdir -p "${V20_ROOT}/selected"
run_selected_task() {
  local task="$1" gpu="$2"
  local output="${V20_ROOT}/selected/${task}.json"
  local log="${RUN_ROOT}/logs/validation20_selected_${task}.log"
  if is_complete "${output}" 20 "${SELECTED_SHA}"; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_predictive_team_belief \
    --checkpoint "${SELECTED}" --mode n2 --task "${task}" \
    --seed-file "${SEED_ROOT}/${task}.json" --episodes 20 \
    --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
    --resume-log "${log}" --output "${output}" >>"${log}" 2>&1
}
run_wave() {
  PIDS=()
  local item task gpu pid code=0 child_code
  for item in "$@"; do
    task="${item%%:*}"; gpu="${item##*:}"
    run_selected_task "${task}" "${gpu}" & PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    set +e
    wait "${pid}"
    child_code=$?
    set -e
    ((child_code == 0)) || code="${child_code}"
  done
  PIDS=()
  ((code == 0))
}
run_wave camera_alignment:0 long_pipeline_delivery:1 take_photo:2 lift_barrier:3
run_wave pass_shoe:0 place_food:1

"${PYTHON}" scripts/before_we_act/analyze_base_relative_belief.py \
  --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
  --bcore-training-root "${BCORE_ROOT}/training" \
  --sufficiency "${SUFFICIENCY}" --validation5-root "${V5_ROOT}" \
  --validation20-root "${V20_ROOT}" --output "${RUN_ROOT}/final_acceptance.json" \
  >"${RUN_ROOT}/logs/final_analysis.log" 2>&1
final="$(jq -r '.status' "${RUN_ROOT}/final_acceptance.json")"
if [[ "${final}" == PASSED_STEP4_ACCEPT || "${final}" == PASSED_STEP4_CONDITIONAL_ACTION_ATTRIBUTION_OPEN ]]; then
  write_status PASSED complete "${final}"
else
  write_status FAILED complete "${final}"
fi
trap - ERR
printf 'A4_CLOSED_LOOP_COMPLETED status=%s\n' "${final}"
