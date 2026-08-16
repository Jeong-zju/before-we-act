#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_N2_REPO_ROOT:-/workspace/fe-pc-wam-b-core}"
RUN_ROOT="${BWA_N2_RUN_ROOT:-/workspace/bwa_runs/b-core/n2-r3-evidence-gated-persistence-v1}"
STEP2_ROOT="${BWA_STEP2_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
SEED_ROOT="${BWA_N2_SEED_ROOT:-/workspace/bwa_runs/w10-six-task-v1/seeds/validation}"
PYTHON="${BWA_N2_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
ROADMAP="${ROOT}/docs/plans/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_ZH.md"
CONTRACT="${RUN_ROOT}/contract/n2_contract.json"
OLD_CONCLUSION="${RUN_ROOT}/conclusion.json"
AUTHORIZATION="${RUN_ROOT}/contract/owner_closed_loop_revision.json"
OWNER_ROOT="${RUN_ROOT}/owner_closed_loop"
VALIDATION_ROOT="${OWNER_ROOT}/validation5"
VALIDATION_SUMMARY="${OWNER_ROOT}/validation5_summary.json"
STATUS="${OWNER_ROOT}/status.json"
B0H_CHECKPOINT="${STEP2_ROOT}/hidden_residual/formal/checkpoint_120000.pt"
OWNER_TOKEN="AUTHORIZED_OWNER_N2_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU_20260816"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

fail() {
  printf >&2 'B3-N2 owner closed loop: %s\n' "$*"
  exit 1
}

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
  write_status FAILED error "owner closed-loop runner exited with code ${code}" || true
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
trap 'stop_children; write_status STOPPED interrupted signal; exit 130' INT TERM

[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -z "$(git -c safe.directory="${ROOT}" -C "${ROOT}" status --short)" ]] || \
  fail "worktree must be clean"
SOURCE_COMMIT="$(git -c safe.directory="${ROOT}" -C "${ROOT}" rev-parse HEAD)"
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
[[ -f "${CONTRACT}" && -f "${OLD_CONCLUSION}" ]] || fail "frozen N2 receipts are missing"
[[ -f "${B0H_CHECKPOINT}" ]] || fail "B0-H checkpoint is missing"
grep -Fq "${OWNER_TOKEN}" "${ROADMAP}" || fail "owner authorization is not frozen"
mkdir -p "${OWNER_ROOT}/logs" "${VALIDATION_ROOT}"

if [[ ! -f "${AUTHORIZATION}" ]]; then
  write_status RUNNING authorization "freezing owner exception without changing the old conclusion"
  "${PYTHON}" scripts/before_we_act/prepare_b3_n2_owner_closed_loop.py \
    --roadmap "${ROADMAP}" --contract "${CONTRACT}" --run-root "${RUN_ROOT}" \
    --old-conclusion "${OLD_CONCLUSION}" --b0h-checkpoint "${B0H_CHECKPOINT}" \
    --source-commit "${SOURCE_COMMIT}" --output "${AUTHORIZATION}" \
    >"${OWNER_ROOT}/logs/authorization.log" 2>&1
fi

"${PYTHON}" - "${AUTHORIZATION}" "${OLD_CONCLUSION}" "${OWNER_TOKEN}" <<'PY'
import hashlib,json,sys
authorization=json.load(open(sys.argv[1],encoding="utf-8"))
old_path=sys.argv[2]
old_sha=hashlib.sha256(open(old_path,"rb").read()).hexdigest()
assert authorization["status"]=="OWNER_AUTHORIZED_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU"
assert authorization["authorization_token"]==sys.argv[3]
assert authorization["old_conclusion_sha256"]==old_sha
assert json.load(open(old_path,encoding="utf-8"))["status"]=="INCONCLUSIVE_TRAINING_NOT_CONVERGED"
assert authorization["validation20_candidate"]["closed_loop_results_used_for_selection"] is False
PY

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
SEEDS=(20260815 20260816 20260817)
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)
for task in "${TASKS[@]}"; do
  [[ -f "${SEED_ROOT}/${task}.json" ]] || fail "seed file is missing: ${task}"
done

is_complete() {
  local output="$1" mode="$2" expected_sha="$3"
  [[ -f "${output}" ]] && "${PYTHON}" - "${output}" "${mode}" "${expected_sha}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
ok=(value.get("mode")==sys.argv[2] and value.get("episodes")==5
    and len(value.get("rows",[]))==5
    and value.get("checkpoint_sha256")==sys.argv[3])
raise SystemExit(0 if ok else 1)
PY
}

run_model_validation() {
  local label="$1" mode="$2" checkpoint="$3" expected_sha="$4" gpu="$5"
  local output_root="${VALIDATION_ROOT}/${label}"
  mkdir -p "${output_root}"
  for task in "${TASKS[@]}"; do
    local output="${output_root}/${task}.json"
    local log="${OWNER_ROOT}/logs/validation5_${label}_${task}.log"
    if is_complete "${output}" "${mode}" "${expected_sha}"; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_b3_n2 \
      --checkpoint "${checkpoint}" --mode "${mode}" --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" --episodes 5 \
      --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
      --resume-log "${log}" --output "${output}" >>"${log}" 2>&1
  done
}

write_status RUNNING validation5 "running B0-H and all three frozen N2 seeds"
B0H_SHA="$(jq -r '.b0h_checkpoint_sha256' "${AUTHORIZATION}")"
run_model_validation b0h b0h "${B0H_CHECKPOINT}" "${B0H_SHA}" 3 & PIDS+=("$!")
for gpu in 0 1 2; do
  seed="${SEEDS[$gpu]}"
  checkpoint="$(jq -r --arg seed "${seed}" '.training_receipts[$seed].deployment_checkpoint' "${AUTHORIZATION}")"
  checkpoint_sha="$(jq -r --arg seed "${seed}" '.training_receipts[$seed].deployment_checkpoint_sha256' "${AUTHORIZATION}")"
  run_model_validation "seed_${seed}" n2 "${checkpoint}" "${checkpoint_sha}" "${gpu}" & PIDS+=("$!")
done
child_code=0
for pid in "${PIDS[@]}"; do
  set +e
  wait "${pid}"
  code=$?
  set -e
  ((code == 0)) || child_code="${code}"
done
((child_code == 0)) || fail "Validation5 worker failed with code ${child_code}"

"${PYTHON}" scripts/before_we_act/summarize_b3_n2_owner_validation5.py \
  --contract "${CONTRACT}" --authorization "${AUTHORIZATION}" \
  --validation-root "${VALIDATION_ROOT}" --seed-root "${SEED_ROOT}" \
  --output "${VALIDATION_SUMMARY}" \
  >"${OWNER_ROOT}/logs/validation5_summary.log" 2>&1

write_status RUNNING validation20 "running the pre-selected seed regardless of Validation5 direction"
BWA_N2_CONCLUSION="${AUTHORIZATION}" \
  bash scripts/before_we_act/run_b3_n2_validation20.sh \
  >"${OWNER_ROOT}/logs/validation20_runner.log" 2>&1

write_status PASSED complete "owner-authorized Validation5 and Validation20 diagnostics completed"
trap - ERR
printf 'B3_N2_OWNER_CLOSED_LOOP_COMPLETED validation5=%s validation20=%s\n' \
  "${VALIDATION_SUMMARY}" "${RUN_ROOT}/validation20/comparison.json"
