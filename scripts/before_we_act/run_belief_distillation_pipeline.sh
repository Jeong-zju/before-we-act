#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BWA_REPO_ROOT:-/workspace/fe-pc-wam}"
BASE_RUN_ROOT="${BWA_ACTION_GROUNDED_BASE_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-action-grounded-belief}"
RUN_ROOT="${BWA_BELIEF_DISTILLATION_RUN_ROOT:-/workspace/bwa_runs/b-core/n1-r1-owner-revision-teacher-student}"
TEAM_SIGNAL_CACHE="${BWA_TEAM_SIGNAL_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-b-core-n1-cache-v1}"
PYTHON="${BWA_PYTHON:-/venv/robofactory-act/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor/stereo-core:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/contract" "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

PARENT_CONTRACT="${BASE_RUN_ROOT}/contract/r1_contract.json"
SCENARIO_SPLIT="${BASE_RUN_ROOT}/contract/scenario_split.json"
FAIR_ROOT="${BASE_RUN_ROOT}/r1_1_fair_probe"
FAIR_CONCLUSION="${FAIR_ROOT}/conclusion.json"
PILOT_ROOT="${BASE_RUN_ROOT}/r1_3_counterfactual_pilot"
OWNER_REVISION="${RUN_ROOT}/contract/owner_revision.json"
TEACHER_CONTRACT="${RUN_ROOT}/contract/r1_4_teacher_contract.json"
STUDENT_CONTRACT="${RUN_ROOT}/contract/r1_5_student_contract.json"
STUDENT_CONTINUATION="${RUN_ROOT}/contract/owner_student_continuation.json"

if [[ ! -f "${OWNER_REVISION}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_belief_owner_revision.py \
    --parent-contract "${PARENT_CONTRACT}" \
    --fair-conclusion "${FAIR_CONCLUSION}" \
    --pilot-conclusion "${PILOT_ROOT}/conclusion.json" \
    --pilot-diagnostic "${PILOT_ROOT}/diagnostic_receipt.json" \
    --output "${OWNER_REVISION}" \
    >"${RUN_ROOT}/logs/owner_revision_prepare.log" 2>&1
fi

if [[ ! -f "${TEACHER_CONTRACT}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_belief_teacher.py \
    --parent-contract "${PARENT_CONTRACT}" \
    --fair-conclusion "${FAIR_CONCLUSION}" \
    --owner-revision "${OWNER_REVISION}" \
    --output "${TEACHER_CONTRACT}" \
    >"${RUN_ROOT}/logs/r1_4_prepare.log" 2>&1
fi

if [[ ! -f "${RUN_ROOT}/contract/f0_receipt.json" ]]; then
  "${PYTHON}" scripts/before_we_act/verify_belief_distillation.py \
    --owner-revision "${OWNER_REVISION}" \
    --teacher-contract "${TEACHER_CONTRACT}" \
    --output "${RUN_ROOT}/contract/f0_receipt.json" \
    >"${RUN_ROOT}/logs/f0.log" 2>&1
fi

seeds=(20260815 20260816 20260817)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  output="${RUN_ROOT}/r1_4_teacher/seed_${seed}"
  status="$(${PYTHON} -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "") if p.is_file() else "")' "${output}/status.json")"
  if [[ "${status}" =~ ^(PLATFORM_REACHED|SATURATED_BY_OVERFIT|INCONCLUSIVE_TRAINING_NOT_CONVERGED)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_belief_teacher \
    --cache "${TEAM_SIGNAL_CACHE}" \
    --parent-contract "${PARENT_CONTRACT}" \
    --teacher-contract "${TEACHER_CONTRACT}" \
    --scenario-split "${SCENARIO_SPLIT}" \
    --fair-run-root "${FAIR_ROOT}" \
    --output "${output}" \
    --seed "${seed}" \
    >"${RUN_ROOT}/logs/r1_4_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]:-}"; do
  [[ -n "${pid}" ]] && wait "${pid}"
done

if [[ ! -f "${RUN_ROOT}/r1_4_teacher/conclusion.json" ]]; then
  CUDA_VISIBLE_DEVICES=3 "${PYTHON}" scripts/before_we_act/analyze_belief_teacher.py \
    --cache "${TEAM_SIGNAL_CACHE}" \
    --parent-contract "${PARENT_CONTRACT}" \
    --teacher-contract "${TEACHER_CONTRACT}" \
    --scenario-split "${SCENARIO_SPLIT}" \
    --fair-run-root "${FAIR_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/r1_4_teacher/conclusion.json" \
    >"${RUN_ROOT}/logs/r1_4_analyze.log" 2>&1
fi

teacher_status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RUN_ROOT}/r1_4_teacher/conclusion.json")"
if [[ ! -f "${STUDENT_CONTINUATION}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_belief_student_continuation.py \
    --owner-revision "${OWNER_REVISION}" \
    --teacher-conclusion "${RUN_ROOT}/r1_4_teacher/conclusion.json" \
    --output "${STUDENT_CONTINUATION}" \
    >"${RUN_ROOT}/logs/r1_5_continuation_prepare.log" 2>&1
fi

continuation_status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${STUDENT_CONTINUATION}")"
if [[ "${continuation_status}" != "AUTHORIZED_R1_5_EXPLORATORY_VALIDATION_ONLY" ]]; then
  exit 0
fi

if [[ ! -f "${STUDENT_CONTRACT}" ]]; then
  "${PYTHON}" scripts/before_we_act/prepare_belief_student.py \
    --parent-contract "${PARENT_CONTRACT}" \
    --teacher-contract "${TEACHER_CONTRACT}" \
    --teacher-conclusion "${RUN_ROOT}/r1_4_teacher/conclusion.json" \
    --fair-conclusion "${FAIR_CONCLUSION}" \
    --owner-revision "${OWNER_REVISION}" \
    --student-continuation "${STUDENT_CONTINUATION}" \
    --output "${STUDENT_CONTRACT}" \
    >"${RUN_ROOT}/logs/r1_5_prepare.log" 2>&1
fi

pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  output="${RUN_ROOT}/r1_5_student/seed_${seed}"
  status="$(${PYTHON} -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status", "") if p.is_file() else "")' "${output}/status.json")"
  if [[ "${status}" =~ ^(PLATFORM_REACHED|INCONCLUSIVE_TRAINING_NOT_CONVERGED)$ ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m before_we_act.train_belief_student \
    --cache "${TEAM_SIGNAL_CACHE}" \
    --parent-contract "${PARENT_CONTRACT}" \
    --student-contract "${STUDENT_CONTRACT}" \
    --scenario-split "${SCENARIO_SPLIT}" \
    --fair-run-root "${FAIR_ROOT}" \
    --teacher-run-root "${RUN_ROOT}" \
    --output "${output}" \
    --seed "${seed}" \
    >"${RUN_ROOT}/logs/r1_5_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]:-}"; do
  [[ -n "${pid}" ]] && wait "${pid}"
done

if [[ ! -f "${RUN_ROOT}/r1_5_student/conclusion.json" ]]; then
  CUDA_VISIBLE_DEVICES=3 "${PYTHON}" scripts/before_we_act/analyze_belief_student.py \
    --cache "${TEAM_SIGNAL_CACHE}" \
    --parent-contract "${PARENT_CONTRACT}" \
    --student-contract "${STUDENT_CONTRACT}" \
    --scenario-split "${SCENARIO_SPLIT}" \
    --fair-run-root "${FAIR_ROOT}" \
    --teacher-run-root "${RUN_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/r1_5_student/conclusion.json" \
    >"${RUN_ROOT}/logs/r1_5_analyze.log" 2>&1
fi
