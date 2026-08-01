#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FE_ROOT}/experiments/wam_flow/s3_r6/candidate.env"
for name in S3_R6_RUN_ROOT S3_R6_READY_FILE S3_R6_FAILED_FILE S3_R6_BASE_REPO \
  S3_R6_RUN_ID S3_R6_UV_CACHE_DIR S3_R6_UV_ENV S3_R6_ROBOFACTORY_ROOT \
  S3_R6_RF_PYTHON GPU_INDEX; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'Missing %s\n' "${name}"; exit 2; fi
done
if [[ ! -f "${ENV_FILE}" ]]; then printf >&2 'Missing %s\n' "${ENV_FILE}"; exit 3; fi
# shellcheck source=/dev/null
source "${ENV_FILE}"
for name in S3_R6_CANDIDATE_ID S3_R6_TOTAL_UPDATES S3_R6_CONFIG_REL; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'candidate.env missing %s\n' "${name}"; exit 2; fi
done

SLUG="$(printf '%s' "${S3_R6_CANDIDATE_ID}" | tr '[:upper:]-' '[:lower:]_')"
MICRO_ROUND="${S3_R6_CANDIDATE_ID%%-*}"
CANDIDATE_SHORT="${S3_R6_CANDIDATE_ID##*-}"
CANDIDATE_ROOT="${S3_R6_RUN_ROOT}/candidates/${SLUG}"
STATUS_TOOL="${S3_R6_BASE_REPO}/scripts/s3_r6_runtime.py"
CONFIG="${FE_ROOT}/${S3_R6_CONFIG_REL}"
CHECKPOINT="${CANDIDATE_ROOT}/checkpoints/policy.pt"
RESUME="${CANDIDATE_ROOT}/checkpoints/resume.pt"
TRAIN_PROGRESS="${CANDIDATE_ROOT}/train/progress.jsonl"
STAGE_LOG="${CANDIDATE_ROOT}/train/stages.jsonl"
LOG_PATH="${CANDIDATE_ROOT}/logs/candidate.log"
HEARTBEAT_PID=""
COMPLETED=0
mkdir -p "${CANDIDATE_ROOT}"/{logs,train,validation,checkpoints,outputs} || exit $?
exec > >(tee -a "${LOG_PATH}") 2>&1

status() {
  arguments=(status --run-root "${S3_R6_RUN_ROOT}" \
    --candidate "${S3_R6_CANDIDATE_ID}" --phase "$1" --program "$2" \
    --detail "${3:-}" --gpu-index "${GPU_INDEX}" \
    --total-updates "${S3_R6_TOTAL_UPDATES}")
  if (( $# >= 4 )); then arguments+=(--exit-code "$4"); fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}
heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat --run-root "${S3_R6_RUN_ROOT}" \
      --candidate "${S3_R6_CANDIDATE_ID}" || true
    sleep 20
  done
}
on_exit() {
  code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  if (( code != 0 )) && (( COMPLETED == 0 )); then
    status failed run_s3_r6_candidate.sh \
      "candidate exited ${code}; see ${LOG_PATH}" "${code}" || true
  fi
}
trap on_exit EXIT
heartbeat_loop & HEARTBEAT_PID=$!

if [[ "${MICRO_ROUND}" == "R6J" ]]; then
  while [[ ! -f "${S3_R6_RUN_ROOT}/pairs/r6l_acceptance.json" ]]; do
    status queued run_s3_r6_candidate.sh \
      "phase2 queued; waiting for both R6L closed-loop results, GPU not occupied"
    if [[ -f "${S3_R6_FAILED_FILE}" ]]; then exit 4; fi
    sleep 10
  done
fi
while [[ ! -f "${S3_R6_READY_FILE}" ]]; do
  if [[ -f "${S3_R6_FAILED_FILE}" ]]; then exit 4; fi
  status waiting run_s3_r6_candidate.sh \
    "waiting for one-copy shared data/artifacts and accepted S2 parents"
  sleep 5
done

link_shared() {
  target="$1"; source="$2"
  if [[ -L "${target}" ]]; then
    if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then
      printf >&2 'Mismatched shared link: %s\n' "${target}"; return 3
    fi
  elif [[ -e "${target}" ]]; then
    printf >&2 'Refusing existing non-link shared path: %s\n' "${target}"; return 3
  else
    ln -s "${source}" "${target}" || return $?
  fi
}
status setup run_s3_r6_candidate.sh "linking shared datasets/artifacts; outputs isolated"
link_shared "${FE_ROOT}/datasets" "${S3_R6_BASE_REPO}/datasets" || exit $?
link_shared "${FE_ROOT}/artifacts" "${S3_R6_BASE_REPO}/artifacts" || exit $?
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export UV_CACHE_DIR="${S3_R6_UV_CACHE_DIR}"
export UV_PROJECT_ENVIRONMENT="${S3_R6_UV_ENV}"
export LPD_STAGE_LOG="${STAGE_LOG}"
unset HF_TOKEN

if [[ ! -f "${CHECKPOINT}" ]]; then
  if (( S3_R6_TOTAL_UPDATES > 0 )); then
    detail="adapter/gate-only five-task training; Flow and all world predictors frozen"
  else
    detail="zero-training off-path control composition and structural invariant check"
  fi
  status training train_s3_r6_world_action_flow.py "${detail}"
  ( cd "${FE_ROOT}" && PYTHONUNBUFFERED=1 uv run --frozen python \
    scripts/train_s3_r6_world_action_flow.py --config "${CONFIG}" --device cuda:0 \
    --updates "${S3_R6_TOTAL_UPDATES}" --output "${CHECKPOINT}" \
    --resume "${RESUME}" --progress-log "${TRAIN_PROGRESS}" ) || exit $?
fi

export ROBOFACTORY_ROOT="${S3_R6_ROBOFACTORY_ROOT}"
export RF_PYTHON="${S3_R6_RF_PYTHON}"
export LPD_POLICY_KIND=s3_flow
export LPD_EXPERIMENT_SLUG="s3_r6_${SLUG}"
export LPD_CONFIG="${CONFIG}"
export LPD_CHECKPOINT="${CHECKPOINT}"
export LPD_PORT="$((8872 + GPU_INDEX))"
export LPD_RUN_ID="${S3_R6_RUN_ID}_${SLUG}"
export LPD_EPISODES="${S3_R6_GATE_EPISODES:-20}"
export LPD_SEED_START="${S3_R6_GATE_SEED_START:-900}"
export LPD_OUTPUT_ROOT="${CANDIDATE_ROOT}/validation/gate_${S3_R6_RUN_ID}"
if [[ ! -f "${LPD_OUTPUT_ROOT}/gate_summary.json" ]]; then
  status validating run_lpd_fixed_seed_gate.sh \
    "paired Gate${LPD_EPISODES}: all five tasks; macro-average acceptance"
  bash "${FE_ROOT}/scripts/run_lpd_single_5090.sh" gate || exit $?
fi

mkdir -p "${S3_R6_RUN_ROOT}/pairs" || exit $?
exec {ACCEPT_FD}>"${S3_R6_RUN_ROOT}/.acceptance.lock"
flock -x "${ACCEPT_FD}"
PAIR_SLUG="$(printf '%s' "${MICRO_ROUND}" | tr '[:upper:]' '[:lower:]')"
P0_ROOT="${S3_R6_RUN_ROOT}/candidates/${PAIR_SLUG}_p0"
P1_ROOT="${S3_R6_RUN_ROOT}/candidates/${PAIR_SLUG}_p1"
P0_GATE="${P0_ROOT}/validation/gate_${S3_R6_RUN_ID}/gate_summary.json"
P1_GATE="${P1_ROOT}/validation/gate_${S3_R6_RUN_ID}/gate_summary.json"
P0_CHECKPOINT="${P0_ROOT}/checkpoints/policy.pt"
P1_CHECKPOINT="${P1_ROOT}/checkpoints/policy.pt"
PAIR_ACCEPTANCE="${S3_R6_RUN_ROOT}/pairs/${PAIR_SLUG}_acceptance.json"
if [[ -f "${P0_GATE}" && -f "${P1_GATE}" && ! -f "${PAIR_ACCEPTANCE}" ]]; then
  status accepting accept_s3_r6.py \
    "applying five-task macro-average P1>=P0 plus protected-own structural invariant"
  ( cd "${S3_R6_BASE_REPO}" && uv run --frozen python scripts/accept_s3_r6.py pair \
    --micro-round "${MICRO_ROUND}" --p0-gate "${P0_GATE}" --p1-gate "${P1_GATE}" \
    --p0-checkpoint "${P0_CHECKPOINT}" --p1-checkpoint "${P1_CHECKPOINT}" \
    --output "${PAIR_ACCEPTANCE}" ) || exit $?
fi
R6L_ACCEPTANCE="${S3_R6_RUN_ROOT}/pairs/r6l_acceptance.json"
R6J_ACCEPTANCE="${S3_R6_RUN_ROOT}/pairs/r6j_acceptance.json"
if [[ -f "${R6L_ACCEPTANCE}" && -f "${R6J_ACCEPTANCE}" && \
      ! -f "${S3_R6_RUN_ROOT}/acceptance.json" ]]; then
  ( cd "${S3_R6_BASE_REPO}" && uv run --frozen python scripts/accept_s3_r6.py final \
    --r6l "${R6L_ACCEPTANCE}" --r6j "${R6J_ACCEPTANCE}" \
    --output "${S3_R6_RUN_ROOT}/acceptance.json" ) || exit $?
fi
decision="pending paired candidate"
if [[ -f "${PAIR_ACCEPTANCE}" ]]; then decision="$(jq -r '.decision' "${PAIR_ACCEPTANCE}")"; fi
status complete run_s3_r6_candidate.sh \
  "checkpoint and paired closed-loop validation complete; ${decision}" 0
COMPLETED=1
printf 'S3-R6 %s complete: %s\n' "${S3_R6_CANDIDATE_ID}" "${CANDIDATE_ROOT}"
