#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FE_ROOT}/experiments/wam_flow/s2_r5/candidate.env"
for name in S2_R5_RUN_ROOT S2_R5_READY_FILE S2_R5_FAILED_FILE \
  S2_R5_BASE_REPO S2_R5_RUN_ID S2_R5_UV_CACHE_DIR S2_R5_UV_ENV GPU_INDEX; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'Missing %s\n' "${name}"; exit 2; fi
done
if [[ ! -f "${ENV_FILE}" ]]; then printf >&2 'Missing %s\n' "${ENV_FILE}"; exit 3; fi
# shellcheck source=/dev/null
source "${ENV_FILE}"
for name in S2_R5_CANDIDATE_ID S2_R5_TOTAL_UPDATES S2_R5_CONFIG_REL; do
  if [[ -z "${!name:-}" ]]; then printf >&2 'candidate.env missing %s\n' "${name}"; exit 2; fi
done

SLUG="$(printf '%s' "${S2_R5_CANDIDATE_ID}" | tr '[:upper:]' '[:lower:]')"
CANDIDATE_ROOT="${S2_R5_RUN_ROOT}/candidates/${SLUG}"
STATUS_TOOL="${S2_R5_BASE_REPO}/scripts/s2_r5_runtime.py"
CONFIG="${FE_ROOT}/${S2_R5_CONFIG_REL}"
CHECKPOINT="${CANDIDATE_ROOT}/checkpoints/predictor.pt"
RESUME="${CANDIDATE_ROOT}/checkpoints/resume.pt"
EVALUATION="${CANDIDATE_ROOT}/validation/evaluation.json"
TRAIN_PROGRESS="${CANDIDATE_ROOT}/train/progress.jsonl"
VALIDATION_PROGRESS="${CANDIDATE_ROOT}/validation/progress.jsonl"
STAGE_LOG="${CANDIDATE_ROOT}/train/stages.jsonl"
LOG_PATH="${CANDIDATE_ROOT}/logs/candidate.log"
HEARTBEAT_PID=""
COMPLETED=0
mkdir -p "${CANDIDATE_ROOT}"/{logs,train,validation,checkpoints,outputs}
exec > >(tee -a "${LOG_PATH}") 2>&1

status() {
  arguments=(status --run-root "${S2_R5_RUN_ROOT}" \
    --candidate "${S2_R5_CANDIDATE_ID}" --phase "$1" --program "$2" \
    --detail "${3:-}" --gpu-index "${GPU_INDEX}" \
    --total-updates "${S2_R5_TOTAL_UPDATES}")
  if (( $# >= 4 )); then arguments+=(--exit-code "$4"); fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}
heartbeat_loop() {
  while true; do
    python3 "${STATUS_TOOL}" heartbeat --run-root "${S2_R5_RUN_ROOT}" \
      --candidate "${S2_R5_CANDIDATE_ID}" || true
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
    status failed run_s2_r5_candidate.sh \
      "candidate exited ${code}; see ${LOG_PATH}" "${code}" || true
  fi
}
trap on_exit EXIT
heartbeat_loop & HEARTBEAT_PID=$!

while [[ ! -f "${S2_R5_READY_FILE}" ]]; do
  if [[ -f "${S2_R5_FAILED_FILE}" ]]; then exit 4; fi
  status waiting run_s2_r5_candidate.sh \
    "waiting for shared data/artifacts and protected R4-P0"
  sleep 5
done

link_shared() {
  target="$1"; source="$2"
  if [[ "$(realpath -m "${target}")" == "$(realpath -m "${source}")" ]]; then return; fi
  if [[ -L "${target}" ]]; then
    if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then exit 3; fi
  elif [[ -e "${target}" ]]; then exit 3
  else ln -s "${source}" "${target}"; fi
}
status setup run_s2_r5_candidate.sh "linking shared dataset/artifacts; outputs isolated"
link_shared "${FE_ROOT}/datasets" "${S2_R5_BASE_REPO}/datasets"
link_shared "${FE_ROOT}/artifacts" "${S2_R5_BASE_REPO}/artifacts"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export UV_CACHE_DIR="${S2_R5_UV_CACHE_DIR}"
export UV_PROJECT_ENVIRONMENT="${S2_R5_UV_ENV}"
export LPD_STAGE_LOG="${STAGE_LOG}"
unset HF_TOKEN

if [[ ! -f "${CHECKPOINT}" ]]; then
  status training train_s2_r5_protected_team.py \
    "team-only five-task training; protected own excluded from optimizer"
  ( cd "${FE_ROOT}" && PYTHONUNBUFFERED=1 uv run --frozen python \
    scripts/train_s2_r5_protected_team.py --config "${CONFIG}" --device cuda:0 \
    --updates "${S2_R5_TOTAL_UPDATES}" --output "${CHECKPOINT}" \
    --resume "${RESUME}" --progress-log "${TRAIN_PROGRESS}" )
  code=$?; if (( code != 0 )); then exit "${code}"; fi
fi

if [[ ! -f "${EVALUATION}" ]]; then
  status validating evaluate_s2_r5_protected_team.py \
    "fixed five-task validation; exact-own, persistence and shuffle-CI gates"
  ( cd "${FE_ROOT}" && PYTHONUNBUFFERED=1 uv run --frozen python \
    scripts/evaluate_s2_r5_protected_team.py --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" --output "${EVALUATION}" \
    --progress-log "${VALIDATION_PROGRESS}" --device cuda:0 )
  code=$?; if (( code != 0 )); then exit "${code}"; fi
fi

exec {ACCEPT_FD}>"${S2_R5_RUN_ROOT}/.acceptance.lock"
flock -x "${ACCEPT_FD}"
P0_EVAL="${S2_R5_RUN_ROOT}/candidates/p0/validation/evaluation.json"
P1_EVAL="${S2_R5_RUN_ROOT}/candidates/p1/validation/evaluation.json"
if [[ -f "${P0_EVAL}" && -f "${P1_EVAL}" && ! -f "${S2_R5_RUN_ROOT}/acceptance.json" ]]; then
  status accepting accept_s2_r5.py "applying special R5 gate and winner rule"
  ( cd "${S2_R5_BASE_REPO}" && uv run --frozen python scripts/accept_s2_r5.py \
    --p0 "${P0_EVAL}" --p1 "${P1_EVAL}" \
    --output "${S2_R5_RUN_ROOT}/acceptance.json" )
  code=$?; if (( code != 0 )); then exit "${code}"; fi
fi
decision="pending peer evaluation"
if [[ -f "${S2_R5_RUN_ROOT}/acceptance.json" ]]; then
  decision="$(jq -r '.decision' "${S2_R5_RUN_ROOT}/acceptance.json")"
fi
status complete run_s2_r5_candidate.sh "training/evaluation complete; ${decision}" 0
COMPLETED=1
printf 'S2-R5 %s complete: %s\n' "${S2_R5_CANDIDATE_ID}" "${CANDIDATE_ROOT}"
