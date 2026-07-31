#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${S2_R4_HYBRID_RUN_ROOT:-}"
CONFIG="${S2_R4_HYBRID_CONFIG:-${FE_ROOT}/configs/wam_flow/s2_r4_hybrid_diagnostic.yaml}"
RUNTIME="${FE_ROOT}/scripts/s2_r4_hybrid_runtime.py"

if [[ -z "${RUN_ROOT}" ]]; then
  printf >&2 'Missing S2_R4_HYBRID_RUN_ROOT.\n'
  exit 2
fi

status() {
  python3 "${RUNTIME}" status --run-root "${RUN_ROOT}" "$@"
}

heartbeat_loop() {
  while true; do
    python3 "${RUNTIME}" heartbeat --run-root "${RUN_ROOT}" || return 0
    sleep 20
  done
}

status --phase waiting --program run_s2_r4_hybrid_evaluation.sh \
  --detail "waiting for protected source composition"
while [[ ! -f "${RUN_ROOT}/prepare.ready" ]]; do
  if [[ -f "${RUN_ROOT}/prepare.failed" ]]; then
    status --phase failed --program run_s2_r4_hybrid_evaluation.sh \
      --detail "prepare failed; see ${RUN_ROOT}/prepare.log" --exit-code 3
    exit 3
  fi
  python3 "${RUNTIME}" heartbeat --run-root "${RUN_ROOT}"
  printf '[%s] waiting for prepare.ready\n' "$(date -Is)"
  sleep 20
done

status --phase evaluating --program evaluate_s2_r4_hybrid_checkpoint.py \
  --detail "five-task exact-own/persistence/peer-shuffle diagnostic on GPU0"
heartbeat_loop &
HEARTBEAT_PID=$!
uv run --frozen python scripts/evaluate_s2_r4_hybrid_checkpoint.py \
  --config "${CONFIG}" \
  --hybrid-manifest "${RUN_ROOT}/hybrid_manifest.json" \
  --output "${RUN_ROOT}/hybrid_diagnostic.json" \
  --progress-log "${RUN_ROOT}/evaluation_progress.jsonl" \
  --device cuda:0 \
  2>&1 | tee "${RUN_ROOT}/evaluate.log"
RESULT=${PIPESTATUS[0]}
kill "${HEARTBEAT_PID}" 2>/dev/null
wait "${HEARTBEAT_PID}" 2>/dev/null
if [[ -f "${RUN_ROOT}/hybrid_diagnostic.json" ]]; then
  CONCLUSION="$(jq -r '.diagnostic.conclusion' "${RUN_ROOT}/hybrid_diagnostic.json")"
  PASSED="$(jq -r '.diagnostic.passed' "${RUN_ROOT}/hybrid_diagnostic.json")"
  status --phase complete --program evaluate_s2_r4_hybrid_checkpoint.py \
    --detail "diagnostic_passed=${PASSED} conclusion=${CONCLUSION}" \
    --exit-code "${RESULT}"
  printf 'S2-R4 diagnostic complete: passed=%s conclusion=%s\n' \
    "${PASSED}" "${CONCLUSION}"
  exit 0
fi
status --phase failed --program evaluate_s2_r4_hybrid_checkpoint.py \
  --detail "evaluation crashed before diagnostic; see ${RUN_ROOT}/evaluate.log" \
  --exit-code "${RESULT}"
exit "${RESULT}"

