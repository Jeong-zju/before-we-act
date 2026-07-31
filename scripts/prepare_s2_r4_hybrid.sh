#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${S2_R4_HYBRID_RUN_ROOT:-}"
OWN_SOURCE="${S2_R4_HYBRID_OWN_SOURCE:-}"
TEAM_SOURCE="${S2_R4_HYBRID_TEAM_SOURCE:-}"
CONFIG="${S2_R4_HYBRID_CONFIG:-${FE_ROOT}/configs/wam_flow/s2_r4_hybrid_diagnostic.yaml}"
RUNTIME="${FE_ROOT}/scripts/s2_r4_hybrid_runtime.py"

if [[ -z "${RUN_ROOT}" || -z "${OWN_SOURCE}" || -z "${TEAM_SOURCE}" ]]; then
  printf >&2 'Missing S2_R4_HYBRID_RUN_ROOT/source environment.\n'
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

status --phase composing --program compose_s2_r4_hybrid_checkpoint.py \
  --detail "validating P0/P1 hashes and writing an evaluate-only reference manifest"
heartbeat_loop &
HEARTBEAT_PID=$!
uv run --frozen python scripts/compose_s2_r4_hybrid_checkpoint.py \
  --config "${CONFIG}" \
  --own-source "${OWN_SOURCE}" \
  --team-source "${TEAM_SOURCE}" \
  --output "${RUN_ROOT}/hybrid_manifest.json" \
  >>"${RUN_ROOT}/prepare.log" 2>&1
RESULT=$?
kill "${HEARTBEAT_PID}" 2>/dev/null
wait "${HEARTBEAT_PID}" 2>/dev/null
if (( RESULT != 0 )); then
  touch "${RUN_ROOT}/prepare.failed"
  status --phase failed --program compose_s2_r4_hybrid_checkpoint.py \
    --detail "source composition failed; see ${RUN_ROOT}/prepare.log" \
    --exit-code "${RESULT}"
  exit "${RESULT}"
fi
touch "${RUN_ROOT}/prepare.ready"
status --phase ready --program compose_s2_r4_hybrid_checkpoint.py \
  --detail "source manifest ready; no optimizer/statistics/training created"
printf 'S2-R4 hybrid source manifest ready: %s\n' \
  "${RUN_ROOT}/hybrid_manifest.json"

