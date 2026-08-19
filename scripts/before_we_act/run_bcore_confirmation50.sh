#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${BWA_CONFIRMATION_REPO_ROOT:-/workspace/fe-pc-wam}"
RUN_ROOT="${BWA_CONFIRMATION_RUN_ROOT:-/workspace/bwa_runs/b-core/n3-minimal-attribution-v1}"
CONTRACT="${BWA_CONFIRMATION_CONTRACT:-${ROOT}/docs/plans/contracts/b_core/n3_minimal_attribution/20260818/b_core_confirmation50_contract.json}"
PYTHON="${BWA_CONFIRMATION_PYTHON:-/venv/robofactory-act/bin/python}"
ROBOFACTORY_ROOT="${BWA_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
PHASE_ROOT="${RUN_ROOT}/confirmation50"
SEED_ROOT="${PHASE_ROOT}/seeds"
BCORE_ROOT="${PHASE_ROOT}/bcore"
DIRECT_ROOT="${PHASE_ROOT}/direct_reactive"
LOG_ROOT="${PHASE_ROOT}/logs"
STATUS="${PHASE_ROOT}/status.json"
SUMMARY="${PHASE_ROOT}/comparison.json"
export PYTHONPATH="${ROOT}:${ROOT}/vendor/stereo-core/stereo_core:${ROBOFACTORY_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
cd "${ROOT}"

write_status() {
  local status="$1" stage="$2" detail="$3"
  "${PYTHON}" - "${STATUS}" "${status}" "${stage}" "${detail}" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
value={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
tmp=p.with_name(f".{p.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
os.replace(tmp,p)
PY
}

PIDS=()
stop_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -INT "${pid}" 2>/dev/null || true
  done
}
on_error() {
  local code=$?
  write_status FAILED error "Confirmation50 exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'stop_children; write_status STOPPED interrupted "Confirmation50 interrupted"; exit 130' INT TERM

mkdir -p "${LOG_ROOT}" "${BCORE_ROOT}" "${DIRECT_ROOT}"
[[ -f "${CONTRACT}" ]] || { echo "missing frozen Confirmation50 contract" >&2; exit 2; }
[[ "$(jq -r '.status' "${CONTRACT}")" == "FROZEN_BEFORE_CONFIRMATION50_RESULTS" ]] || {
  echo "Confirmation50 contract is not frozen" >&2
  exit 2
}

for spec in bcore_checkpoint training_checkpoint b0h_checkpoint; do
  path="$(jq -r --arg spec "${spec}" '.immutable_inputs[$spec].path' "${CONTRACT}")"
  expected="$(jq -r --arg spec "${spec}" '.immutable_inputs[$spec].sha256' "${CONTRACT}")"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "immutable input drifted: ${spec}" >&2
    exit 2
  }
done

while IFS=$'\t' read -r path expected required_status; do
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "authorization evidence drifted: ${path}" >&2
    exit 2
  }
  [[ "$(jq -r '.status' "${path}")" == "${required_status}" ]] || {
    echo "authorization status is invalid: ${path}" >&2
    exit 2
  }
done < <(jq -r '.authorization_evidence | to_entries[] | [.value.path,.value.sha256,.value.required_status] | @tsv' "${CONTRACT}")

while IFS=$'\t' read -r path expected; do
  [[ -n "${path}" ]] || continue
  [[ -f "${ROOT}/${path}" && "$(sha256sum "${ROOT}/${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "frozen code drifted: ${path}" >&2
    exit 2
  }
done < <(jq -r '.code_sha256 | to_entries[] | [.key,.value] | @tsv' "${CONTRACT}")

write_status RUNNING prepare_seeds "materializing frozen independent seed manifests"
"${PYTHON}" scripts/before_we_act/prepare_bcore_confirmation50_seeds.py \
  --contract "${CONTRACT}" --output-root "${SEED_ROOT}" \
  >"${LOG_ROOT}/prepare_seeds.log" 2>&1

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
declare -A MAX_STEPS=(
  [lift_barrier]=500 [camera_alignment]=1500 [long_pipeline_delivery]=1500
  [take_photo]=1500 [pass_shoe]=500 [place_food]=500
)
for task in "${TASKS[@]}"; do
  expected="$(jq -r --arg task "${task}" '.seed_protocol.confirmation50_sha256[$task]' "${CONTRACT}")"
  [[ "$(sha256sum "${SEED_ROOT}/${task}.json" | awk '{print $1}')" == "${expected}" ]] || {
    echo "Confirmation50 seed file drifted: ${task}" >&2
    exit 2
  }
done

BCORE_CHECKPOINT="$(jq -r '.immutable_inputs.bcore_checkpoint.path' "${CONTRACT}")"
BCORE_SHA="$(jq -r '.immutable_inputs.bcore_checkpoint.sha256' "${CONTRACT}")"
TRAINING_CHECKPOINT="$(jq -r '.immutable_inputs.training_checkpoint.path' "${CONTRACT}")"
B0H_CHECKPOINT="$(jq -r '.immutable_inputs.b0h_checkpoint.path' "${CONTRACT}")"

is_complete() {
  local output="$1" mode="$2" checkpoint_sha="$3" seed_sha="$4"
  [[ -f "${output}" ]] && "${PYTHON}" - "${output}" "${mode}" "${checkpoint_sha}" "${seed_sha}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
ok=(value.get("mode")==sys.argv[2] and value.get("episodes")==50
    and len(value.get("rows",[]))==50
    and len({int(row["seed"]) for row in value.get("rows",[])})==50
    and value.get("checkpoint_sha256", value.get("training_checkpoint_sha256"))==sys.argv[3]
    and value.get("seed_protocol",{}).get("sha256")==sys.argv[4])
raise SystemExit(0 if ok else 1)
PY
}

launch_job() {
  local task="$1" mode="$2" gpu="$3"
  local seed_sha output log checkpoint_sha
  seed_sha="$(jq -r --arg task "${task}" '.seed_protocol.confirmation50_sha256[$task]' "${CONTRACT}")"
  if [[ "${mode}" == "n2" ]]; then
    output="${BCORE_ROOT}/${task}.json"
    log="${LOG_ROOT}/bcore_${task}.log"
    checkpoint_sha="${BCORE_SHA}"
    if is_complete "${output}" "${mode}" "${checkpoint_sha}" "${seed_sha}"; then
      printf '[%s] preserve complete mode=%s task=%s\n' "$(date -u +%FT%TZ)" "${mode}" "${task}"
      return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_predictive_team_belief \
      --checkpoint "${BCORE_CHECKPOINT}" --mode n2 --task "${task}" \
      --seed-file "${SEED_ROOT}/${task}.json" --episodes 50 \
      --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
      --resume-log "${log}" --output "${output}" >>"${log}" 2>&1 &
  else
    output="${DIRECT_ROOT}/${task}.json"
    log="${LOG_ROOT}/direct_${task}.log"
    checkpoint_sha="$(jq -r '.immutable_inputs.training_checkpoint.sha256' "${CONTRACT}")"
    if is_complete "${output}" "${mode}" "${checkpoint_sha}" "${seed_sha}"; then
      printf '[%s] preserve complete mode=%s task=%s\n' "$(date -u +%FT%TZ)" "${mode}" "${task}"
      return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m before_we_act.evaluate_direct_reactive \
      --b0h-checkpoint "${B0H_CHECKPOINT}" \
      --training-checkpoint "${TRAINING_CHECKPOINT}" \
      --task "${task}" --seed-file "${SEED_ROOT}/${task}.json" --episodes 50 \
      --max-steps "${MAX_STEPS[$task]}" --device cuda:0 \
      --resume-log "${log}" --output "${output}" >>"${log}" 2>&1 &
  fi
  PIDS+=("$!")
  printf '[%s] launch mode=%s task=%s gpu=%s pid=%s\n' \
    "$(date -u +%FT%TZ)" "${mode}" "${task}" "${gpu}" "$!"
}

run_wave() {
  PIDS=()
  local item task mode gpu pid code=0 child_code
  for item in "$@"; do
    IFS=: read -r task mode gpu <<<"${item}"
    launch_job "${task}" "${mode}" "${gpu}"
  done
  for pid in "${PIDS[@]:-}"; do
    set +e
    wait "${pid}"
    child_code=$?
    set -e
    ((child_code == 0)) || code="${child_code}"
  done
  PIDS=()
  ((code == 0))
}

write_status RUNNING confirmation50 "running 300 paired episodes / 600 policy rollouts"
run_wave camera_alignment:n2:0 camera_alignment:direct_reactive:1 long_pipeline_delivery:n2:2 long_pipeline_delivery:direct_reactive:3
run_wave take_photo:n2:0 take_photo:direct_reactive:1 lift_barrier:n2:2 lift_barrier:direct_reactive:3
run_wave pass_shoe:n2:0 pass_shoe:direct_reactive:1 place_food:n2:2 place_food:direct_reactive:3

"${PYTHON}" scripts/before_we_act/summarize_bcore_confirmation50.py \
  --contract "${CONTRACT}" --seed-root "${SEED_ROOT}" \
  --bcore-root "${BCORE_ROOT}" --direct-root "${DIRECT_ROOT}" \
  --output "${SUMMARY}" >"${LOG_ROOT}/summarize.log" 2>&1
DECISION="$(jq -r '.status' "${SUMMARY}")"
write_status COMPLETED decision "${DECISION}"
trap - ERR
printf 'B3_N3_CONFIRMATION50_COMPLETED summary=%s decision=%s\n' "${SUMMARY}" "${DECISION}"
