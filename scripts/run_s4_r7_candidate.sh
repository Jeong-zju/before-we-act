#!/usr/bin/env bash

# Run one S4-R7 candidate.  Every fallible operation is checked explicitly so
# failures remain visible in the permanent tmux window and runtime monitor.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_NAME="run_s4_r7_candidate.sh"
RUN_ID=""
RUN_ROOT_INPUT=""
READY_FILE_INPUT=""
FAILED_FILE_INPUT=""
CONFIG_INPUT=""
CANDIDATE=""
GPU_INDEX=""
HEARTBEAT_SECONDS=""
TOTAL_UPDATES=125000
PREFLIGHT_UPDATES=200
EARLY_STATUS_ACTIVE=0
STATUS_TOOL=""

usage() {
  printf 'usage: %s --candidate P0|P1 --run-id ID --run-root PATH --ready-file PATH --failed-file PATH --config PATH --gpu-index 0|1 --heartbeat-seconds 20\n' "$0"
}

fail_early() {
  local message="$*"
  printf >&2 'S4-R7 candidate runner error: %s\n' "${message}"
  if (( EARLY_STATUS_ACTIVE == 1 )) && [[ -f "${STATUS_TOOL}" ]]; then
    python3 "${STATUS_TOOL}" status --run-root "${RUN_ROOT}" \
      --candidate "${CANDIDATE}" --phase failed --program "${RUNNER_NAME}" \
      --detail "startup validation failed: ${message}" --pid "$$" \
      --child-pid 0 --gpu-index "${GPU_INDEX}" --gpu-pid 0 \
      --total-updates "${TOTAL_UPDATES}" --exit-code 3 || true
  fi
  exit 3
}

while (( $# )); do
  case "$1" in
    --candidate)
      if (( $# < 2 )); then printf >&2 '%s\n' '--candidate requires a value'; exit 2; fi
      CANDIDATE="$2"; shift 2
      ;;
    --run-id)
      if (( $# < 2 )); then printf >&2 '%s\n' '--run-id requires a value'; exit 2; fi
      RUN_ID="$2"; shift 2
      ;;
    --run-root)
      if (( $# < 2 )); then printf >&2 '%s\n' '--run-root requires a value'; exit 2; fi
      RUN_ROOT_INPUT="$2"; shift 2
      ;;
    --ready-file)
      if (( $# < 2 )); then printf >&2 '%s\n' '--ready-file requires a value'; exit 2; fi
      READY_FILE_INPUT="$2"; shift 2
      ;;
    --failed-file)
      if (( $# < 2 )); then printf >&2 '%s\n' '--failed-file requires a value'; exit 2; fi
      FAILED_FILE_INPUT="$2"; shift 2
      ;;
    --config)
      if (( $# < 2 )); then printf >&2 '%s\n' '--config requires a value'; exit 2; fi
      CONFIG_INPUT="$2"; shift 2
      ;;
    --gpu-index)
      if (( $# < 2 )); then printf >&2 '%s\n' '--gpu-index requires a value'; exit 2; fi
      GPU_INDEX="$2"; shift 2
      ;;
    --heartbeat-seconds)
      if (( $# < 2 )); then printf >&2 '%s\n' '--heartbeat-seconds requires a value'; exit 2; fi
      HEARTBEAT_SECONDS="$2"; shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      printf >&2 'Unknown argument: %s\n' "$1"
      exit 2
      ;;
  esac
done

for required_value in RUN_ID RUN_ROOT_INPUT READY_FILE_INPUT FAILED_FILE_INPUT \
  CONFIG_INPUT CANDIDATE GPU_INDEX HEARTBEAT_SECONDS; do
  if [[ -z "${!required_value}" ]]; then
    usage >&2
    printf >&2 'Missing required argument for %s\n' "${required_value}"
    exit 2
  fi
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid run id: %s\n' "${RUN_ID}"
  exit 2
fi
case "${CANDIDATE}" in
  P0)
    EXPECTED_GPU=0
    EXPECTED_BRANCH="s4/r7-p0-token-preserving-evidence"
    EXPECTED_MODEL_KIND="s4_r7_token_preserving"
    ;;
  P1)
    EXPECTED_GPU=1
    EXPECTED_BRANCH="s4/r7-p1-world-utility-coupling"
    EXPECTED_MODEL_KIND="s4_r7_world_utility_coupling"
    ;;
  *)
    printf >&2 'Invalid S4-R7 candidate: %s\n' "${CANDIDATE}"
    exit 2
    ;;
esac
if [[ "${GPU_INDEX}" != "${EXPECTED_GPU}" ]]; then
  printf >&2 'GPU mapping mismatch: %s must use physical GPU %s, got %s\n' \
    "${CANDIDATE}" "${EXPECTED_GPU}" "${GPU_INDEX}"
  exit 2
fi
if [[ "${HEARTBEAT_SECONDS}" != "20" ]]; then
  printf >&2 'S4-R7 heartbeat interval is fixed at 20 seconds, got %s\n' \
    "${HEARTBEAT_SECONDS}"
  exit 2
fi

for required_command in python3 uv jq realpath sha256sum flock git tee grep awk \
  wc ln mv kill sleep tr nvidia-smi date; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    fail_early "missing required command: ${required_command}"
  fi
done
RUNNER_START_EPOCH="$(date +%s)" || fail_early "could not read startup time"

RUN_ROOT="$(realpath -m -- "${RUN_ROOT_INPUT}")" || \
  fail_early "cannot resolve run root: ${RUN_ROOT_INPUT}"
READY_FILE="$(realpath -m -- "${READY_FILE_INPUT}")" || \
  fail_early "cannot resolve ready file: ${READY_FILE_INPUT}"
FAILED_FILE="$(realpath -m -- "${FAILED_FILE_INPUT}")" || \
  fail_early "cannot resolve failed file: ${FAILED_FILE_INPUT}"
CONFIG="$(realpath -e -- "${CONFIG_INPUT}")" || \
  fail_early "candidate config does not exist: ${CONFIG_INPUT}"
MANIFEST="${RUN_ROOT}/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  fail_early "missing immutable run manifest: ${MANIFEST}"
fi

python3 - "${MANIFEST}" "${RUN_ROOT}" "${RUN_ID}" "${FE_ROOT}" \
  "${CANDIDATE}" "${GPU_INDEX}" "${CONFIG}" "${READY_FILE}" \
  "${FAILED_FILE}" <<'PY'
import json
from pathlib import Path
import sys

(
    manifest_path,
    run_root_raw,
    run_id,
    worktree_raw,
    candidate,
    gpu_raw,
    config_raw,
    ready_raw,
    failed_raw,
) = sys.argv[1:]
manifest = json.loads(Path(manifest_path).resolve(strict=True).read_text())
if not isinstance(manifest, dict):
    raise ValueError("run manifest must be a JSON object")
root = Path(run_root_raw).resolve(strict=True)
worktree = Path(worktree_raw).resolve(strict=True)
config = Path(config_raw).resolve(strict=True)
expected_branch = {
    "P0": "s4/r7-p0-token-preserving-evidence",
    "P1": "s4/r7-p1-world-utility-coupling",
}[candidate]
checks = {
    "format_version": manifest.get("format_version") == "wam.robofactory.s4_r7.runtime/1",
    "round_id": manifest.get("round_id") == "s4-r7",
    "run_id": manifest.get("run_id") == run_id,
    "run_root": Path(str(manifest.get("run_root", ""))).resolve() == root,
    "permanent_tmux": manifest.get("tmux_session") == "ssh_tmux",
    "heartbeat_seconds": manifest.get("heartbeat_seconds") == 20,
    "candidate_worktree": Path(str(manifest.get("worktrees", {}).get(candidate, ""))).resolve() == worktree,
    "gpu_mapping": manifest.get("gpu_assignment", {}).get(candidate) == int(gpu_raw),
    "branch": manifest.get("branches", {}).get(candidate) == expected_branch,
    "ready_path": Path(ready_raw).resolve() == root / "shared.ready",
    "failed_path": Path(failed_raw).resolve() == root / "shared.failed",
    "config_path": config == worktree / "configs/wam_flow/s4_r7.yaml",
    "total_updates": manifest.get("training", {}).get("updates") == 125000,
    "effective_batch": manifest.get("training", {}).get("effective_team_batch") == 12,
}
base = Path(str(manifest.get("base_repo", ""))).resolve(strict=True)
checks["shared_data"] = Path(str(manifest.get("shared_data", ""))).resolve() == base / "datasets/robofactory_multitask"
checks["shared_artifacts"] = Path(str(manifest.get("shared_artifacts", ""))).resolve() == base / "artifacts"
worktrees = manifest.get("worktrees", {})
checks["both_worktrees"] = all(
    Path(str(worktrees.get(name, ""))).resolve(strict=True).is_dir()
    for name in ("P0", "P1")
)
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise ValueError(f"run manifest/candidate identity failed: {failed}")
PY
manifest_code=$?
if (( manifest_code != 0 )); then
  fail_early "run manifest or launcher argument identity validation failed"
fi

BASE_REPO="$(jq -er '.base_repo | strings' "${MANIFEST}")" || \
  fail_early "invalid base_repo in ${MANIFEST}"
BASE_REPO="$(realpath -e -- "${BASE_REPO}")" || \
  fail_early "base repository is unavailable"
STATUS_TOOL="${BASE_REPO}/scripts/s4_r7_runtime.py"
if [[ ! -f "${STATUS_TOOL}" ]]; then
  fail_early "missing S4-R7 runtime status program: ${STATUS_TOOL}"
fi
SLUG="$(printf '%s' "${CANDIDATE}" | tr '[:upper:]' '[:lower:]')"
CANDIDATE_ROOT="${RUN_ROOT}/candidates/${SLUG}"
EARLY_STATUS_ACTIVE=1
if ! python3 "${STATUS_TOOL}" status --run-root "${RUN_ROOT}" \
  --candidate "${CANDIDATE}" --phase starting --program "${RUNNER_NAME}" \
  --detail "validating branch, candidate.env, config registry and GPU identity" \
  --pid "$$" --child-pid 0 --gpu-index "${GPU_INDEX}" --gpu-pid 0 \
  --total-updates "${TOTAL_UPDATES}"; then
  fail_early "could not publish startup validation status"
fi
P0_WORKTREE="$(jq -er '.worktrees.P0 | strings' "${MANIFEST}")" || \
  fail_early "missing P0 worktree in run manifest"
P1_WORKTREE="$(jq -er '.worktrees.P1 | strings' "${MANIFEST}")" || \
  fail_early "missing P1 worktree in run manifest"
P0_WORKTREE="$(realpath -e -- "${P0_WORKTREE}")" || fail_early "P0 worktree is unavailable"
P1_WORKTREE="$(realpath -e -- "${P1_WORKTREE}")" || fail_early "P1 worktree is unavailable"
P0_CONFIG="$(realpath -e -- "${P0_WORKTREE}/configs/wam_flow/s4_r7.yaml")" || \
  fail_early "P0 config is unavailable"
P1_CONFIG="$(realpath -e -- "${P1_WORKTREE}/configs/wam_flow/s4_r7.yaml")" || \
  fail_early "P1 config is unavailable"

CURRENT_BRANCH="$(git -C "${FE_ROOT}" branch --show-current)" || \
  fail_early "cannot identify current candidate branch"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  fail_early "candidate ${CANDIDATE} must run from ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH}"
fi
PARENT_COMMIT="$(jq -er '.parent_commit | strings' "${MANIFEST}")" || \
  fail_early "invalid parent commit in run manifest"
if ! git -C "${FE_ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" HEAD; then
  fail_early "candidate branch does not descend from the manifest parent commit"
fi

ENV_FILE="${FE_ROOT}/experiments/wam_flow/s4_r7/candidate.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  fail_early "missing candidate identity file: ${ENV_FILE}"
fi
python3 - "${ENV_FILE}" "${CANDIDATE}" "${CONFIG}" "${FE_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

env_path, candidate, config_raw, root_raw = sys.argv[1:]
expected_keys = {
    "S4_R7_CANDIDATE_ID",
    "S4_R7_TOTAL_UPDATES",
    "S4_R7_CONFIG_REL",
}
values = {}
for line_number, raw in enumerate(Path(env_path).read_text().splitlines(), 1):
    if not raw:
        continue
    match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./-]+)", raw)
    if match is None:
        raise ValueError(f"candidate.env line {line_number} is not a literal assignment")
    key, value = match.groups()
    if key in values:
        raise ValueError(f"candidate.env duplicates {key}")
    values[key] = value
if set(values) != expected_keys:
    raise ValueError(f"candidate.env keys differ: {sorted(values)}")
if values["S4_R7_CANDIDATE_ID"] != candidate:
    raise ValueError("candidate.env candidate identity differs from launcher")
if values["S4_R7_TOTAL_UPDATES"] != "125000":
    raise ValueError("candidate.env training budget must be exactly 125000")
if values["S4_R7_CONFIG_REL"] != "configs/wam_flow/s4_r7.yaml":
    raise ValueError("candidate.env config path is outside the registered public slice")
expected_config = Path(root_raw).resolve(strict=True) / values["S4_R7_CONFIG_REL"]
if Path(config_raw).resolve(strict=True) != expected_config.resolve(strict=True):
    raise ValueError("candidate.env config path differs from --config")
PY
env_code=$?
if (( env_code != 0 )); then
  fail_early "candidate.env identity validation failed"
fi

CONFIG_RECIPE="$( cd "${BASE_REPO}" && uv run --frozen python - "${CONFIG}" \
    "${CANDIDATE}" "${EXPECTED_MODEL_KIND}" <<'PY'
from pathlib import Path
import sys

from scripts.train_static_rgb_act_moe import _load_yaml
from train.s4_model_registry import validate_s4_r7_candidate

path, expected_candidate, expected_kind = sys.argv[1:]
config = _load_yaml(Path(path).resolve(strict=True))
candidate, kind, utility = validate_s4_r7_candidate(config)
if candidate != expected_candidate or kind != expected_kind:
    raise ValueError("config registry identity differs from candidate branch")
expected_utility = 0.0 if expected_candidate == "P0" else 0.05
if utility != expected_utility:
    raise ValueError("config utility axis differs from the S4-R7 registry")
if config.get("format_version") != "wam.robofactory.s4_r7.world_utility.config/1":
    raise ValueError("unsupported S4-R7 config format")
training = config.get("training", {})
if (
    training.get("updates") != 125000
    or training.get("preflight_updates") != 200
    or training.get("effective_team_batch") != 12
    or (
        training.get("micro_team_batch"),
        training.get("gradient_accumulation"),
    ) not in ((2, 6), (1, 12))
):
    raise ValueError("config budget or initial paired micro-batch recipe differs")
print(
    int(training["micro_team_batch"]),
    int(training["gradient_accumulation"]),
    int(training["effective_team_batch"]),
)
PY
)"
registry_code=$?
if (( registry_code != 0 )); then
  fail_early "config format/model allowlist validation failed"
fi
read -r MICRO_TEAM_BATCH GRADIENT_ACCUMULATION EFFECTIVE_TEAM_BATCH \
  <<< "${CONFIG_RECIPE}"
if [[ ! "${MICRO_TEAM_BATCH}" =~ ^[0-9]+$ || \
      ! "${GRADIENT_ACCUMULATION}" =~ ^[0-9]+$ || \
      "${EFFECTIVE_TEAM_BATCH}" != "12" ]]; then
  fail_early "could not read the validated micro/accum/effective recipe"
fi

PAIR_VALIDATOR="${BASE_REPO}/scripts/validate_s4_r7_branch_pair.py"
ACCEPT_TOOL="${BASE_REPO}/scripts/accept_s4_r7.py"
TRAINER="${FE_ROOT}/scripts/train_s4_r7_world_utility.py"
EVALUATOR="${FE_ROOT}/scripts/evaluate_s4_r7_causal.py"
for required_script in "${STATUS_TOOL}" "${PAIR_VALIDATOR}" "${ACCEPT_TOOL}"; do
  if [[ ! -f "${required_script}" ]]; then
    fail_early "missing shared S4-R7 control program: ${required_script}"
  fi
done
if ! ( cd "${BASE_REPO}" && uv run --frozen python "${PAIR_VALIDATOR}" \
  --p0-config "${P0_CONFIG}" --p1-config "${P1_CONFIG}" --config-only \
  >/dev/null ); then
  fail_early "P0/P1 registered config pair differs before preflight"
fi

PREFLIGHT_ROOT="${CANDIDATE_ROOT}/preflight"
TRAIN_ROOT="${CANDIDATE_ROOT}/train"
VALIDATION_ROOT="${CANDIDATE_ROOT}/validation"
CHECKPOINT_ROOT="${CANDIDATE_ROOT}/checkpoints"
LOG_ROOT="${CANDIDATE_ROOT}/logs"
PAIR_ROOT="${RUN_ROOT}/pairs"
PREFLIGHT_REPORT="${PREFLIGHT_ROOT}/preflight.json"
PREFLIGHT_CHECKPOINT="${PREFLIGHT_ROOT}/checkpoints/policy.pt"
PREFLIGHT_RESUME="${PREFLIGHT_ROOT}/checkpoints/resume.pt"
PREFLIGHT_PROGRESS="${PREFLIGHT_ROOT}/progress.jsonl"
CHECKPOINT="${CHECKPOINT_ROOT}/policy.pt"
RESUME="${CHECKPOINT_ROOT}/resume.pt"
TRAIN_PROGRESS="${TRAIN_ROOT}/progress.jsonl"
CANDIDATE_REPORT="${VALIDATION_ROOT}/candidate_report.json"
VALIDATION_PROGRESS="${VALIDATION_ROOT}/progress.jsonl"
PAIR_EXACT="${PAIR_ROOT}/pair_exact.json"
ACCEPTANCE="${RUN_ROOT}/acceptance.json"
LOG_PATH="${LOG_ROOT}/candidate.log"
P0_ROOT="${RUN_ROOT}/candidates/p0"
P1_ROOT="${RUN_ROOT}/candidates/p1"
P0_PREFLIGHT="${P0_ROOT}/preflight/preflight.json"
P1_PREFLIGHT="${P1_ROOT}/preflight/preflight.json"
P0_PREFLIGHT_PROVENANCE="${P0_PREFLIGHT}.provenance.json"
P1_PREFLIGHT_PROVENANCE="${P1_PREFLIGHT}.provenance.json"
P0_CHECKPOINT="${P0_ROOT}/checkpoints/policy.pt"
P1_CHECKPOINT="${P1_ROOT}/checkpoints/policy.pt"
P0_REPORT="${P0_ROOT}/validation/candidate_report.json"
P1_REPORT="${P1_ROOT}/validation/candidate_report.json"
P0_CONFIG_SHA256="$(sha256sum "${P0_CONFIG}" | awk '{print $1}')" || \
  fail_early "could not hash P0 config"
P1_CONFIG_SHA256="$(sha256sum "${P1_CONFIG}" | awk '{print $1}')" || \
  fail_early "could not hash P1 config"
CONFIG_SHA256="${P0_CONFIG_SHA256}"
if [[ "${CANDIDATE}" == "P1" ]]; then CONFIG_SHA256="${P1_CONFIG_SHA256}"; fi
[[ -n "${CONFIG_SHA256}" ]] || \
  fail_early "could not hash candidate config"

mkdir -p "${PREFLIGHT_ROOT}" "${TRAIN_ROOT}" "${VALIDATION_ROOT}" \
  "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${PAIR_ROOT}" || \
  fail_early "cannot create candidate-isolated run directories"
exec > >(tee -a "${LOG_PATH}") 2>&1

CURRENT_PHASE="starting"
CURRENT_PROGRAM="${RUNNER_NAME}"
CURRENT_DETAIL="validating immutable candidate identity"
CURRENT_PREFLIGHT="pending"
CHILD_PID=""
HEARTBEAT_PID=""
COMPLETED=0
STATUS_ACTIVE=1
FAILURE_DETAIL=""
EARLY_STATUS_ACTIVE=0

current_gpu_pid() {
  local observed
  observed="$(nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print; exit}')"
  if [[ "${observed}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${observed}"
  else
    printf '0'
  fi
}

publish_status() {
  local exit_code="${1:-}"
  local child_value=0
  local gpu_pid_value
  local status_total_updates="${TOTAL_UPDATES}"
  local arguments
  if [[ -n "${CHILD_PID}" ]]; then
    child_value="${CHILD_PID}"
  fi
  gpu_pid_value="$(current_gpu_pid)"
  case "${CURRENT_PHASE}" in
    preflight|waiting_peer_preflight|waiting_pair_lock|pair_validation)
      status_total_updates="${PREFLIGHT_UPDATES}"
      ;;
  esac
  arguments=(status --run-root "${RUN_ROOT}" --candidate "${CANDIDATE}"
    --phase "${CURRENT_PHASE}" --program "${CURRENT_PROGRAM}"
    --detail "${CURRENT_DETAIL}" --pid "$$" --child-pid "${child_value}"
    --gpu-index "${GPU_INDEX}" --gpu-pid "${gpu_pid_value}"
    --micro-batch "${MICRO_TEAM_BATCH}" \
    --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
    --effective-batch "${EFFECTIVE_TEAM_BATCH}"
    --total-updates "${status_total_updates}" --preflight "${CURRENT_PREFLIGHT}")
  if [[ -n "${exit_code}" ]]; then arguments+=(--exit-code "${exit_code}"); fi
  python3 "${STATUS_TOOL}" "${arguments[@]}"
}

set_stage() {
  CURRENT_PHASE="$1"
  CURRENT_PROGRAM="$2"
  CURRENT_DETAIL="$3"
  if ! publish_status; then
    printf >&2 'Warning: failed to publish status phase=%s program=%s\n' \
      "${CURRENT_PHASE}" "${CURRENT_PROGRAM}"
    return 3
  fi
}

heartbeat_loop() {
  local gpu_pid_value
  while true; do
    gpu_pid_value="$(current_gpu_pid)"
    python3 "${STATUS_TOOL}" heartbeat --run-root "${RUN_ROOT}" \
      --candidate "${CANDIDATE}" --pid "$$" \
      --gpu-pid "${gpu_pid_value}" || true
    sleep "${HEARTBEAT_SECONDS}" || return 0
  done
}

on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  FAILURE_DETAIL="received ${signal_name}; candidate child was asked to stop"
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -s "${signal_name}" "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
    CHILD_PID=""
  fi
  exit "${exit_code}"
}

on_exit() {
  local exit_code=$?
  if [[ -n "${HEARTBEAT_PID}" ]]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  if (( exit_code != 0 )) && (( COMPLETED == 0 )) && (( STATUS_ACTIVE == 1 )); then
    CHILD_PID=""
    CURRENT_PHASE="failed"
    if [[ -n "${FAILURE_DETAIL}" ]]; then
      CURRENT_DETAIL="${FAILURE_DETAIL}; exit=${exit_code}; log=${LOG_PATH}"
    else
      CURRENT_DETAIL="${CURRENT_PROGRAM} exited ${exit_code}; log=${LOG_PATH}"
    fi
    publish_status "${exit_code}" || true
  fi
}

trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
heartbeat_loop & HEARTBEAT_PID=$!
if ! set_stage starting "${RUNNER_NAME}" \
  "candidate=${CANDIDATE} branch=${EXPECTED_BRANCH} GPU=${GPU_INDEX} pid=$$"; then
  FAILURE_DETAIL="initial runtime status publication failed"
  exit 3
fi

abort() {
  FAILURE_DETAIL="$*"
  printf >&2 'S4-R7 %s error: %s\n' "${CANDIDATE}" "${FAILURE_DETAIL}"
  exit 3
}

run_stage() {
  local phase="$1"
  local program="$2"
  local detail="$3"
  local stage_code
  shift 3
  if ! set_stage "${phase}" "${program}" "${detail}"; then
    FAILURE_DETAIL="could not publish ${phase}/${program} status"
    return 3
  fi
  printf 'Starting phase=%s program=%s\n' "${phase}" "${program}"
  ( cd "${FE_ROOT}" && env PYTHONUNBUFFERED=1 "$@" ) &
  CHILD_PID=$!
  if ! publish_status; then
    printf >&2 'Warning: could not publish child PID %s for %s\n' \
      "${CHILD_PID}" "${program}"
  fi
  wait "${CHILD_PID}"
  stage_code=$?
  CHILD_PID=""
  CURRENT_DETAIL="${detail}; child exited ${stage_code}"
  publish_status "${stage_code}" || true
  return "${stage_code}"
}

peer_failed() {
  local peer="$1"
  local peer_status="${RUN_ROOT}/candidates/$(printf '%s' "${peer}" | tr '[:upper:]' '[:lower:]')/status.json"
  local now_epoch
  [[ -f "${peer_status}" ]] || return 1
  [[ "$(jq -r '.phase // empty' "${peer_status}" 2>/dev/null)" == "failed" ]] || return 1
  # A repaired tmux window can briefly see the peer's terminal status from the
  # previous process. Give the launcher one minute to respawn both windows;
  # after that, an unchanged failed peer is a real blocker, not a reason to
  # wait forever.
  now_epoch="$(date +%s)" || return 0
  if (( now_epoch - RUNNER_START_EPOCH < 60 )); then return 1; fi
  return 0
}

ensure_digest() {
  local path="$1"
  local sidecar="${path}.sha256"
  local observed expected temporary
  if [[ ! -f "${path}" ]]; then
    printf >&2 'Cannot hash missing artifact: %s\n' "${path}"
    return 3
  fi
  observed="$(sha256sum "${path}" | awk '{print $1}')" || return $?
  if [[ -f "${sidecar}" ]]; then
    expected="$(tr -d '[:space:]' < "${sidecar}")" || return $?
    if [[ ! "${expected}" =~ ^[0-9a-f]{64}$ || "${expected}" != "${observed}" ]]; then
      printf >&2 'Artifact hash sidecar mismatch: %s\n' "${path}"
      return 3
    fi
    return 0
  fi
  temporary="${sidecar}.tmp.$$"
  printf '%s\n' "${observed}" > "${temporary}" || return $?
  mv "${temporary}" "${sidecar}" || return $?
}

verify_preflight() {
  local path="$1"
  local candidate="$2"
  local mode="${3:-pass}"
  local expected_kind
  local expected_config_sha
  if [[ "${candidate}" == "P0" ]]; then
    expected_kind="s4_r7_token_preserving"
    expected_config_sha="${P0_CONFIG_SHA256}"
  else
    expected_kind="s4_r7_world_utility_coupling"
    expected_config_sha="${P1_CONFIG_SHA256}"
  fi
  python3 - "${path}" "${candidate}" "${expected_kind}" \
    "${expected_config_sha}" "${MICRO_TEAM_BATCH}" \
    "${GRADIENT_ACCUMULATION}" "${mode}" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

path, candidate, expected_kind, config_sha256, micro_raw, accumulation_raw, mode = sys.argv[1:]
value = json.loads(Path(path).resolve(strict=True).read_text())
if not isinstance(value, dict):
    raise ValueError("preflight report must be a JSON object")
identity = value.get("identity")
if not isinstance(identity, dict):
    raise ValueError("preflight identity must be a mapping")
if (
    value.get("format_version") != "wam.robofactory.s4_r7.preflight/1"
    or identity.get("round_id") != "s4-r7"
    or identity.get("candidate_id") != candidate
    or identity.get("model_kind") != expected_kind
    or value.get("updates") != 200
):
    raise ValueError("preflight format/identity differs")
if value.get("micro_team_batch") != int(micro_raw) or value.get(
    "gradient_accumulation"
) != int(accumulation_raw) or value.get("effective_team_batch") != 12:
    raise ValueError("preflight batch recipe differs from the paired configs")
peak = value.get("peak_memory_bytes")
total = value.get("gpu_total_memory_bytes")
if (
    isinstance(peak, bool)
    or not isinstance(peak, (int, float))
    or isinstance(total, bool)
    or not isinstance(total, (int, float))
    or not math.isfinite(float(peak))
    or not math.isfinite(float(total))
    or float(peak) < 0.0
    or float(total) <= 0.0
    or float(peak) > float(total)
):
    raise ValueError("preflight GPU memory measurements are invalid")
if value.get("completed") is not True:
    if mode == "terminal" and value.get("completed") is False and value.get("oom") is True:
        raise SystemExit(0)
    raise ValueError("preflight did not complete 200 updates")
for key in (
    "dataset_index_sequence_sha256",
    "update_1_trainable_name_sha256",
    "update_26668_trainable_name_sha256",
    "learning_rate_curve_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))) is None:
        raise ValueError(f"preflight {key} is not a SHA256 digest")
histogram = value.get("agent_count_histogram")
if not isinstance(histogram, dict) or not histogram:
    raise ValueError("preflight agent_count_histogram is empty")
if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in histogram.values()):
    raise ValueError("preflight agent_count_histogram contains invalid counts")
if sum(histogram.values()) <= 0:
    raise ValueError("preflight agent_count_histogram contains no observations")
if (
    value.get("resume_next_batch_exact") is not True
    or value.get("oom") is not False
    or float(peak) <= 0.0
    or isinstance(value.get("forced_audit_seconds"), bool)
    or not isinstance(value.get("forced_audit_seconds"), (int, float))
    or float(value["forced_audit_seconds"]) <= 0.0
    or not math.isfinite(float(value["forced_audit_seconds"]))
    or isinstance(value.get("updates_per_second"), bool)
    or not isinstance(value.get("updates_per_second"), (int, float))
    or float(value["updates_per_second"]) <= 0.0
    or not math.isfinite(float(value["updates_per_second"]))
):
    raise ValueError("preflight runtime/batch/resume measurements are invalid")
reported_config = value.get("config_sha256")
if reported_config is not None and reported_config != config_sha256:
    raise ValueError("preflight config hash differs from the launched config")
PY
}

preflight_provenance() {
  local report="$1"
  local candidate="$2"
  local action="$3"
  local provenance="${report}.provenance.json"
  local candidate_worktree candidate_config candidate_config_sha candidate_kind
  if [[ "${candidate}" == "P0" ]]; then
    candidate_worktree="${P0_WORKTREE}"
    candidate_config="${P0_CONFIG}"
    candidate_config_sha="${P0_CONFIG_SHA256}"
    candidate_kind="s4_r7_token_preserving"
  else
    candidate_worktree="${P1_WORKTREE}"
    candidate_config="${P1_CONFIG}"
    candidate_config_sha="${P1_CONFIG_SHA256}"
    candidate_kind="s4_r7_world_utility_coupling"
  fi
  python3 - "${action}" "${provenance}" "${report}" "${candidate}" \
    "${candidate_kind}" "${RUN_ID}" "${candidate_config}" \
    "${candidate_config_sha}" "${candidate_worktree}/scripts/train_s4_r7_world_utility.py" \
    "$(git -C "${candidate_worktree}" rev-parse HEAD)" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

(
    action,
    provenance_raw,
    report_raw,
    candidate,
    model_kind,
    run_id,
    config_raw,
    config_sha256,
    trainer_raw,
    git_commit,
) = sys.argv[1:]
provenance = Path(provenance_raw).resolve()
report = Path(report_raw).resolve(strict=True)
config = Path(config_raw).resolve(strict=True)
trainer = Path(trainer_raw).resolve(strict=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

expected = {
    "format_version": "wam.robofactory.s4_r7.preflight_provenance/1",
    "round_id": "s4-r7",
    "run_id": run_id,
    "candidate_id": candidate,
    "model_kind": model_kind,
    "report_path": str(report),
    "report_sha256": sha256(report),
    "config_path": str(config),
    "config_sha256": config_sha256,
    "trainer_path": str(trainer),
    "trainer_sha256": sha256(trainer),
    "git_commit": git_commit,
}
if action == "verify":
    observed = json.loads(provenance.resolve(strict=True).read_text())
    if not isinstance(observed, dict):
        raise ValueError("preflight provenance must be a JSON object")
    observed.pop("created_at", None)
    if observed != expected:
        raise ValueError("preflight provenance differs from current report/config/code/run")
elif action == "create":
    if provenance.exists():
        raise FileExistsError(f"refusing to overwrite {provenance}")
    provenance.parent.mkdir(parents=True, exist_ok=True)
    payload = {**expected, "created_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=provenance.parent,
        prefix=f".{provenance.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(provenance)
else:
    raise ValueError(f"unknown provenance action {action!r}")
PY
}

verify_pair_exact() {
  python3 - "${PAIR_EXACT}" "${P0_PREFLIGHT}" "${P1_PREFLIGHT}" <<'PY'
import json
from pathlib import Path
import sys

pair_path, p0_path, p1_path = map(Path, sys.argv[1:])
pair = json.loads(pair_path.resolve(strict=True).read_text())
p0 = json.loads(p0_path.resolve(strict=True).read_text())
p1 = json.loads(p1_path.resolve(strict=True).read_text())
if not isinstance(pair, dict):
    raise ValueError("pair_exact must be a JSON object")
checks = pair.get("checks")
preflight = pair.get("preflight")
if (
    pair.get("format_version") != "wam.robofactory.s4_r7.pair_exact/1"
    or pair.get("round_id") != "s4-r7"
    or pair.get("scope") != "paired_200_step_preflight"
    or pair.get("passed") is not True
    or not isinstance(checks, dict)
    or not checks
    or any(passed is not True for passed in checks.values())
    or pair.get("candidate_axis") != {"name": "utility_coupling_weight", "P0": 0.0, "P1": 0.05}
    or not isinstance(preflight, dict)
    or preflight.get("P0") != p0
    or preflight.get("P1") != p1
    or preflight.get("required_fallback") is not None
):
    raise ValueError("pair_exact content/identity does not match current preflights")
PY
}

verify_checkpoint() {
  local path="$1"
  local candidate="$2"
  local mode="$3"
  local expected_kind
  local expected_config
  local expected_config_sha
  local expected_git_commit
  if [[ "${candidate}" == "P0" ]]; then
    expected_kind="s4_r7_token_preserving"
    expected_config="${P0_CONFIG}"
    expected_config_sha="${P0_CONFIG_SHA256}"
    expected_git_commit="$(git -C "${P0_WORKTREE}" rev-parse HEAD)"
  else
    expected_kind="s4_r7_world_utility_coupling"
    expected_config="${P1_CONFIG}"
    expected_config_sha="${P1_CONFIG_SHA256}"
    expected_git_commit="$(git -C "${P1_WORKTREE}" rev-parse HEAD)"
  fi
  ( cd "${BASE_REPO}" && uv run --frozen python - "${path}" "${candidate}" \
      "${expected_kind}" "${mode}" "${TOTAL_UPDATES}" "${expected_config}" \
      "${expected_config_sha}" "${expected_git_commit}" <<'PY'
from collections.abc import Mapping
from pathlib import Path
import sys
import torch

from scripts.train_static_rgb_act_moe import _load_yaml

path_raw, candidate, expected_kind, mode, total_raw, config_raw, config_sha, expected_git_commit = sys.argv[1:]
path = Path(path_raw).resolve(strict=True)
value = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(value, Mapping):
    raise ValueError("checkpoint/resume must contain a mapping")
checkpoint_format = "wam.robofactory.s4_r7.world_utility.checkpoint/1"
resume_format = "wam.robofactory.s4_r7.world_utility.resume/1"
if mode == "complete":
    if value.get("format_version") != checkpoint_format:
        raise ValueError("formal checkpoint has an unsupported format")
    identity = value.get("method")
else:
    if value.get("format_version") != resume_format:
        raise ValueError("resume has an unsupported S4-R7 format")
    identity = value.get("identity")
if not isinstance(identity, Mapping):
    raise ValueError("checkpoint/resume identity must be a mapping")
if (
    identity.get("round_id") != "s4-r7"
    or identity.get("candidate_id") != candidate
    or identity.get("model_kind") != expected_kind
):
    raise ValueError("checkpoint/resume identity differs from candidate registry")
config = _load_yaml(Path(config_raw).resolve(strict=True))
parent_config = config.get("parent", {})
expected_parent = {
    "legacy_r6l_policy_sha256": parent_config.get("expected_legacy_r6l_policy_sha256"),
    "active_flow_checkpoint_sha256": parent_config.get("expected_active_flow_sha256"),
    "local_future_checkpoint_sha256": parent_config.get("expected_local_future_sha256"),
    "team_future_checkpoint_sha256": parent_config.get("expected_team_future_sha256"),
    "pca_artifact_sha256": parent_config.get("expected_pca_sha256"),
}
parent = value.get("parent_identity", identity.get("parent_identity"))
if not isinstance(parent, Mapping):
    raise ValueError("checkpoint/resume parent identity is missing")
if any(parent.get(key) != expected for key, expected in expected_parent.items()):
    raise ValueError("checkpoint/resume ancestor hashes differ from the registered config")
source = value.get("source", {})
if mode == "complete":
    if not isinstance(source, Mapping) or source.get("config_sha256") != config_sha:
        raise ValueError("formal checkpoint lacks the exact source config hash")
    if source.get("git_commit") != expected_git_commit:
        raise ValueError("formal checkpoint source commit differs from the candidate branch")
elif identity.get("config_sha256") != config_sha:
    raise ValueError("resume lacks the exact identity config hash")
total = int(total_raw)
current_update = value.get("update")
if isinstance(current_update, bool) or not isinstance(current_update, int):
    raise ValueError("checkpoint/resume has no exact top-level optimizer update")
if mode == "complete" and current_update != total:
    raise ValueError(f"formal checkpoint is not complete at update {total}")
if mode == "resume" and not 0 <= current_update < total:
    raise ValueError("resume update must be in [0, total_updates)")
PY
  )
}

verify_candidate_report() {
  local report="$1"
  local checkpoint="$2"
  local candidate="$3"
  ( cd "${BASE_REPO}" && uv run --frozen python - "${report}" \
      "${checkpoint}" "${candidate}" <<'PY'
from pathlib import Path
import sys

from scripts.accept_s4_r7 import _evaluate_candidate, _read_checkpoint, _read_json

report_path, checkpoint_path, candidate = sys.argv[1:]
report = _read_json(Path(report_path))
checkpoint = _read_checkpoint(Path(checkpoint_path))
_evaluate_candidate(candidate, report, checkpoint)
PY
  )
}

verify_acceptance() {
  ( cd "${BASE_REPO}" && uv run --frozen python - "${ACCEPTANCE}" \
      "${PAIR_EXACT}" "${P0_REPORT}" "${P1_REPORT}" "${P0_CHECKPOINT}" \
      "${P1_CHECKPOINT}" <<'PY'
from pathlib import Path
import sys

from scripts.accept_s4_r7 import build_acceptance, _read_checkpoint, _read_json

acceptance_path, pair_path, p0_report_path, p1_report_path, p0_ckpt_path, p1_ckpt_path = map(Path, sys.argv[1:])
observed = dict(_read_json(acceptance_path))
expected = build_acceptance(
    _read_json(pair_path),
    _read_json(p0_report_path),
    _read_json(p1_report_path),
    _read_checkpoint(p0_ckpt_path),
    _read_checkpoint(p1_ckpt_path),
)
if observed.get("format_version") != "wam.robofactory.s4_r7.acceptance/1" or observed.get("round_id") != "s4-r7":
    raise ValueError("acceptance format/round identity is invalid")
observed.pop("created_at", None)
expected.pop("created_at", None)
if observed != expected:
    raise ValueError("existing acceptance does not match current pair/reports/checkpoint hashes")
PY
  )
}

wait_for_peer_file() {
  local path="$1"
  local peer="$2"
  local phase="$3"
  local detail="$4"
  while [[ ! -f "${path}" ]]; do
    if [[ -f "${FAILED_FILE}" ]]; then
      abort "shared preparation failure marker appeared while ${detail}"
    fi
    if peer_failed "${peer}"; then
      abort "${peer} entered failed state before producing ${path}"
    fi
    if ! set_stage "${phase}" "${RUNNER_NAME}" "${detail}"; then
      abort "could not publish wait heartbeat/status"
    fi
    sleep "${HEARTBEAT_SECONDS}" || abort "wait for ${path} was interrupted"
  done
}

if [[ -f "${FAILED_FILE}" ]]; then
  abort "shared preparation already failed: ${FAILED_FILE}"
fi
while [[ ! -f "${READY_FILE}" ]]; do
  if [[ -f "${FAILED_FILE}" ]]; then
    abort "shared preparation failed before ready: ${FAILED_FILE}"
  fi
  if ! set_stage waiting_shared "${RUNNER_NAME}" \
    "waiting for one-copy datasets and exact accepted ancestor hashes"; then
    abort "could not publish shared-wait status"
  fi
  sleep "${HEARTBEAT_SECONDS}" || abort "shared-ready wait was interrupted"
done
if [[ -f "${FAILED_FILE}" ]]; then
  abort "shared preparation has both ready and failed markers"
fi
if ! grep -Fq 's4-r7 shared inputs verified' "${READY_FILE}"; then
  abort "shared.ready content is not an S4-R7 verification marker"
fi
SHARED_HASHES="${RUN_ROOT}/shared_artifact_sha256.txt"
if [[ ! -f "${SHARED_HASHES}" ]]; then
  abort "shared.ready exists without ${SHARED_HASHES}"
fi
python3 - "${SHARED_HASHES}" "${BASE_REPO}" <<'PY'
from pathlib import Path
import re
import sys

manifest_path, base_raw = map(Path, sys.argv[1:])
base = base_raw.resolve(strict=True)
expected = {
    (base / "artifacts/s4_r7_parent/legacy_r6l_policy.pt").resolve(strict=True),
    (base / "artifacts/s4_r7_parent/active_flow.pt").resolve(strict=True),
    (base / "artifacts/s4_r7_parent/local_future.pt").resolve(strict=True),
    (base / "artifacts/s4_r7_parent/team_future.pt").resolve(strict=True),
    (base / "artifacts/s2_r4/dino_pca_statistics.pt").resolve(strict=True),
}
observed = set()
for line in manifest_path.resolve(strict=True).read_text().splitlines():
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
        raise ValueError("shared artifact hash manifest contains an invalid row")
    observed.add(Path(parts[1]).resolve(strict=True))
if observed != expected:
    raise ValueError("shared artifact hash manifest paths differ from the five registered ancestors")
PY
shared_manifest_code=$?
if (( shared_manifest_code != 0 )); then
  abort "shared artifact hash manifest identity validation failed"
fi
if ! sha256sum --check --strict "${SHARED_HASHES}"; then
  abort "one or more shared ancestor hashes changed after shared.ready"
fi

link_shared_read_only() {
  local target="$1"
  local source="$2"
  local resolved_source resolved_target
  if [[ ! -d "${source}" ]]; then
    printf >&2 'Missing shared directory: %s\n' "${source}"
    return 3
  fi
  resolved_source="$(realpath -e -- "${source}")" || return $?
  if [[ -L "${target}" ]]; then
    resolved_target="$(realpath -e -- "${target}")" || return $?
    if [[ "${resolved_target}" != "${resolved_source}" ]]; then
      printf >&2 'Shared link points at a different tree: %s\n' "${target}"
      return 3
    fi
    return 0
  fi
  if [[ -e "${target}" ]]; then
    printf >&2 'Refusing non-symlink candidate-local shared path: %s\n' "${target}"
    return 3
  fi
  ln -s "${resolved_source}" "${target}" || return $?
  [[ -L "${target}" ]] || return 3
}

if ! set_stage setup "${RUNNER_NAME}" \
  "linking shared datasets/artifacts read-only by path; candidate outputs stay isolated"; then
  abort "could not publish setup status"
fi
link_shared_read_only "${FE_ROOT}/datasets" "${BASE_REPO}/datasets" || \
  abort "could not establish exact shared datasets symlink"
link_shared_read_only "${FE_ROOT}/artifacts" "${BASE_REPO}/artifacts" || \
  abort "could not establish exact shared artifacts symlink"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export GPU_INDEX
if [[ -z "${UV_CACHE_DIR:-}" ]]; then UV_CACHE_DIR="${BASE_REPO}/.uv-cache"; fi
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then UV_PROJECT_ENVIRONMENT="${BASE_REPO}/.venv"; fi
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT
ROBOFACTORY_ROOT="${S4_R7_ROBOFACTORY_ROOT:-$(dirname "${BASE_REPO}")/RoboFactory}"
RF_PYTHON="${S4_R7_RF_PYTHON:-${ROBOFACTORY_ROOT}/.venv/bin/python}"
S4_R7_ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}"
S4_R7_RF_PYTHON="${RF_PYTHON}"
LPD_POLICY_KIND=s4_flow
LPD_EXPERIMENT_SLUG="s4_r7_${SLUG}"
LPD_CONFIG="${CONFIG}"
LPD_CHECKPOINT="${CHECKPOINT}"
LPD_PORT="$((8872 + GPU_INDEX))"
LPD_RUN_ID="${RUN_ID}_${SLUG}"
LPD_EPISODES=20
LPD_SEED_START=900
LPD_OUTPUT_ROOT="${VALIDATION_ROOT}/gate_${RUN_ID}"
export ROBOFACTORY_ROOT RF_PYTHON S4_R7_ROBOFACTORY_ROOT S4_R7_RF_PYTHON
export LPD_POLICY_KIND LPD_EXPERIMENT_SLUG LPD_CONFIG LPD_CHECKPOINT LPD_PORT
export LPD_RUN_ID LPD_EPISODES LPD_SEED_START LPD_OUTPUT_ROOT
export LPD_STAGE_LOG="${TRAIN_ROOT}/stages.jsonl"
unset HF_TOKEN

if [[ -f "${PREFLIGHT_REPORT}" ]]; then
  if ! verify_preflight "${PREFLIGHT_REPORT}" "${CANDIDATE}" terminal; then
    abort "existing preflight report failed format/identity/measurement validation"
  fi
  if ! preflight_provenance "${PREFLIGHT_REPORT}" "${CANDIDATE}" verify || \
     ! ensure_digest "${PREFLIGHT_REPORT}"; then
    abort "existing preflight report hash/config/code provenance validation failed"
  fi
  if [[ "$(jq -r '.completed' "${PREFLIGHT_REPORT}")" == "true" ]]; then
    CURRENT_PREFLIGHT="local-pass"
  else
    CURRENT_PREFLIGHT="local-oom-await-paired-fallback"
  fi
else
  if [[ ! -f "${TRAINER}" ]]; then
    abort "missing candidate trainer: ${TRAINER}"
  fi
  CURRENT_PREFLIGHT="running-200"
  run_stage preflight train_s4_r7_world_utility.py \
    "200-step paired preflight; micro=${MICRO_TEAM_BATCH} accum=${GRADIENT_ACCUMULATION}; measuring exact resume/hash/memory" \
    uv run --frozen python scripts/train_s4_r7_world_utility.py \
      --config "${CONFIG}" --device cuda:0 --output "${PREFLIGHT_CHECKPOINT}" \
      --resume "${PREFLIGHT_RESUME}" --progress-log "${PREFLIGHT_PROGRESS}" \
      --preflight-only --preflight-updates "${PREFLIGHT_UPDATES}" \
      --preflight-report "${PREFLIGHT_REPORT}"
  preflight_code=$?
  if [[ ! -f "${PREFLIGHT_REPORT}" ]] || \
     ! verify_preflight "${PREFLIGHT_REPORT}" "${CANDIDATE}" terminal || \
     ! ensure_digest "${PREFLIGHT_REPORT}" || \
     ! preflight_provenance "${PREFLIGHT_REPORT}" "${CANDIDATE}" create; then
    if (( preflight_code != 0 )); then
      FAILURE_DETAIL="train_s4_r7_world_utility.py preflight exited ${preflight_code} without a valid terminal report"
      exit "${preflight_code}"
    fi
    abort "trainer returned without a valid hashed/config-bound terminal preflight report"
  fi
  if [[ "$(jq -r '.completed' "${PREFLIGHT_REPORT}")" == "true" ]]; then
    CURRENT_PREFLIGHT="local-pass"
  else
    CURRENT_PREFLIGHT="local-oom-await-paired-fallback"
  fi
fi

if [[ "${CANDIDATE}" == "P0" ]]; then PEER="P1"; else PEER="P0"; fi
PEER_PREFLIGHT="${P1_PREFLIGHT}"
if [[ "${PEER}" == "P0" ]]; then PEER_PREFLIGHT="${P0_PREFLIGHT}"; fi
wait_for_peer_file "${PEER_PREFLIGHT}" "${PEER}" waiting_peer_preflight \
  "local 200-step preflight complete; waiting for ${PEER} preflight without occupying its GPU"
PEER_PREFLIGHT_PROVENANCE="${PEER_PREFLIGHT}.provenance.json"
wait_for_peer_file "${PEER_PREFLIGHT_PROVENANCE}" "${PEER}" \
  waiting_peer_preflight \
  "${PEER} report exists; waiting for its atomic config/code/hash provenance"
if ! verify_preflight "${P0_PREFLIGHT}" P0 terminal || \
   ! verify_preflight "${P1_PREFLIGHT}" P1 terminal || \
   ! preflight_provenance "${P0_PREFLIGHT}" P0 verify || \
   ! preflight_provenance "${P1_PREFLIGHT}" P1 verify || \
   ! ensure_digest "${P0_PREFLIGHT}" || \
   ! ensure_digest "${P1_PREFLIGHT}"; then
  abort "one of the paired preflight reports failed current identity/hash validation"
fi

exec {PAIR_FD}>"${RUN_ROOT}/.pair_exact.lock"
if ! set_stage waiting_pair_lock flock \
  "waiting for exclusive paired-preflight validator lock"; then
  abort "could not publish pair-lock wait status"
fi
if ! flock -x "${PAIR_FD}"; then abort "could not acquire pair-exact lock"; fi
if [[ ! -f "${PAIR_EXACT}" ]]; then
  run_stage pair_validation validate_s4_r7_branch_pair.py \
    "both 200-step preflights complete; checking exact indices/optimizer/LR/resume/memory" \
    uv run --frozen python "${PAIR_VALIDATOR}" \
      --p0-config "${P0_CONFIG}" --p1-config "${P1_CONFIG}" \
      --p0-preflight "${P0_PREFLIGHT}" --p1-preflight "${P1_PREFLIGHT}" \
      --output "${PAIR_EXACT}"
  pair_code=$?
else
  pair_code=0
fi
PAIR_FALLBACK=""
if [[ -f "${PAIR_EXACT}" ]]; then
  PAIR_FALLBACK="$(jq -r '.preflight.required_fallback // empty' "${PAIR_EXACT}" 2>/dev/null)"
fi
if [[ "${PAIR_FALLBACK}" == "micro1_accum12" ]]; then
  flock -u "${PAIR_FD}" || true
  CURRENT_PROGRAM="validate_s4_r7_branch_pair.py"
  CURRENT_PREFLIGHT="FAIL-required-fallback-micro1_accum12"
  abort "paired preflight requires micro1_accum12; no one-sided auto-change is allowed—update both candidate configs/branches together and start a new run"
fi
if (( pair_code != 0 )); then
  flock -u "${PAIR_FD}" || true
  CURRENT_PROGRAM="validate_s4_r7_branch_pair.py"
  CURRENT_PREFLIGHT="FAIL-pair-exact"
  FAILURE_DETAIL="paired preflight exactness validator exited ${pair_code}"
  exit "${pair_code}"
fi
if ! verify_pair_exact || ! ensure_digest "${PAIR_EXACT}"; then
  flock -u "${PAIR_FD}" || true
  abort "pair_exact failed format/identity/hash validation"
fi
if ! flock -u "${PAIR_FD}"; then abort "could not release pair-exact lock"; fi
if ! verify_preflight "${P0_PREFLIGHT}" P0 pass || \
   ! verify_preflight "${P1_PREFLIGHT}" P1 pass; then
  abort "pair_exact claimed pass but a preflight is not a complete valid 200-step run"
fi
CURRENT_PREFLIGHT="PASS-pair-exact"

if [[ -f "${CHECKPOINT}" ]]; then
  if ! verify_checkpoint "${CHECKPOINT}" "${CANDIDATE}" complete || \
     ! ensure_digest "${CHECKPOINT}"; then
    abort "existing formal checkpoint failed completion/identity/ancestor/config hash validation"
  fi
else
  if [[ -f "${RESUME}" ]]; then
    if ! verify_checkpoint "${RESUME}" "${CANDIDATE}" resume; then
      abort "existing formal resume failed update/identity/ancestor/config hash validation"
    fi
    RESUME_INPUT_SHA256="$(sha256sum "${RESUME}" | awk '{print $1}')" || \
      abort "could not hash validated resume input"
    printf 'Validated resume input SHA256: %s\n' "${RESUME_INPUT_SHA256}"
  fi
  if [[ ! -f "${TRAINER}" ]]; then
    abort "missing candidate trainer: ${TRAINER}"
  fi
  run_stage training train_s4_r7_world_utility.py \
    "pair-exact passed; formal 125000 updates with validated candidate-isolated resume" \
    uv run --frozen python scripts/train_s4_r7_world_utility.py \
      --config "${CONFIG}" --device cuda:0 --updates "${TOTAL_UPDATES}" \
      --output "${CHECKPOINT}" --resume "${RESUME}" \
      --progress-log "${TRAIN_PROGRESS}"
  training_code=$?
  if (( training_code != 0 )); then
    FAILURE_DETAIL="train_s4_r7_world_utility.py formal training exited ${training_code}"
    exit "${training_code}"
  fi
  if [[ ! -f "${CHECKPOINT}" ]] || \
     ! verify_checkpoint "${CHECKPOINT}" "${CANDIDATE}" complete || \
     ! ensure_digest "${CHECKPOINT}"; then
    abort "trainer returned success without a complete identity/hash-valid checkpoint"
  fi
fi

if [[ -f "${CANDIDATE_REPORT}" ]]; then
  if ! verify_candidate_report "${CANDIDATE_REPORT}" "${CHECKPOINT}" \
      "${CANDIDATE}" || ! ensure_digest "${CANDIDATE_REPORT}"; then
    abort "existing causal candidate report failed format/identity/checkpoint-hash validation"
  fi
else
  if [[ ! -f "${EVALUATOR}" ]]; then
    abort "missing candidate evaluator: ${EVALUATOR}"
  fi
  if [[ ! -x "${RF_PYTHON}" ]]; then
    abort "RoboFactory Python required for causal Gate20 is unavailable: ${RF_PYTHON}"
  fi
  run_stage validating evaluate_s4_r7_causal.py \
    "structural audits plus paired 8-condition five-task Gate20 and utility calibration" \
    uv run --frozen python scripts/evaluate_s4_r7_causal.py \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
      --output "${CANDIDATE_REPORT}" --progress-log "${VALIDATION_PROGRESS}" \
      --device cuda:0
  evaluation_code=$?
  if (( evaluation_code != 0 )); then
    FAILURE_DETAIL="evaluate_s4_r7_causal.py exited ${evaluation_code}"
    exit "${evaluation_code}"
  fi
  if [[ ! -f "${CANDIDATE_REPORT}" ]] || \
     ! verify_candidate_report "${CANDIDATE_REPORT}" "${CHECKPOINT}" \
       "${CANDIDATE}" || ! ensure_digest "${CANDIDATE_REPORT}"; then
    abort "evaluator returned success without a valid checkpoint-bound candidate report"
  fi
fi

PEER_REPORT="${P1_REPORT}"
if [[ "${PEER}" == "P0" ]]; then PEER_REPORT="${P0_REPORT}"; fi
wait_for_peer_file "${PEER_REPORT}" "${PEER}" waiting_peer_report \
  "local causal validation complete; waiting for ${PEER} report before special acceptance"
if ! verify_pair_exact || \
   ! verify_checkpoint "${P0_CHECKPOINT}" P0 complete || \
   ! verify_checkpoint "${P1_CHECKPOINT}" P1 complete || \
   ! ensure_digest "${P0_CHECKPOINT}" || \
   ! ensure_digest "${P1_CHECKPOINT}" || \
   ! verify_candidate_report "${P0_REPORT}" "${P0_CHECKPOINT}" P0 || \
   ! verify_candidate_report "${P1_REPORT}" "${P1_CHECKPOINT}" P1 || \
   ! ensure_digest "${P0_REPORT}" || ! ensure_digest "${P1_REPORT}"; then
  abort "paired formal checkpoint/report inputs failed identity/hash validation"
fi

exec {ACCEPT_FD}>"${RUN_ROOT}/.acceptance.lock"
if ! set_stage waiting_acceptance_lock flock \
  "waiting for exclusive special-acceptance lock"; then
  abort "could not publish acceptance-lock wait status"
fi
if ! flock -x "${ACCEPT_FD}"; then abort "could not acquire acceptance lock"; fi
if [[ ! -f "${ACCEPTANCE}" ]]; then
  run_stage accepting accept_s4_r7.py \
    "applying R7 structural, causal, utility and paired Gate20 winner rules" \
    uv run --frozen python "${ACCEPT_TOOL}" --pair-exact "${PAIR_EXACT}" \
      --p0-report "${P0_REPORT}" --p1-report "${P1_REPORT}" \
      --p0-checkpoint "${P0_CHECKPOINT}" --p1-checkpoint "${P1_CHECKPOINT}" \
      --output "${ACCEPTANCE}"
  acceptance_code=$?
  if (( acceptance_code != 0 )); then
    flock -u "${ACCEPT_FD}" || true
    FAILURE_DETAIL="accept_s4_r7.py exited ${acceptance_code}"
    exit "${acceptance_code}"
  fi
fi
if ! verify_acceptance || ! ensure_digest "${ACCEPTANCE}"; then
  flock -u "${ACCEPT_FD}" || true
  abort "acceptance failed recomputation/identity/hash validation"
fi
if ! flock -u "${ACCEPT_FD}"; then abort "could not release acceptance lock"; fi

DECISION="$(jq -er '.decision | strings' "${ACCEPTANCE}")" || \
  abort "acceptance has no decision"
WINNER="$(jq -er '.winner | strings' "${ACCEPTANCE}")" || \
  abort "acceptance has no winner"
CURRENT_PREFLIGHT="PASS-pair-exact"
if ! set_stage complete "${RUNNER_NAME}" \
  "training and causal validation complete; decision=${DECISION}; winner=${WINNER}"; then
  abort "could not publish terminal completion status"
fi
if ! publish_status 0; then abort "could not publish terminal exit code"; fi
COMPLETED=1
printf 'S4-R7 %s complete: decision=%s winner=%s root=%s\n' \
  "${CANDIDATE}" "${DECISION}" "${WINNER}" "${CANDIDATE_ROOT}"
