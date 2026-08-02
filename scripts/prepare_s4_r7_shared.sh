#!/usr/bin/env bash

# S4-R7 shared preparation.  This script deliberately has no `set -e` mode:
# every fallible operation reports its own failure and preserves partial data.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in S4_R7_RUN_ROOT S4_R7_READY_FILE S4_R7_FAILED_FILE; do
  if [[ -z "${!name:-}" ]]; then
    printf >&2 'Missing %s\n' "${name}"
    exit 2
  fi
done

STATUS_TOOL="${FE_ROOT}/scripts/s4_r7_runtime.py"
HEARTBEAT_PID=""
COMPLETED=0
LEGACY_SHA="5f3a05628563a0b2e26ea62941cda6ae49a6f161739d26abb351cdc483a18fc9"
FLOW_SHA="f0075e9671d887c5546a9bd9eadaa3db082fd7eef1aad8c72aab61461dca8d53"
LOCAL_SHA="c04f8ea12c5b6d8f7c04992d7dd4a8c0a33aa7d0058987679e6553b17e410a2f"
TEAM_SHA="fcc0af76c2acd6805750f12e828a1249eb91e466e51f4aa77c118b6e9d330c67"
PCA_SHA="a0d236540b2fbe58b2771573f0d5674ac39ff4a6a65b16e2b39691de186483b9"

status() {
  if [[ -f "${STATUS_TOOL}" ]]; then
    python3 "${STATUS_TOOL}" shared-status --run-root "${S4_R7_RUN_ROOT}" \
      --phase "$1" --program "$2" --detail "${3:-}" || true
  fi
}

heartbeat_loop() {
  while true; do
    if [[ -f "${STATUS_TOOL}" ]]; then
      python3 "${STATUS_TOOL}" heartbeat --run-root "${S4_R7_RUN_ROOT}" \
        --shared || true
    fi
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
    touch "${S4_R7_FAILED_FILE}" 2>/dev/null || true
    status failed prepare_s4_r7_shared.sh \
      "preparation exited ${code}; inspect ${S4_R7_RUN_ROOT}/prepare.log"
  fi
}
trap on_exit EXIT

mkdir -p "${S4_R7_RUN_ROOT}" || {
  printf >&2 'Cannot create run root: %s\n' "${S4_R7_RUN_ROOT}"
  exit 3
}
exec > >(tee -a "${S4_R7_RUN_ROOT}/prepare.log") 2>&1
heartbeat_loop & HEARTBEAT_PID=$!
status verifying prepare_s4_r7_shared.sh \
  "verifying one-copy data and exact R6L-P1/R5-P0 ancestors"

if [[ "${S4_R7_USE_S0_PREP:-0}" == "1" ]]; then
  for name in S4_R7_HF_TOKEN_FIFO S4_R7_ROBOFACTORY_ROOT S4_R7_RF_PYTHON \
    UV_CACHE_DIR UV_PROJECT_ENVIRONMENT; do
    if [[ -z "${!name:-}" ]]; then
      printf >&2 'S0 preparation requires %s\n' "${name}"
      exit 3
    fi
  done
  status s0_prepare prepare_s3_r6_from_s0.sh \
    "S0 mode-0600 FIFO; pinned hf download; datasets use Xet, DINO/RoboFactory do not"
  S3_R6_HF_TOKEN_FIFO="${S4_R7_HF_TOKEN_FIFO}" \
  S3_R6_ROBOFACTORY_ROOT="${S4_R7_ROBOFACTORY_ROOT}" \
  S3_R6_RF_PYTHON="${S4_R7_RF_PYTHON}" \
  S3_R6_P0_CONFIG="${FE_ROOT}/configs/wam_flow/s2_r5_protected_team.yaml" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
  ROBOFACTORY_ROOT="${S4_R7_ROBOFACTORY_ROOT}" \
  RF_PYTHON="${S4_R7_RF_PYTHON}" \
    bash "${FE_ROOT}/scripts/prepare_s3_r6_from_s0.sh"
  prep_code=$?
  if (( prep_code != 0 )); then
    printf >&2 'S0-compatible preparation failed with code %d\n' "${prep_code}"
    exit "${prep_code}"
  fi
fi

verify_file() {
  label="$1"
  path="$2"
  expected="$3"
  if [[ ! -f "${path}" ]]; then
    printf >&2 'Missing %s: %s\n' "${label}" "${path}"
    return 3
  fi
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    printf >&2 '%s hash mismatch: expected %s, got %s (%s)\n' \
      "${label}" "${expected}" "${observed}" "${path}"
    return 3
  fi
}

find_exact() {
  expected="$1"
  shift
  for path in "$@"; do
    if [[ -f "${path}" ]] && \
       [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]; then
      printf '%s' "${path}"
      return 0
    fi
  done
  return 1
}

LEGACY_SOURCE="${S4_R7_LEGACY_POLICY_SOURCE:-}"
FLOW_SOURCE="${S4_R7_ACTIVE_FLOW_SOURCE:-}"
LOCAL_SOURCE="${S4_R7_LOCAL_FUTURE_SOURCE:-}"
TEAM_SOURCE="${S4_R7_TEAM_FUTURE_SOURCE:-}"

if [[ -z "${LEGACY_SOURCE}" ]]; then
  LEGACY_SOURCE="$(find_exact "${LEGACY_SHA}" \
    "${FE_ROOT}/outputs/s3_r6_runs/s3-r6-five-task-retrain-round1/candidates/r6l_p1/checkpoints/policy.pt" \
    "${FE_ROOT}/artifacts/s4_r7_parent/legacy_r6l_policy.pt")"
fi
if [[ -z "${FLOW_SOURCE}" ]]; then
  FLOW_SOURCE="$(find_exact "${FLOW_SHA}" \
    "${FE_ROOT}/outputs/s3_r6_runs/s3-r6-five-task-retrain-round1/candidates/r6l_p1/checkpoints/five_task_flow.pt" \
    "${FE_ROOT}/artifacts/s4_r7_parent/active_flow.pt")"
fi
if [[ -z "${LOCAL_SOURCE}" ]]; then
  LOCAL_SOURCE="$(find_exact "${LOCAL_SHA}" \
    "${FE_ROOT}/outputs/s2_r4_runs/s2-r4-round1-resume2/candidates/p0/checkpoints/predictor.pt" \
    "${FE_ROOT}/artifacts/s2_r5_protected_p0/predictor.pt" \
    "${FE_ROOT}/artifacts/s4_r7_parent/local_future.pt")"
fi
if [[ -z "${TEAM_SOURCE}" ]]; then
  TEAM_SOURCE="$(find_exact "${TEAM_SHA}" \
    "${FE_ROOT}/outputs/s2_r5_runs/s2-r5-round1/candidates/p0/checkpoints/predictor.pt" \
    "${FE_ROOT}/artifacts/s4_r7_parent/team_future.pt")"
fi

verify_file "accepted R6L-P1 policy" "${LEGACY_SOURCE}" "${LEGACY_SHA}" || exit $?
verify_file "accepted R6L-P1 five-task Flow" "${FLOW_SOURCE}" "${FLOW_SHA}" || exit $?
verify_file "accepted R4-P0 local future" "${LOCAL_SOURCE}" "${LOCAL_SHA}" || exit $?
verify_file "accepted R5-P0 team future" "${TEAM_SOURCE}" "${TEAM_SHA}" || exit $?
verify_file "S2 PCA/statistics artifact" \
  "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt" "${PCA_SHA}" || exit $?

required=(
  "${FE_ROOT}/datasets/robofactory_multitask/lift_barrier/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/long_pipeline_delivery/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/take_photo/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/three_robots_stack_cube/training_manifest.json"
  "${FE_ROOT}/datasets/robofactory_multitask/camera_alignment/training_manifest.json"
  "${FE_ROOT}/artifacts/vision/dinov3_vitl16_lvd/model.safetensors"
  "${S4_R7_ROBOFACTORY_ROOT:-}/robofactory/assets/scenes/table/table.glb"
  "${S4_R7_RF_PYTHON:-}"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    printf >&2 'Missing S4-R7 shared input: %s\n' "${path}"
    printf >&2 'Use --prepare-from-s0 only for HF/data assets; accepted parents are never retrained here.\n'
    exit 3
  fi
done

status parent_identity prepare_s4_r7_shared.sh \
  "validating policy kind and exact links; R6J is forbidden"
( cd "${FE_ROOT}" && uv run --frozen python - "${LEGACY_SOURCE}" "${FLOW_SOURCE}" \
    "${LOCAL_SOURCE}" "${TEAM_SOURCE}" <<'PY'
from collections.abc import Mapping
from pathlib import Path
import sys
import torch

legacy, flow, local, team = [Path(value).resolve(strict=True) for value in sys.argv[1:]]
p = torch.load(legacy, map_location="cpu", weights_only=False)
f = torch.load(flow, map_location="cpu", weights_only=False)
l = torch.load(local, map_location="cpu", weights_only=False)
t = torch.load(team, map_location="cpu", weights_only=False)
if not all(isinstance(value, Mapping) for value in (p, f, l, t)):
    raise ValueError("all S4 parent artifacts must contain mappings")
m = p.get("method", {})
if p.get("format_version") != "wam.robofactory.s3_r6.world_action_flow.checkpoint/1":
    raise ValueError("legacy policy is not an S3-R6 checkpoint")
if (m.get("micro_round"), m.get("candidate_id"), m.get("model_kind")) != (
    "R6L", "P1", "s3_r6l_protected_local_gated"
):
    raise ValueError("S4 requires accepted R6L-P1; R6J and R6L-P0 are forbidden")
identity = p.get("parent_identity", {})
if Path(str(identity.get("flow_checkpoint", ""))).name != flow.name:
    raise ValueError("legacy policy does not identify the selected five-task Flow")
if identity.get("flow_checkpoint_sha256") != "f0075e9671d887c5546a9bd9eadaa3db082fd7eef1aad8c72aab61461dca8d53":
    raise ValueError("legacy policy Flow hash differs from the accepted artifact")
lm = l.get("method", {})
tm = t.get("method", {})
if (lm.get("candidate_id"), lm.get("model_kind")) != (
    "P0", "s2_r4_local_action_conditioned"
):
    raise ValueError("local future parent is not accepted R4-P0")
if (tm.get("candidate_id"), tm.get("model_kind"), tm.get("team_mixer")) != (
    "P0", "s2_r5_protected_shared_team", "shared"
):
    raise ValueError("team future parent is not accepted R5-P0")
if f.get("format_version") != "wam.robofactory.agent_factorized_flow.checkpoint/1":
    # FLOW_FORMAT is versioned in code; retain a narrow compatibility message.
    raise ValueError(f"unexpected accepted Flow format: {f.get('format_version')!r}")
PY
)
identity_code=$?
if (( identity_code != 0 )); then
  printf >&2 'S4-R7 parent identity validation failed with code %d\n' "${identity_code}"
  exit "${identity_code}"
fi

parent_dir="${FE_ROOT}/artifacts/s4_r7_parent"
mkdir -p "${parent_dir}" || exit $?
link_parent() {
  source_path="$(realpath "$1")"
  target_path="$2"
  if [[ -L "${target_path}" && ! -e "${target_path}" ]]; then
    unlink "${target_path}" || return $?
  fi
  if [[ ! -e "${target_path}" ]]; then
    ln -s "${source_path}" "${target_path}" || return $?
  elif [[ "$(realpath "${target_path}")" != "${source_path}" ]]; then
    printf >&2 'S4 parent link already points elsewhere: %s\n' "${target_path}"
    return 3
  fi
}
link_parent "${LEGACY_SOURCE}" "${parent_dir}/legacy_r6l_policy.pt" || exit $?
link_parent "${FLOW_SOURCE}" "${parent_dir}/active_flow.pt" || exit $?
link_parent "${LOCAL_SOURCE}" "${parent_dir}/local_future.pt" || exit $?
link_parent "${TEAM_SOURCE}" "${parent_dir}/team_future.pt" || exit $?

manifest_tmp="${S4_R7_RUN_ROOT}/.shared_artifact_sha256.tmp"
sha256sum "${parent_dir}/legacy_r6l_policy.pt" \
  "${parent_dir}/active_flow.pt" \
  "${parent_dir}/local_future.pt" \
  "${parent_dir}/team_future.pt" \
  "${FE_ROOT}/artifacts/s2_r4/dino_pca_statistics.pt" > "${manifest_tmp}" || exit $?
mv "${manifest_tmp}" "${S4_R7_RUN_ROOT}/shared_artifact_sha256.txt" || exit $?

ready_tmp="${S4_R7_READY_FILE}.tmp.$$"
printf '%s\n' "s4-r7 shared inputs verified at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${ready_tmp}" || exit $?
mv "${ready_tmp}" "${S4_R7_READY_FILE}" || exit $?
status complete prepare_s4_r7_shared.sh \
  "one-copy data plus exact R6L-P1/Flow/R4-P0/R5-P0/PCA hashes verified"
COMPLETED=1
printf 'S4-R7 shared preparation complete.\n'
