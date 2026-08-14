#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${SSC_V7_RUN_ROOT:-/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2}"
REPO_ROOT="${SSC_V7_REPO_ROOT:-/workspace/fe-pc-wam}"
ROBOFACTORY_ROOT="${SSC_V7_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
PRE_REG_ROOT="${RUN_ROOT}/pre_registration"
BASE_COMMIT="945d1b49247612f6e67d79104726b67915cf86bf"
W10_CHECKPOINT="/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt"
W10_SHA256="e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574"
DRY_RUN=0

fail() {
  printf >&2 'SSC-V7-M1 preflight failed: %s\n' "$*"
  exit 1
}

warn() {
  printf >&2 'SSC-V7-M1 warning: %s\n' "$*"
}

usage() {
  printf 'Usage: %s --dry-run\n' "$0"
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

((DRY_RUN == 1)) || fail "only --dry-run is authorized by Step 0; Measurement execution is not implemented here"

for command in git jq sha256sum nvidia-smi python3; do
  command -v "${command}" >/dev/null || fail "required command is missing: ${command}"
done

[[ -d "${REPO_ROOT}/.git" ]] || fail "repository is missing: ${REPO_ROOT}"
[[ -d "${ROBOFACTORY_ROOT}/.git" ]] || fail "RoboFactory repository is missing: ${ROBOFACTORY_ROOT}"
[[ -d "${PRE_REG_ROOT}" ]] || fail "pre-registration root is missing: ${PRE_REG_ROOT}"
[[ "$(git -C "${REPO_ROOT}" rev-parse feat/model-improvements)" == "${BASE_COMMIT}" ]] || fail "base branch moved"
[[ "$(git -C "${REPO_ROOT}" rev-parse origin/feat/model-improvements)" == "${BASE_COMMIT}" ]] || fail "origin base branch moved"
[[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] || fail "base repository worktree is dirty"

while IFS= read -r branch; do
  [[ "$(git -C "${REPO_ROOT}" rev-parse "${branch}" 2>/dev/null)" == "${BASE_COMMIT}" ]] || fail "branch is missing or not rooted at base: ${branch}"
done < <(jq -r '.branches[]' "${PRE_REG_ROOT}/contracts/stage_contract.json")

for json_file in "${PRE_REG_ROOT}"/contracts/*.json "${PRE_REG_ROOT}"/receipts/*.json; do
  jq empty "${json_file}" || fail "invalid JSON: ${json_file}"
done

[[ "$(jq -r '.stage_id' "${PRE_REG_ROOT}/contracts/measurement_gate.json")" == "SSC-V7-M1" ]] || fail "measurement gate is not the M1 revision"
dry_run_status="$(jq -r '.status' "${PRE_REG_ROOT}/receipts/dry_run_receipt.json")"
[[ "${dry_run_status}" == "NOT_RUN" || "${dry_run_status}" == "PASSED" ]] || fail "M1 dry-run receipt must be NOT_RUN or PASSED"

(
  cd "${RUN_ROOT}"
  sha256sum --check "${PRE_REG_ROOT}/sha256sums.txt"
) || fail "pre-registration artifact hash mismatch"

[[ -f "${W10_CHECKPOINT}" ]] || fail "W10 checkpoint is missing"
[[ "$(sha256sum "${W10_CHECKPOINT}" | cut -d ' ' -f1)" == "${W10_SHA256}" ]] || fail "W10 checkpoint hash mismatch"

declare -A MANIFEST_SHA256=(
  [lift_barrier]="a4180b2730c1ca5bbe8f28359f91bad9575a60197698c7ae650a54b14612aeb0"
  [camera_alignment]="909379b070286e7b1eda90623fef5cafbc712007fd89f071b5b579ae1415a420"
  [long_pipeline_delivery]="35ce94c4cbe121c2fbc1013d8a5366a2c4799c3ce78f46baf3e12fc00b2b7114"
  [take_photo]="d382dd6f99964770a600d8b13da9c3cf55a69a20a54a15b6e78d6b9f6108003f"
  [pass_shoe]="94e489a65917e51638ec6db322e6c18583d45ffb22014d2ecd21bd540260b6c9"
  [place_food]="a7f8d280cfcb363ed3d9cdc851136e8f39adc23770e62f828ad8b0abc24bc68e"
)
for task in "${!MANIFEST_SHA256[@]}"; do
  manifest="/workspace/datasets/robofactory_multitask/${task}/training_manifest.json"
  [[ -f "${manifest}" ]] || fail "training manifest is missing: ${task}"
  [[ "$(sha256sum "${manifest}" | cut -d ' ' -f1)" == "${MANIFEST_SHA256[$task]}" ]] || fail "training manifest hash mismatch: ${task}"
  [[ -f "/workspace/bwa_runs/w10-six-task-v1/seeds/validation/${task}.json" ]] || fail "W10 validation seeds are missing: ${task}"
done

python3 "${PRE_REG_ROOT}/bin/verify_ssc_v7_seed_contract.py" \
  --contract "${PRE_REG_ROOT}/contracts/seed_contract.json" \
  --w10-seed-root "/workspace/bwa_runs/w10-six-task-v1/seeds/validation" \
  || fail "seed derivation or disjointness check failed"

[[ "$(jq -r '.datasets.full_hdf5_verification.status' "${PRE_REG_ROOT}/receipts/source_receipt.json")" == "PASSED" ]] || fail "full HDF5 hash verification did not pass"
[[ "$(jq -r '.datasets.full_hdf5_verification.checked_files' "${PRE_REG_ROOT}/receipts/source_receipt.json")" == "900" ]] || fail "full HDF5 verification count is not 900"
[[ "$(git -C "${ROBOFACTORY_ROOT}" rev-parse HEAD)" == "5868242322414a91454e22f1dd9641f613ba1bcf" ]] || fail "RoboFactory commit mismatch"
[[ -z "$(git -C "${ROBOFACTORY_ROOT}" status --short)" ]] || fail "RoboFactory worktree is dirty"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${gpu_count}" == "4" ]] || fail "expected 4 GPUs, found ${gpu_count}"

for subroot in measurement b0 p-progress t-teammate b-belief pt ptb integration; do
  path="${RUN_ROOT}/${subroot}"
  [[ -d "${path}" ]] || fail "allocated run subroot is missing: ${path}"
  [[ -z "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "run subroot is not empty: ${path}"
done

if pgrep -af 'train_wam|train_w10|torchrun|evaluate_no_wrist_pair' >/dev/null; then
  fail "a training or W10 evaluation process is already running"
fi

if command -v vast-capabilities >/dev/null && [[ "$(vast-capabilities | jq -r '.instance.workspace_is_volume')" != "true" ]]; then
  warn "/workspace is not a persistent volume; sync every irreplaceable gate artifact off-box"
fi
[[ -f "${REPO_ROOT}/LICENSE" ]] || warn "project has no top-level LICENSE; internal research only, no external redistribution"

printf 'SSC_V7_M1_DRY_RUN_PASSED\n'
printf 'base_commit=%s\n' "${BASE_COMMIT}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'authorized_stage=measurement\n'
printf 'gpu_count=%s\n' "${gpu_count}"
