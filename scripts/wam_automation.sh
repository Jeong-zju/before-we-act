#!/usr/bin/env bash
#
# Ordered, resumable automation for FE-PC WAM and RoboFactory.
#
# This file intentionally uses only Bash and common system tools before the
# uv environments exist, so it can also be copied to a fresh remote server.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE=""
DRY_RUN=false
RESUME=false
STATE_FILE_OVERRIDE=""
SHOW_ACTIONS=false
declare -a REQUESTED_ACTIONS=()

usage() {
  cat <<'EOF'
Usage:
  wam_automation.sh [OPTIONS] ACTION [ACTION ...]

Options:
  --config FILE       Load a trusted Bash-style environment file.
  --dry-run           Print commands without changing files or starting jobs.
  --resume            Skip action positions already completed in --state-file.
  --state-file FILE   Override the resume state file.
  --list              List actions and exit.
  -h, --help          Show this help.

Actions are executed from left to right and may also be comma-separated:
  code                Clone/update FE-PC WAM and switch FE_REF.
  robofactory         Clone/update RoboFactory and switch ROBOFACTORY_REF.
  env                 Install uv/Python 3.11 and sync the WAM lockfile.
  robofactory-env     Create the isolated RoboFactory Python 3.9 environment.
  assets              Download RoboFactory simulation assets.
  hf-auth             Verify non-interactive Hugging Face authentication.
  hf-download         Download HF_DATASET_REPO into HF_DATASET_DIR.
  hf-upload           Upload HF_UPLOAD_DIR to HF_UPLOAD_REPO.
  vision              Download and verify the pinned DINOv3 artifact.
  doctor              Check disk, GPU, repositories, environments, and tools.
  data-check          Verify the downloaded M1 manifest and all data splits.
  test                Run the automation-relevant WAM test subset.
  train-smoke         Run a short end-to-end M1 training preflight.
  train               Run the formal LiftBarrier M1 scratch training.
  validate-smoke      Run a 3-episode cross-environment closed-loop smoke.
  validate            Run the formal 100-seed closed-loop benchmark.
  snapshot            Save Git, environment, dataset, and output provenance.
  bootstrap           code through environment/assets setup plus doctor.
  full-smoke          Fresh setup, data/model download, smoke train/validation.
  full                Fresh setup, formal training, and formal validation.

Examples:
  ./scripts/wam_automation.sh --config configs/automation.env \
    code robofactory env robofactory-env hf-download vision data-check train

  ./scripts/wam_automation.sh --config configs/automation.env full

Secrets are read from the environment. Set HF_TOKEN there; do not put it on the
command line or commit it to a configuration file.
EOF
}

list_actions() {
  printf '%s\n' \
    code robofactory env robofactory-env assets hf-auth hf-download hf-upload \
    vision doctor data-check test train-smoke train validate-smoke validate \
    snapshot bootstrap full-smoke full
}

while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || {
        printf 'ERROR: --config requires a file\n' >&2
        exit 2
      }
      CONFIG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --state-file)
      (($# >= 2)) || {
        printf 'ERROR: --state-file requires a file\n' >&2
        exit 2
      }
      STATE_FILE_OVERRIDE="$2"
      shift 2
      ;;
    --list)
      SHOW_ACTIONS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        REQUESTED_ACTIONS+=("$1")
        shift
      done
      ;;
    -*)
      printf 'ERROR: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      REQUESTED_ACTIONS+=("$1")
      shift
      ;;
  esac
done

if "${SHOW_ACTIONS}"; then
  list_actions
  exit 0
fi

if [[ -n "${CONFIG_FILE}" ]]; then
  [[ -f "${CONFIG_FILE}" ]] || {
    printf 'ERROR: config file does not exist: %s\n' "${CONFIG_FILE}" >&2
    exit 2
  }
  CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"
  # The config is sourced deliberately so operators can use ${VAR:-default}.
  # Only source a file controlled by the operator.
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

if [[ -f "${SCRIPT_PROJECT_ROOT}/pyproject.toml" ]]; then
  DEFAULT_WORKSPACE_ROOT="$(cd "${SCRIPT_PROJECT_ROOT}/.." && pwd)"
  DEFAULT_FE_DIR_NAME="$(basename "${SCRIPT_PROJECT_ROOT}")"
else
  DEFAULT_WORKSPACE_ROOT="${PWD}/wam-workspace"
  DEFAULT_FE_DIR_NAME="fe_pc_wam"
fi

: "${WORKSPACE_ROOT:=${DEFAULT_WORKSPACE_ROOT}}"
: "${FE_DIR_NAME:=${DEFAULT_FE_DIR_NAME}}"
: "${FE_REPO_URL:=https://github.com/Jeong-zju/fe-pc-wam.git}"
: "${FE_REF:=main}"
: "${ROBOFACTORY_DIR_NAME:=RoboFactory}"
: "${ROBOFACTORY_REPO_URL:=https://github.com/MARS-EAI/RoboFactory.git}"
: "${ROBOFACTORY_REF:=main}"
: "${WAM_PYTHON:=3.11}"
: "${ROBOFACTORY_PYTHON:=3.9}"
: "${ROBOFACTORY_ENV_MODE:=auto}"
: "${CUDA_HOME_PATH:=/usr/local/cuda-12.8}"
: "${TORCH_CUDA_ARCH_LIST_VALUE:=12.0}"
: "${REQUIRE_CUDA:=true}"
: "${REQUIRE_VULKAN:=false}"
: "${MIN_FREE_DISK_GB:=30}"
: "${HF_REQUIRE_AUTH:=true}"
: "${HF_DATASET_REPO:=}"
: "${HF_DATASET_REVISION:=main}"
: "${HF_DATASET_DIR:=datasets/robofactory_lift_barrier_m1_v1}"
: "${HF_UPLOAD_REPO:=${HF_DATASET_REPO}}"
: "${HF_UPLOAD_DIR:=${HF_DATASET_DIR}}"
: "${HF_UPLOAD_REVISION:=main}"
: "${HF_UPLOAD_MODE:=large}"
: "${HF_UPLOAD_PRIVATE:=true}"
: "${HF_UPLOAD_WORKERS:=8}"
: "${WAM_CONFIG:=configs/wam_multimodal/m1_liftbarrier_scratch.yaml}"
: "${WAM_MANIFEST:=${HF_DATASET_DIR}/training_manifest.json}"
: "${WAM_CHECKPOINT:=checkpoints/m1_liftbarrier_scratch_seed101}"
: "${WAM_SMOKE_CHECKPOINT:=checkpoints/preflight/m1_liftbarrier_automation_smoke}"
: "${TRAIN_DEVICE:=cuda:0}"
: "${TRAIN_SMOKE_STEPS_SCALE:=0.001}"
: "${CLOSED_LOOP_HOST:=127.0.0.1}"
: "${CLOSED_LOOP_PORT:=8765}"
: "${CLOSED_LOOP_DEVICE:=cuda:0}"
: "${CLOSED_LOOP_SMOKE_EPISODES:=3}"
: "${CLOSED_LOOP_SMOKE_SEED_START:=900}"
: "${CLOSED_LOOP_EPISODES:=100}"
: "${CLOSED_LOOP_SEED_START:=1000}"
: "${CLOSED_LOOP_MAX_STEPS:=500}"
: "${CLOSED_LOOP_SIM_BACKEND:=cpu}"
: "${CLOSED_LOOP_SHADER:=default}"
: "${CLOSED_LOOP_VIDEO_FPS:=20}"
: "${AUTOMATION_ROOT:=${WORKSPACE_ROOT}/.wam-automation}"
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"

FE_ROOT="${WORKSPACE_ROOT}/${FE_DIR_NAME}"
ROBOFACTORY_ROOT="${WORKSPACE_ROOT}/${ROBOFACTORY_DIR_NAME}"
LOG_DIR="${AUTOMATION_ROOT}/logs"
RUN_DIR="${AUTOMATION_ROOT}/runs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
STATE_FILE="${STATE_FILE_OVERRIDE:-${AUTOMATION_ROOT}/state}"
LOCK_FILE="${AUTOMATION_ROOT}/lock"
UV_BIN=""
CURRENT_ACTION="initialization"
CURRENT_ACTION_INDEX=-1
SERVER_PID=""

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *)
      printf 'ERROR: expected boolean, got %q\n' "$1" >&2
      return 2
      ;;
  esac
}

quote_command() {
  local value
  printf '  +'
  for value in "$@"; do
    printf ' %q' "${value}"
  done
  printf '\n'
}

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*" >&2
  return 1
}

run_cmd() {
  quote_command "$@"
  if "${DRY_RUN}"; then
    return 0
  fi
  "$@"
}

run_in() {
  local directory="$1"
  shift
  printf '  + cd %q &&' "${directory}"
  local value
  for value in "$@"; do
    printf ' %q' "${value}"
  done
  printf '\n'
  if "${DRY_RUN}"; then
    return 0
  fi
  (
    cd "${directory}"
    "$@"
  )
}

require_command() {
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  "${DRY_RUN}" && return 0
  die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] && return 0
  "${DRY_RUN}" && return 0
  die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] && return 0
  "${DRY_RUN}" && return 0
  die "required directory not found: $1"
}

fe_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${FE_ROOT}" "$1"
  fi
}

ensure_uv() {
  if [[ -n "${UV_BIN}" ]]; then
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi

  require_command curl
  local installer
  installer="${AUTOMATION_ROOT}/uv-installer.sh"
  log "uv not found; downloading the official installer"
  run_cmd mkdir -p "${AUTOMATION_ROOT}"
  run_cmd curl -LsSf https://astral.sh/uv/install.sh -o "${installer}"
  run_cmd sh "${installer}"

  local candidate
  for candidate in \
    "${HOME}/.local/bin/uv" \
    "${HOME}/.cargo/bin/uv"
  do
    if [[ -x "${candidate}" ]]; then
      UV_BIN="${candidate}"
      return 0
    fi
  done
  if "${DRY_RUN}"; then
    UV_BIN="uv"
    return 0
  fi
  die "uv installer completed but the uv binary was not found"
}

check_clean_checkout() {
  local directory="$1"
  local label="$2"
  local changes
  changes="$(git -C "${directory}" status --porcelain --untracked-files=no)"
  [[ -z "${changes}" ]] || die \
    "${label} has tracked local changes; refusing automated branch switch"
}

sync_repository() {
  local label="$1"
  local url="$2"
  local ref="$3"
  local destination="$4"

  require_command git
  if [[ ! -e "${destination}" ]]; then
    log "cloning ${label} (${ref})"
    run_cmd mkdir -p "$(dirname "${destination}")"
    run_cmd git clone "${url}" "${destination}"
    if "${DRY_RUN}"; then
      run_cmd git -C "${destination}" switch "${ref}"
      return 0
    fi
  fi

  require_dir "${destination}/.git"
  check_clean_checkout "${destination}" "${label}"
  log "fetching ${label} and switching to ${ref}"
  run_cmd git -C "${destination}" fetch --prune origin
  if "${DRY_RUN}"; then
    run_cmd git -C "${destination}" switch "${ref}"
    run_cmd git -C "${destination}" pull --ff-only
    return 0
  fi

  if git -C "${destination}" show-ref --verify --quiet "refs/heads/${ref}"; then
    run_cmd git -C "${destination}" switch "${ref}"
  elif git -C "${destination}" show-ref --verify --quiet "refs/remotes/origin/${ref}"; then
    run_cmd git -C "${destination}" switch --track -c "${ref}" "origin/${ref}"
  elif git -C "${destination}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    run_cmd git -C "${destination}" switch --detach "${ref}"
  else
    die "${label} ref not found after fetch: ${ref}"
  fi

  local branch
  branch="$(git -C "${destination}" symbolic-ref --quiet --short HEAD || true)"
  if [[ -n "${branch}" ]] &&
     git -C "${destination}" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
    run_cmd git -C "${destination}" pull --ff-only origin "${branch}"
  fi
}

action_code() {
  sync_repository "FE-PC WAM" "${FE_REPO_URL}" "${FE_REF}" "${FE_ROOT}"
}

action_robofactory() {
  sync_repository \
    "RoboFactory" \
    "${ROBOFACTORY_REPO_URL}" \
    "${ROBOFACTORY_REF}" \
    "${ROBOFACTORY_ROOT}"
}

action_env() {
  require_file "${FE_ROOT}/pyproject.toml"
  require_file "${FE_ROOT}/uv.lock"
  ensure_uv
  run_cmd mkdir -p "${FE_ROOT}/.uv-cache" "${FE_ROOT}/.uv-python"
  run_in "${FE_ROOT}" env \
    "UV_CACHE_DIR=${FE_ROOT}/.uv-cache" \
    "UV_PYTHON_INSTALL_DIR=${FE_ROOT}/.uv-python" \
    "${UV_BIN}" python install "${WAM_PYTHON}"
  run_in "${FE_ROOT}" env \
    "UV_CACHE_DIR=${FE_ROOT}/.uv-cache" \
    "UV_PYTHON_INSTALL_DIR=${FE_ROOT}/.uv-python" \
    "${UV_BIN}" sync --frozen --python "${WAM_PYTHON}"
}

action_robofactory_env() {
  require_dir "${ROBOFACTORY_ROOT}"
  ensure_uv
  run_cmd mkdir -p \
    "${ROBOFACTORY_ROOT}/.uv-cache" \
    "${ROBOFACTORY_ROOT}/.uv-python"

  local mode="${ROBOFACTORY_ENV_MODE}"
  if [[ "${mode}" == "auto" ]]; then
    if [[ -f "${ROBOFACTORY_ROOT}/pyproject.toml" &&
          -f "${ROBOFACTORY_ROOT}/uv.lock" ]]; then
      mode="locked"
    else
      mode="requirements"
    fi
  fi

  case "${mode}" in
    locked)
      require_file "${ROBOFACTORY_ROOT}/pyproject.toml"
      require_file "${ROBOFACTORY_ROOT}/uv.lock"
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "UV_PYTHON_INSTALL_DIR=${ROBOFACTORY_ROOT}/.uv-python" \
        "${UV_BIN}" python install "${ROBOFACTORY_PYTHON}"
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "UV_PYTHON_INSTALL_DIR=${ROBOFACTORY_ROOT}/.uv-python" \
        "${UV_BIN}" sync --frozen --python "${ROBOFACTORY_PYTHON}"
      ;;
    requirements)
      require_file "${ROBOFACTORY_ROOT}/robofactory/requirements.txt"
      require_file "${ROBOFACTORY_ROOT}/setup.py"
      log "RoboFactory has no root uv.lock; using its pinned requirements fallback"
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "UV_PYTHON_INSTALL_DIR=${ROBOFACTORY_ROOT}/.uv-python" \
        "${UV_BIN}" python install "${ROBOFACTORY_PYTHON}"
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "UV_PYTHON_INSTALL_DIR=${ROBOFACTORY_ROOT}/.uv-python" \
        "${UV_BIN}" venv --python "${ROBOFACTORY_PYTHON}" .venv
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "${UV_BIN}" pip install \
        --python .venv/bin/python \
        -r robofactory/requirements.txt
      run_in "${ROBOFACTORY_ROOT}" env \
        "UV_CACHE_DIR=${ROBOFACTORY_ROOT}/.uv-cache" \
        "${UV_BIN}" pip install \
        --python .venv/bin/python \
        --no-deps \
        --editable .
      ;;
    *)
      die "ROBOFACTORY_ENV_MODE must be auto, locked, or requirements"
      ;;
  esac
}

action_assets() {
  require_file "${ROBOFACTORY_ROOT}/.venv/bin/python"
  require_file "${ROBOFACTORY_ROOT}/robofactory/script/download_assets.py"
  run_in "${ROBOFACTORY_ROOT}/robofactory" \
    "${ROBOFACTORY_ROOT}/.venv/bin/python" \
    script/download_assets.py
}

wam_uv() {
  ensure_uv
  run_in "${FE_ROOT}" env \
    "HF_HUB_DISABLE_XET=1" \
    "HF_XET_HIGH_PERFORMANCE=0" \
    "UV_CACHE_DIR=${FE_ROOT}/.uv-cache" \
    "${UV_BIN}" run --frozen "$@"
}

action_hf_auth() {
  if ! is_true "${HF_REQUIRE_AUTH}"; then
    log "HF_REQUIRE_AUTH=false; skipping authentication check"
    return 0
  fi
  if "${DRY_RUN}"; then
    log "would verify Hugging Face authentication without printing HF_TOKEN"
    wam_uv hf auth whoami
    return 0
  fi
  wam_uv hf auth whoami
}

action_hf_download() {
  [[ -n "${HF_DATASET_REPO}" ]] || die \
    "HF_DATASET_REPO is empty; set it in the config or environment"
  local destination
  destination="$(fe_path "${HF_DATASET_DIR}")"
  run_cmd mkdir -p "${destination}"
  wam_uv hf download \
    "${HF_DATASET_REPO}" \
    --type dataset \
    --revision "${HF_DATASET_REVISION}" \
    --local-dir "${destination}" \
    --max-workers 1
}

action_hf_upload() {
  [[ -n "${HF_UPLOAD_REPO}" ]] || die \
    "HF_UPLOAD_REPO is empty; set it in the config or environment"
  local source
  source="$(fe_path "${HF_UPLOAD_DIR}")"
  require_dir "${source}"

  local -a privacy=(--public)
  if is_true "${HF_UPLOAD_PRIVATE}"; then
    privacy=(--private)
  fi
  wam_uv hf repos create \
    "${HF_UPLOAD_REPO}" \
    --type dataset \
    --exist-ok \
    "${privacy[@]}"

  case "${HF_UPLOAD_MODE}" in
    single)
      wam_uv hf upload \
        "${HF_UPLOAD_REPO}" \
        "${source}" \
        . \
        --type dataset \
        --revision "${HF_UPLOAD_REVISION}" \
        --exclude ".cache/**" \
        --commit-message "Upload FE-PC WAM dataset"
      ;;
    large)
      wam_uv hf upload-large-folder \
        "${HF_UPLOAD_REPO}" \
        "${source}" \
        --type dataset \
        --revision "${HF_UPLOAD_REVISION}" \
        --exclude ".cache/**" \
        --num-workers "${HF_UPLOAD_WORKERS}"
      ;;
    *)
      die "HF_UPLOAD_MODE must be single or large"
      ;;
  esac
}

action_vision() {
  require_file "${FE_ROOT}/scripts/prepare_dinov3_encoder.py"
  wam_uv python scripts/prepare_dinov3_encoder.py \
    --encoder dinov3_vitl16_lvd \
    --output-dir artifacts/vision/dinov3_vitl16_lvd
}

action_doctor() {
  require_command git
  require_command curl
  require_command sha256sum
  require_command df
  require_dir "${FE_ROOT}/.git"
  require_dir "${ROBOFACTORY_ROOT}/.git"
  require_file "${FE_ROOT}/.venv/bin/python"
  require_file "${ROBOFACTORY_ROOT}/.venv/bin/python"

  if "${DRY_RUN}"; then
    run_cmd df -Pk "${WORKSPACE_ROOT}"
    if is_true "${REQUIRE_CUDA}"; then
      run_cmd nvidia-smi
    fi
    if is_true "${REQUIRE_VULKAN}"; then
      run_cmd vulkaninfo --summary
    fi
    run_cmd "${FE_ROOT}/.venv/bin/python" --version
    run_cmd "${ROBOFACTORY_ROOT}/.venv/bin/python" --version
    return 0
  fi

  local available_kb required_kb
  available_kb="$(df -Pk "${WORKSPACE_ROOT}" | awk 'NR == 2 {print $4}')"
  [[ "${available_kb}" =~ ^[0-9]+$ ]] || die \
    "could not determine free disk space for ${WORKSPACE_ROOT}"
  [[ "${MIN_FREE_DISK_GB}" =~ ^[0-9]+$ ]] || die \
    "MIN_FREE_DISK_GB must be a non-negative integer"
  required_kb="$((MIN_FREE_DISK_GB * 1024 * 1024))"
  if ((available_kb < required_kb)); then
    die "only $((available_kb / 1024 / 1024)) GiB free; ${MIN_FREE_DISK_GB} GiB required"
  fi
  log "disk preflight passed: $((available_kb / 1024 / 1024)) GiB free"

  if is_true "${REQUIRE_CUDA}"; then
    require_command nvidia-smi
    run_cmd nvidia-smi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    run_cmd nvidia-smi
  else
    log "CUDA check disabled and nvidia-smi is unavailable"
  fi

  if is_true "${REQUIRE_VULKAN}"; then
    require_command vulkaninfo
    run_cmd vulkaninfo --summary
  elif command -v vulkaninfo >/dev/null 2>&1; then
    run_cmd vulkaninfo --summary
  else
    log "Vulkan CLI check disabled; RoboFactory runtime remains authoritative"
  fi

  run_cmd "${FE_ROOT}/.venv/bin/python" --version
  run_cmd "${ROBOFACTORY_ROOT}/.venv/bin/python" --version
  if ! "${DRY_RUN}"; then
    [[ "$("${FE_ROOT}/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] ||
      die "WAM environment is not Python 3.11"
    [[ "$("${ROBOFACTORY_ROOT}/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.9" ]] ||
      die "RoboFactory environment is not Python 3.9"
  fi
}

action_data_check() {
  local manifest
  manifest="$(fe_path "${WAM_MANIFEST}")"
  require_file "${manifest}"
  wam_uv python scripts/smoke_m1_data_protocol.py \
    --manifest "${manifest}" \
    --splits train validation test \
    --state-history 32 \
    --action-chunk 8 \
    --visual-history 2 \
    --future-horizons 1 2 4 8 \
    --camera global
}

action_test() {
  wam_uv pytest -q \
    tests/test_wam_automation_script.py \
    tests/test_m1_scratch_path.py \
    tests/test_m1_manifest_dataset.py \
    tests/test_robofactory_m1_training_artifacts.py \
    tests/test_robofactory_m1_closed_loop.py \
    -k "not test_m1_window_projection_supports_persistent_hdf5_workers"
}

action_train_smoke() {
  local output
  output="$(fe_path "${WAM_SMOKE_CHECKPOINT}")"
  require_file "$(fe_path "${WAM_CONFIG}")"
  if [[ -e "${output}" ]] && ! "${DRY_RUN}"; then
    die "smoke checkpoint already exists; refusing to overwrite: ${output}"
  fi
  wam_uv python scripts/train_liftbarrier_m1_scratch.py \
    --config "$(fe_path "${WAM_CONFIG}")" \
    --device "${TRAIN_DEVICE}" \
    --steps-scale "${TRAIN_SMOKE_STEPS_SCALE}" \
    --output "${output}"
}

action_train() {
  local checkpoint
  checkpoint="$(fe_path "${WAM_CHECKPOINT}")"
  require_file "$(fe_path "${WAM_CONFIG}")"
  if [[ -e "${checkpoint}" ]] && ! "${DRY_RUN}"; then
    die "checkpoint already exists; refusing to overwrite: ${checkpoint}"
  fi
  wam_uv python scripts/train_liftbarrier_m1_scratch.py \
    --config "$(fe_path "${WAM_CONFIG}")" \
    --device "${TRAIN_DEVICE}" \
    --output "${checkpoint}"
}

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping RoboFactory server pid ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

closed_loop_gate() {
  local summary="$1"
  local expected_episodes="$2"
  local formal="$3"
  local gate_code
  gate_code='
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
formal = sys.argv[3] == "true"
report = json.loads(path.read_text(encoding="utf-8"))
passed = (
    report.get("completed") is True
    and report.get("fatal_error") is None
    and report.get("episodes_completed") == expected
)
if formal:
    passed = (
        passed
        and report.get("formal_benchmark", {}).get("reportable") is True
        and expected == 100
    )
print(json.dumps({
    "passed": passed,
    "formal": formal,
    "episodes_completed": report.get("episodes_completed"),
    "successes": report.get("successes"),
    "success_rate": report.get("success_rate"),
}, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
'
  run_cmd "${FE_ROOT}/.venv/bin/python" \
    -c "${gate_code}" "${summary}" "${expected_episodes}" "${formal}"
}

check_formal_robofactory_contract() {
  local check_code
  check_code='
import hashlib
import json
import pathlib
import runpy
import sys

fe_root = pathlib.Path(sys.argv[1])
rf_root = pathlib.Path(sys.argv[2])
namespace = runpy.run_path(str(fe_root / "scripts/serve_robofactory_m1_rollout.py"))
expected_sources = namespace["FORMAL_LIFTBARRIER_RF_SOURCE_SHA256"]
expected_config = namespace["FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256"]

mismatches = {}
for relative, expected in expected_sources.items():
    path = rf_root / relative
    observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if observed != expected:
        mismatches[relative] = {"expected": expected, "observed": observed}

config_relative = "robofactory/configs/table/lift_barrier.yaml"
config_path = rf_root / config_relative
config_observed = (
    hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_path.is_file()
    else None
)
if config_observed != expected_config:
    mismatches[config_relative] = {
        "expected": expected_config,
        "observed": config_observed,
    }

print(json.dumps({
    "passed": not mismatches,
    "robofactory_root": str(rf_root),
    "mismatches": mismatches,
}, indent=2, sort_keys=True))
raise SystemExit(0 if not mismatches else 1)
'
  log "checking the formal RoboFactory source/config contract"
  run_cmd "${FE_ROOT}/.venv/bin/python" -c "${check_code}" \
    "${FE_ROOT}" "${ROBOFACTORY_ROOT}"
}

run_closed_loop() {
  local label="$1"
  local episodes="$2"
  local seed_start="$3"
  local formal="$4"
  local output="${FE_ROOT}/outputs/automation_${label}_${RUN_ID}"
  local checkpoint_ref="${WAM_CHECKPOINT}"
  if [[ "${formal}" != "true" ]] && is_true "${SMOKE_USES_PREFLIGHT_CHECKPOINT:-false}"; then
    checkpoint_ref="${WAM_SMOKE_CHECKPOINT}"
  fi
  local checkpoint
  checkpoint="$(fe_path "${checkpoint_ref}")"

  require_file "${ROBOFACTORY_ROOT}/.venv/bin/python"
  require_file "${FE_ROOT}/.venv/bin/python"
  require_dir "${checkpoint}"
  require_file "$(fe_path "${WAM_CONFIG}")"
  if [[ "${formal}" == "true" ]]; then
    [[ "${episodes}" == "100" ]] || die \
      "formal validation requires CLOSED_LOOP_EPISODES=100"
    [[ "${seed_start}" == "1000" ]] || die \
      "formal validation requires CLOSED_LOOP_SEED_START=1000"
    check_formal_robofactory_contract
  fi
  if [[ -e "${output}" ]] && ! "${DRY_RUN}"; then
    die "closed-loop output already exists: ${output}"
  fi

  local -a server_command=(
    env
    "CUDA_HOME=${CUDA_HOME_PATH}"
    "PATH=${CUDA_HOME_PATH}/bin:${PATH}"
    "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE}"
    "${ROBOFACTORY_ROOT}/.venv/bin/python"
    "${FE_ROOT}/scripts/serve_robofactory_m1_rollout.py"
    --robofactory-root "${ROBOFACTORY_ROOT}"
    --host "${CLOSED_LOOP_HOST}"
    --port "${CLOSED_LOOP_PORT}"
    --episodes "${episodes}"
    --seed-start "${seed_start}"
    --max-steps "${CLOSED_LOOP_MAX_STEPS}"
    --sim-backend "${CLOSED_LOOP_SIM_BACKEND}"
    --shader "${CLOSED_LOOP_SHADER}"
    --video-fps "${CLOSED_LOOP_VIDEO_FPS}"
    --output-dir "${output}"
  )
  if [[ "${formal}" != "true" ]]; then
    server_command+=(--no-video)
  fi

  log "starting RoboFactory ${label} server"
  printf '  + (cd %q &&' "${ROBOFACTORY_ROOT}"
  local value
  for value in "${server_command[@]}"; do
    printf ' %q' "${value}"
  done
  printf ') &\n'

  if "${DRY_RUN}"; then
    wam_uv python scripts/run_robofactory_m1_inference.py \
      --checkpoint "${checkpoint}" \
      --config "$(fe_path "${WAM_CONFIG}")" \
      --device "${CLOSED_LOOP_DEVICE}" \
      --host "${CLOSED_LOOP_HOST}" \
      --port "${CLOSED_LOOP_PORT}"
    return 0
  fi

  (
    cd "${ROBOFACTORY_ROOT}"
    "${server_command[@]}"
  ) &
  SERVER_PID=$!

  if ! wam_uv python scripts/run_robofactory_m1_inference.py \
    --checkpoint "${checkpoint}" \
    --config "$(fe_path "${WAM_CONFIG}")" \
    --device "${CLOSED_LOOP_DEVICE}" \
    --host "${CLOSED_LOOP_HOST}" \
    --port "${CLOSED_LOOP_PORT}"
  then
    cleanup_server
    die "${label} inference client failed"
  fi

  local server_status=0
  wait "${SERVER_PID}" || server_status=$?
  SERVER_PID=""
  ((server_status == 0)) || die \
    "${label} RoboFactory server failed with exit ${server_status}"
  closed_loop_gate \
    "${output}/rollout_summary.json" \
    "${episodes}" \
    "${formal}"
}

action_validate_smoke() {
  run_closed_loop \
    "closed_loop_smoke" \
    "${CLOSED_LOOP_SMOKE_EPISODES}" \
    "${CLOSED_LOOP_SMOKE_SEED_START}" \
    false
}

action_validate() {
  run_closed_loop \
    "closed_loop_formal" \
    "${CLOSED_LOOP_EPISODES}" \
    "${CLOSED_LOOP_SEED_START}" \
    true
}

action_snapshot() {
  if "${DRY_RUN}"; then
    log "would save provenance under ${RUN_DIR}"
    return 0
  fi
  run_cmd mkdir -p "${RUN_DIR}"
  local snapshot="${RUN_DIR}/provenance.json"
  local snapshot_code
  snapshot_code='
import json
import os
import pathlib
import platform
import subprocess
import sys

def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

payload = {
    "format_version": "wam.automation.provenance/1",
    "run_id": sys.argv[1],
    "fe_root": sys.argv[2],
    "fe_git_head": git_head(sys.argv[2]),
    "robofactory_root": sys.argv[3],
    "robofactory_git_head": git_head(sys.argv[3]),
    "dataset_repo": sys.argv[4] or None,
    "dataset_revision": sys.argv[5],
    "dataset_manifest": sys.argv[6],
    "checkpoint": sys.argv[7],
    "python": platform.python_version(),
    "platform": platform.platform(),
}
path = pathlib.Path(sys.argv[8])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
'
  run_cmd "${FE_ROOT}/.venv/bin/python" -c "${snapshot_code}" \
    "${RUN_ID}" \
    "${FE_ROOT}" \
    "${ROBOFACTORY_ROOT}" \
    "${HF_DATASET_REPO}" \
    "${HF_DATASET_REVISION}" \
    "$(fe_path "${WAM_MANIFEST}")" \
    "$(fe_path "${WAM_CHECKPOINT}")" \
    "${snapshot}"
}

declare -a EXPANDED_ACTIONS=()

append_action_token() {
  local token="$1"
  case "${token}" in
    bootstrap)
      EXPANDED_ACTIONS+=(
        code robofactory env robofactory-env assets doctor
      )
      ;;
    full-smoke)
      EXPANDED_ACTIONS+=(
        code robofactory env robofactory-env assets hf-auth hf-download vision
        doctor data-check test train-smoke validate-smoke snapshot
      )
      ;;
    full)
      EXPANDED_ACTIONS+=(
        code robofactory env robofactory-env assets hf-auth hf-download vision
        doctor data-check test train validate snapshot
      )
      ;;
    *)
      EXPANDED_ACTIONS+=("${token}")
      ;;
  esac
}

for requested in "${REQUESTED_ACTIONS[@]}"; do
  IFS=',' read -r -a comma_actions <<<"${requested}"
  for requested_action in "${comma_actions[@]}"; do
    [[ -n "${requested_action}" ]] || continue
    append_action_token "${requested_action}"
  done
done

((${#EXPANDED_ACTIONS[@]} > 0)) || {
  usage >&2
  exit 2
}

SMOKE_USES_PREFLIGHT_CHECKPOINT=false
seen_smoke_training=false
for requested in "${EXPANDED_ACTIONS[@]}"; do
  if [[ "${requested}" == "train-smoke" ]]; then
    seen_smoke_training=true
  elif [[ "${requested}" == "validate-smoke" ]] && "${seen_smoke_training}"; then
    SMOKE_USES_PREFLIGHT_CHECKPOINT=true
  fi
done

for requested in "${EXPANDED_ACTIONS[@]}"; do
  case "${requested}" in
    code|robofactory|env|robofactory-env|assets|hf-auth|hf-download|hf-upload|\
    vision|doctor|data-check|test|train-smoke|train|validate-smoke|validate|\
    snapshot)
      ;;
    *)
      printf 'ERROR: unknown action: %s\n' "${requested}" >&2
      printf 'Use --list to see supported actions.\n' >&2
      exit 2
      ;;
  esac
done

if ! "${DRY_RUN}"; then
  mkdir -p "${LOG_DIR}" "${RUN_DIR}" "$(dirname "${STATE_FILE}")"
  touch "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

if command -v flock >/dev/null 2>&1 && ! "${DRY_RUN}"; then
  exec 9>"${LOCK_FILE}"
  flock -n 9 || {
    die "another automation process holds ${LOCK_FILE}"
    exit 1
  }
fi

ACTION_SEQUENCE="$(IFS=,; printf '%s' "${EXPANDED_ACTIONS[*]}")"
require_command sha256sum
CONFIG_SHA256=""
if [[ -n "${CONFIG_FILE}" ]]; then
  CONFIG_SHA256="$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
fi
FINGERPRINT="$(
  printf '%s\n' \
    "${CONFIG_SHA256}" \
    "${FE_REPO_URL}" "${FE_REF}" \
    "${ROBOFACTORY_REPO_URL}" "${ROBOFACTORY_REF}" \
    "${ROBOFACTORY_ENV_MODE}" "${WAM_PYTHON}" "${ROBOFACTORY_PYTHON}" \
    "${HF_DATASET_REPO}" "${HF_DATASET_REVISION}" "${HF_DATASET_DIR}" \
    "${HF_UPLOAD_REPO}" "${HF_UPLOAD_REVISION}" "${HF_UPLOAD_DIR}" \
    "${WAM_CONFIG}" "${WAM_CHECKPOINT}" "${WAM_SMOKE_CHECKPOINT}" \
    "${TRAIN_DEVICE}" "${CLOSED_LOOP_DEVICE}" \
    "${CLOSED_LOOP_EPISODES}" "${CLOSED_LOOP_SEED_START}" \
    "${ACTION_SEQUENCE}" |
    sha256sum |
    awk '{print $1}'
)"

initialize_state() {
  if "${DRY_RUN}"; then
    return 0
  fi
  if "${RESUME}"; then
    require_file "${STATE_FILE}"
    local observed
    observed="$(sed -n 's/^fingerprint=//p' "${STATE_FILE}" | head -n 1)"
    [[ "${observed}" == "${FINGERPRINT}" ]] || die \
      "resume state fingerprint does not match this pipeline/config"
  else
    printf 'fingerprint=%s\n' "${FINGERPRINT}" >"${STATE_FILE}"
  fi
}

position_completed() {
  local position="$1"
  local action="$2"
  grep -Fqx "completed=${position}:${action}" "${STATE_FILE}" 2>/dev/null
}

mark_completed() {
  local position="$1"
  local action="$2"
  "${DRY_RUN}" && return 0
  printf 'completed=%s:%s\n' "${position}" "${action}" >>"${STATE_FILE}"
}

on_error() {
  local status=$?
  log "FAILED action ${CURRENT_ACTION_INDEX}:${CURRENT_ACTION} (exit ${status})"
  log "resume with the same config/actions plus: --resume --state-file ${STATE_FILE}"
  exit "${status}"
}

on_exit() {
  cleanup_server
}

trap on_error ERR
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

initialize_state
log "run_id=${RUN_ID}"
log "workspace=${WORKSPACE_ROOT}"
log "actions=${ACTION_SEQUENCE}"
log "log=${LOG_FILE}"
if "${DRY_RUN}"; then
  log "dry-run enabled"
fi

for index in "${!EXPANDED_ACTIONS[@]}"; do
  CURRENT_ACTION_INDEX="${index}"
  CURRENT_ACTION="${EXPANDED_ACTIONS[${index}]}"
  if "${RESUME}" && position_completed "${index}" "${CURRENT_ACTION}"; then
    log "SKIP completed action ${index}:${CURRENT_ACTION}"
    continue
  fi

  log "START action ${index}:${CURRENT_ACTION}"
  case "${CURRENT_ACTION}" in
    code) action_code ;;
    robofactory) action_robofactory ;;
    env) action_env ;;
    robofactory-env) action_robofactory_env ;;
    assets) action_assets ;;
    hf-auth) action_hf_auth ;;
    hf-download) action_hf_download ;;
    hf-upload) action_hf_upload ;;
    vision) action_vision ;;
    doctor) action_doctor ;;
    data-check) action_data_check ;;
    test) action_test ;;
    train-smoke) action_train_smoke ;;
    train) action_train ;;
    validate-smoke) action_validate_smoke ;;
    validate) action_validate ;;
    snapshot) action_snapshot ;;
  esac
  mark_completed "${index}" "${CURRENT_ACTION}"
  log "DONE action ${index}:${CURRENT_ACTION}"
done

CURRENT_ACTION="complete"
CURRENT_ACTION_INDEX="${#EXPANDED_ACTIONS[@]}"
log "pipeline completed successfully"
log "state=${STATE_FILE}"
