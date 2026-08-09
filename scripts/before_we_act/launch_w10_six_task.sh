#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE_ROOT="${ROOT}/vendor/stereo-core"
DATA_ROOT="${W10_SIX_DATA_ROOT:-/workspace/datasets/robofactory_multitask}"
PYTHON_BIN="${W10_SIX_PYTHON:-/venv/robofactory-act/bin/python}"
DINO_MODEL="${W10_SIX_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
OUTPUT="${W10_SIX_OUTPUT:-/workspace/bwa_runs/w10-six-task-v1/train/formal}"
UPDATES="${W10_SIX_UPDATES:-120000}"
WORKERS="${W10_SIX_WORKERS:-8}"

fail() {
  printf >&2 'W10 six-task launcher: %s\n' "$*"
  exit 1
}

case "${UPDATES}" in
  ''|*[!0-9]*|0) fail "W10_SIX_UPDATES must be a positive integer" ;;
esac
case "${WORKERS}" in
  ''|*[!0-9]*) fail "W10_SIX_WORKERS must be a non-negative integer" ;;
esac

[[ -x "${PYTHON_BIN}" ]] || fail "Python is missing: ${PYTHON_BIN}"
[[ -d "${DINO_MODEL}" ]] || fail "pinned DINOv3 directory is missing: ${DINO_MODEL}"
[[ -f "${ROOT}/before_we_act/train_w10_six_task.py" ]] || fail "six-task trainer is missing"

TASKS=(
  lift_barrier
  camera_alignment
  long_pipeline_delivery
  take_photo
  pass_shoe
  place_food
)
MANIFESTS=()
for task in "${TASKS[@]}"; do
  manifest="${DATA_ROOT}/${task}/training_manifest.json"
  [[ -f "${manifest}" ]] || fail "training manifest is missing: ${manifest}"
  MANIFESTS+=("${manifest}")
done

EXTRA=()
if [[ "${UPDATES}" != 120000 ]]; then
  EXTRA+=(--allow-preflight)
fi
if [[ -f "${OUTPUT}/checkpoint_latest.pt" ]]; then
  EXTRA+=(--resume "${OUTPUT}/checkpoint_latest.pt")
fi

mkdir -p "${OUTPUT}"
export PYTHONPATH="${CORE_ROOT}/stereo_core:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u "${ROOT}/before_we_act/train_w10_six_task.py" \
  --manifests "${MANIFESTS[@]}" \
  --dino-model "${DINO_MODEL}" \
  --output "${OUTPUT}" \
  --updates "${UPDATES}" \
  --batch-size 48 \
  --workers "${WORKERS}" \
  --save-every 1000 \
  --milestones 20000,40000,60000,80000,100000,120000 \
  "${EXTRA[@]}"
