#!/usr/bin/env bash

# Portable entry point for the verified no-wrist CoRE reproduction.  The model
# and trainer themselves are copied byte-for-byte from the user's completed
# 2026-08-03 Vast.ai run under vendor/stereo-core/stereo_core/.

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit $?
CORE_ROOT="${FE_ROOT}/vendor/stereo-core"
DATA_ROOT="${S10_CORE_DATA_ROOT:-${FE_ROOT}/datasets/robofactory_multitask}"
CORE_PYTHON="${S10_CORE_PYTHON:-/venv/robofactory-act/bin/python}"
DINO_MODEL="${S10_CORE_DINO_MODEL:-${FE_ROOT}/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
OUTPUT="${S10_CORE_OUTPUT:-${FE_ROOT}/outputs/s10_core/no_wrist_stereo_core_120k}"
UPDATES="${S10_CORE_UPDATES:-120000}"
WORKERS="${S10_CORE_WORKERS:-8}"

fail() {
  printf >&2 'S10 CoRE launcher: %s\n' "$*"
  exit 1
}

case "${UPDATES}" in
  ''|*[!0-9]*) fail "S10_CORE_UPDATES must be a positive integer" ;;
  0) fail "S10_CORE_UPDATES must be positive" ;;
esac
case "${WORKERS}" in
  ''|*[!0-9]*) fail "S10_CORE_WORKERS must be a non-negative integer" ;;
esac

[[ -x "${CORE_PYTHON}" ]] || fail "Python is missing: ${CORE_PYTHON}"
[[ -d "${DINO_MODEL}" ]] || fail "pinned DINOv3-B/16 directory is missing: ${DINO_MODEL}"
[[ -f "${CORE_ROOT}/stereo_core/train_no_wrist_pair.py" ]] || \
  fail "vendored no-wrist CoRE trainer is missing"

TASKS=(
  lift_barrier
  camera_alignment
  three_robots_stack_cube
  long_pipeline_delivery
  take_photo
)
MANIFESTS=()
for task in "${TASKS[@]}"; do
  manifest="${DATA_ROOT}/${task}/training_manifest.json"
  [[ -f "${manifest}" ]] || fail "user training manifest is missing: ${manifest}"
  MANIFESTS+=("${manifest}")
done

EXTRA=()
if [[ "${UPDATES}" != 120000 ]]; then
  EXTRA+=(--allow-preflight)
fi
if [[ -f "${OUTPUT}/checkpoint_latest.pt" ]]; then
  EXTRA+=(--resume "${OUTPUT}/checkpoint_latest.pt")
fi

mkdir -p "${OUTPUT}" || fail "cannot create output directory: ${OUTPUT}"
printf 'S10 CoRE: data=%s output=%s updates=%s workers=%s\n' \
  "${DATA_ROOT}" "${OUTPUT}" "${UPDATES}" "${WORKERS}"

export PYTHONPATH="${CORE_ROOT}/stereo_core${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CORE_PYTHON}" -u \
  "${CORE_ROOT}/stereo_core/train_no_wrist_pair.py" \
  --manifests "${MANIFESTS[@]}" \
  --dino-model "${DINO_MODEL}" \
  --output "${OUTPUT}" \
  --updates "${UPDATES}" \
  --batch-size 40 \
  --workers "${WORKERS}" \
  --save-every 1000 \
  --milestones 20000,40000,60000,80000,100000,120000 \
  "${EXTRA[@]}"
