#!/usr/bin/env bash
set -Eeuo pipefail

# Opt-in entry point.  Credentials (including HF_TOKEN) are inherited from
# the environment and never embedded in command arguments or receipts.
if [[ -f "${BICOORD_ENV_FILE:-/workspace/.env}" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${BICOORD_ENV_FILE:-/workspace/.env}"
  set +a
fi

export BICOORD_CARE_REPO="${BICOORD_CARE_REPO:-/workspace/repos/before-we-act}"
export BICOORD_BENCH_REPO="${BICOORD_BENCH_REPO:-/workspace/repos/bicoord-bench}"
export BICOORD_DATASET="${BICOORD_DATASET:-/workspace/repos/bicoord-bench/data}"
export BICOORD_CARE_RUN="${BICOORD_CARE_RUN:-/workspace/runs/bicoord-care-v1}"
export BICOORD_DINO_MODEL="${BICOORD_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
export BICOORD_CARE_PYTHON="${BICOORD_CARE_PYTHON:-/venv/main/bin/python}"

exec "${BICOORD_CARE_PYTHON}" -u -m deployment.bicoord_care.supervisor "$@"
