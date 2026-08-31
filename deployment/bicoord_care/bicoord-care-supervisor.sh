#!/usr/bin/env bash
set -Eeuo pipefail

# Opt-in entry point.  Credentials (including HF_TOKEN) are inherited from
# the environment and never embedded in command arguments or receipts.
# A deployment may pin the checkout in the supervisor config.  Preserve that
# non-secret value across the generic .env load so a stale prior-run pin in
# the credential file cannot silently select a different source revision.
pinned_source_revision="${BICOORD_CARE_PINNED_SOURCE_REVISION:-}"
if [[ -f "${BICOORD_ENV_FILE:-/workspace/.env}" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${BICOORD_ENV_FILE:-/workspace/.env}"
  set +a
fi
if [[ -n "${pinned_source_revision}" ]]; then
  export BICOORD_CARE_SOURCE_REVISION="${pinned_source_revision}"
fi

export BICOORD_CARE_REPO="${BICOORD_CARE_REPO:-/workspace/repos/before-we-act}"
export BICOORD_BENCH_REPO="${BICOORD_BENCH_REPO:-/workspace/repos/bicoord-bench}"
export BICOORD_DATASET="${BICOORD_DATASET:-/workspace/repos/bicoord-bench/data}"
export BICOORD_CARE_RUN="${BICOORD_CARE_RUN:-/workspace/runs/bicoord-care-v2}"
export BICOORD_DINO_MODEL="${BICOORD_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
export BICOORD_CARE_PYTHON="${BICOORD_CARE_PYTHON:-/venv/main/bin/python}"

exec "${BICOORD_CARE_PYTHON}" -u -m deployment.bicoord_care.supervisor "$@"
