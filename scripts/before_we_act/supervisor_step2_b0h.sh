#!/usr/bin/env bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export STEP2_REPO_ROOT="${STEP2_REPO_ROOT:-/workspace/fe-pc-wam-step2}"
export STEP2_RUN_ROOT="${STEP2_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v2}"
cd "${STEP2_REPO_ROOT}"
exec "${STEP2_REPO_ROOT}/scripts/before_we_act/run_step2_b0h_pipeline.sh"
