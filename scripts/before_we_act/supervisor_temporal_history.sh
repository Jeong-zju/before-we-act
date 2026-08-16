#!/usr/bin/env bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export TEMPORAL_REPO_ROOT="${TEMPORAL_REPO_ROOT:-/workspace/fe-pc-wam}"
export TEMPORAL_RUN_ROOT="${TEMPORAL_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v7}"
cd "${TEMPORAL_REPO_ROOT}"
exec "${TEMPORAL_REPO_ROOT}/scripts/before_we_act/run_temporal_history_pipeline.sh"
