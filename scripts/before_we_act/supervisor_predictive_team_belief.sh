#!/usr/bin/env bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
exec /workspace/fe-pc-wam/scripts/before_we_act/run_predictive_team_belief_pipeline.sh
