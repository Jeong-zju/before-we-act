#!/usr/bin/env bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
exec /workspace/fe-pc-wam-b-core/scripts/before_we_act/run_b3_n1_pipeline.sh
