#!/bin/bash
set -euo pipefail
umask 077
set -a
. /workspace/.env
set +a
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec /venv/main/bin/python -u /workspace/tools/bicoord/download_base_assets.py
