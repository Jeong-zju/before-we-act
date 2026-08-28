#!/bin/bash
set -euo pipefail
cd /workspace/repos/RoboFactory
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(</workspace/.secrets/hf_token)"
PYTHONPATH=/workspace/repos/RoboFactory /venv/main/bin/python script/download_assets.py
test -d assets
printf '{"status":"complete"}\n' > /workspace/runs/mars_act/assets.json
