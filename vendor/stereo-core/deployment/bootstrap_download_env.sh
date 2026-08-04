#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /workspace/datasets/robofactory_multitask /workspace/artifacts /workspace/logs /workspace/runs

if [[ ! -x /venv/main/bin/python ]]; then
  uv python install 3.10
  uv venv --python 3.10 /venv/main
fi

uv pip install --python /venv/main/bin/python \
  'huggingface_hub[hf_xet]>=0.34,<2' \
  'hf_xet>=1.1'

/venv/main/bin/hf --help >/dev/null

