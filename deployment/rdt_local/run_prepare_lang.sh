#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/repos/rdt-1b
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
exec /workspace/venvs/rdt/bin/python /workspace/repos/before-we-act/deployment/rdt_local/prepare_lang_embeds.py
