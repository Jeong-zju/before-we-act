#!/usr/bin/env bash
set -Eeuo pipefail
export DUO_PI05_REPO=${DUO_PI05_REPO:-/workspace/repos/before-we-act}
export DUO_PI05_OPENPI=${DUO_PI05_OPENPI:-/workspace/repos/openpi}
export DUO_PI05_DUOBENCH=${DUO_PI05_DUOBENCH:-/workspace/repos/duobench}
export DUO_PI05_RCS=${DUO_PI05_RCS:-/workspace/repos/robot-control-stack}
export DUO_PI05_RUN=${DUO_PI05_RUN:-/workspace/runs/pi05_duo}
export DUO_PI05_DATASET=${DUO_PI05_DATASET:-/workspace/datasets/duobench}
export DUO_PI05_PYTHON=${DUO_PI05_PYTHON:-/workspace/venvs/openpi/bin/python}
export DUO_PI05_SIM_PYTHON=${DUO_PI05_SIM_PYTHON:-/workspace/venvs/duobench/bin/python}
export PYTHONPATH="$DUO_PI05_REPO:$DUO_PI05_OPENPI/src:$DUO_PI05_DUOBENCH/src:$DUO_PI05_RCS/python"
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
exec "$DUO_PI05_PYTHON" -m deployment.duo_pi05.supervisor
