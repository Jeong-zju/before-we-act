#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -f /workspace/.env ]]; then
  set -a
  . /workspace/.env
  set +a
fi
export DUO_DINO_REPO=/workspace/repos/care-v2-final
export DUO_DINO_DUOBENCH_REPO=/workspace/repos/duobench
export DUO_DINO_DATASET=/workspace/datasets/duobench
export DUO_DINO_RUN=/workspace/runs/duobench-care-dino-v2-gripper
export DUO_DINO_MODEL=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
export DUO_DINO_PYTHON=/venv/main/bin/python
export PYTHONPATH=/workspace/repos/care-v2-final:/workspace/repos/care-v2-final/deployment:/workspace/repos/duobench/src:/workspace/repos/robot-control-stack/python
export MUJOCO_GL=egl
export DUOBENCH_PREFIX=/workspace/repos/duobench
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/mujoco:${LD_LIBRARY_PATH:-}
exec "${DUO_DINO_PYTHON}" -u -m deployment.duo_dino_reference.supervisor run
