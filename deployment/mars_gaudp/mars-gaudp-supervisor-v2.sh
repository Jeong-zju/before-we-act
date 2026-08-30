#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export MARS_GAUDP_REPO="${MARS_GAUDP_REPO:-/workspace/repos/before-we-act}"
export MARS_GAUDP_RUN_ROOT="${MARS_GAUDP_RUN_ROOT:-/workspace/runs/mars_gaudp_fp32_v2}"
export MARS_GAUDP_CACHE_ROOT="${MARS_GAUDP_CACHE_ROOT:-${MARS_GAUDP_RUN_ROOT}/cache}"
export MARS_GAUDP_ROBOFACTORY="${MARS_GAUDP_ROBOFACTORY:-/workspace/repos/RoboFactory}"
export MARS_GAUDP_DATA_ROOT="${MARS_GAUDP_DATA_ROOT:-/workspace/datasets/mars_control}"
export MARS_GAUDP_WEIGHT="${MARS_GAUDP_WEIGHT:-/workspace/repos/Policy-Lightning/weights/re10k.ckpt}"
export MARS_GAUDP_PYTHON="${MARS_GAUDP_PYTHON:-/venv/main/bin/python}"
cd "$MARS_GAUDP_REPO"
exec "$MARS_GAUDP_PYTHON" -m deployment.mars_gaudp.supervisor_v2
