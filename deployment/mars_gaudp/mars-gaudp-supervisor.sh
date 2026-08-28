#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
exec /venv/main/bin/python -m deployment.mars_gaudp.supervisor
