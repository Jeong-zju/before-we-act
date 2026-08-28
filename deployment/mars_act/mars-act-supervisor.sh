#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
exec /venv/main/bin/python /workspace/repos/before-we-act/deployment/mars_act/supervisor.py
