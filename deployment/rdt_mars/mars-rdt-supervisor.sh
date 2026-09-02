#!/usr/bin/env bash
set -Eeuo pipefail
exec /workspace/venvs/rdt/bin/python /workspace/repos/before-we-act/deployment/rdt_mars/supervisor.py
