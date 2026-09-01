#!/usr/bin/env bash
set -Eeuo pipefail
exec /venv/main/bin/python -u -m deployment.duo_dp.training_probe_supervisor
