#!/bin/bash
set -euo pipefail
exec /venv/main/bin/python -u -m deployment.duo_act.supervisor
