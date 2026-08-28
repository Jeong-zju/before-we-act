#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
cd "$(dirname "$0")/../.."
exec python -u -m deployment.mars_maniflow.supervisor
