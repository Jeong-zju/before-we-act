#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r12-current
CANDIDATE=all
ONCE=0
INTERVAL=30
PYTHON=/venv/robofactory-act/bin/python
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
ARGS=(monitor --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --interval "$INTERVAL")
((ONCE)) && ARGS+=(--once)
exec "$PYTHON" "$ROOT/scripts/before_we_act/r12_runtime.py" "${ARGS[@]}"
