#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ROOT=""; CANDIDATE=all; ONCE=0; INTERVAL=30; PYTHON=/venv/robofactory-act/bin/python
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
[[ -n "$RUN_ROOT" ]] || { printf '%s\n' '--run-root is required' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=(monitor --run-root "$RUN_ROOT" --candidate "$CANDIDATE" --interval "$INTERVAL")
((ONCE)) && ARGS+=(--once)
exec "$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" "${ARGS[@]}"
