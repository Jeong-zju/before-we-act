#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r11-four-way-v1
PYTHON=/venv/robofactory-act/bin/python
SELECTION=""
MODE=""
INTERVAL=5
while (($#)); do
  case "$1" in
    --all) SELECTION=all; shift ;;
    --candidate) SELECTION="${2^^}"; shift 2 ;;
    --once) MODE=once; shift ;;
    --watch) MODE=watch; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$SELECTION" == all || "$SELECTION" =~ ^[A-D]$ ]] || {
  printf 'use --all or --candidate A|B|C|D\n' >&2; exit 2;
}
[[ "$MODE" == once || "$MODE" == watch ]] || {
  printf 'use --once or --watch\n' >&2; exit 2;
}
ARGUMENTS=(
  "$ROOT/scripts/before_we_act/r11_runtime.py" monitor
  --run-root "$RUN_ROOT" --candidate "$SELECTION" --interval "$INTERVAL"
)
[[ "$MODE" == once ]] && ARGUMENTS+=(--once)
exec "$PYTHON" "${ARGUMENTS[@]}"
