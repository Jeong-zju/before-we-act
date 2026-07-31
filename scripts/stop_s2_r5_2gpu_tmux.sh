#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S2_R5_RUN_ID:-${1:-}}"
if [[ -z "${RUN_ID}" ]]; then
  printf >&2 'usage: %s RUN_ID\n' "$0"; exit 2
fi
RUN_ROOT="${S2_R5_RUN_ROOT:-${FE_ROOT}/outputs/s2_r5_runs/${RUN_ID}}"
if [[ ! -f "${RUN_ROOT}/run_manifest.json" ]]; then
  printf >&2 'Not an S2-R5 run: %s\n' "${RUN_ROOT}"; exit 3
fi
SESSION="$(jq -r '.tmux_session' "${RUN_ROOT}/run_manifest.json")"
PREFIX="$(jq -r '.tmux_window_prefix' "${RUN_ROOT}/run_manifest.json")"
for suffix in prepare p0 p1 monitor; do
  window="${PREFIX}-${suffix}"
  tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null \
    | grep -Fxq "${window}" || continue
  pane_pid="$(tmux list-panes -t "${SESSION}:${window}" -F '#{pane_pid}' | head -1)"
  if [[ -n "${pane_pid}" ]]; then kill -TERM -- "-${pane_pid}" 2>/dev/null || true; fi
  tmux kill-window -t "${SESSION}:${window}" 2>/dev/null || true
done
printf 'Stopped only S2-R5 run %s processes and closed its four windows.\n' "${RUN_ID}"
printf 'Preserved permanent tmux session, shared datasets/cache/artifacts, checkpoints, logs and results.\n'
