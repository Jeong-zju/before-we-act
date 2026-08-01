#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${S3_R6_RUN_ID:-${1:-}}"
if [[ -z "${RUN_ID}" ]]; then printf >&2 'usage: %s RUN_ID\n' "$0"; exit 2; fi
RUN_ROOT="${S3_R6_RUN_ROOT:-${FE_ROOT}/outputs/s3_r6_runs/${RUN_ID}}"
if [[ ! -f "${RUN_ROOT}/run_manifest.json" ]]; then
  printf >&2 'Not an S3-R6 run: %s\n' "${RUN_ROOT}"; exit 3
fi
SESSION="$(jq -r '.tmux_session' "${RUN_ROOT}/run_manifest.json")"
PREFIX="$(jq -r '.tmux_window_prefix' "${RUN_ROOT}/run_manifest.json")"
for suffix in prepare r6l-p0 r6l-p1 r6j-p0 r6j-p1 monitor; do
  window="${PREFIX}-${suffix}"
  tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null \
    | grep -Fxq "${window}" || continue
  pane_pid="$(tmux list-panes -t "${SESSION}:${window}" -F '#{pane_pid}' | head -1)"
  if [[ -n "${pane_pid}" ]]; then kill -TERM -- "-${pane_pid}" 2>/dev/null || true; fi
  tmux kill-window -t "${SESSION}:${window}" 2>/dev/null || true
done
printf 'Stopped only S3-R6 run %s and closed its six windows.\n' "${RUN_ID}"
printf 'Preserved permanent tmux, shared data/cache/parents, checkpoints, resumes, logs and acceptance JSON.\n'
