#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION=bwa-r10-hf-assets
RUN_ROOT=/workspace/bwa_runs/shared/r10_hf_assets
ANONYMOUS=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --anonymous) ANONYMOUS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid session\n' >&2; exit 2; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'session already active: %s\n' "$SESSION"
  printf 'monitor: %q --run-root %q --once\n' \
    "$FE_ROOT/scripts/before_we_act/monitor_r10_hf_assets.sh" "$RUN_ROOT"
  exit 3
fi

mkdir -p "$RUN_ROOT"
TOKEN_FIFO="$RUN_ROOT/.hf_token.fifo"
command=(
  "$FE_ROOT/scripts/before_we_act/download_r10_hf_assets.sh"
  --run-root "$RUN_ROOT"
)
((DRY_RUN)) && command+=(--dry-run)
if ((ANONYMOUS)); then
  command+=(--anonymous)
else
  unlink "$TOKEN_FIFO" 2>/dev/null || true
  mkfifo "$TOKEN_FIFO"
  chmod 600 "$TOKEN_FIFO"
  command+=(--token-fifo "$TOKEN_FIFO")
fi
printf -v command_text '%q ' "${command[@]}"
tmux new-session -d -s "$SESSION" "cd '$FE_ROOT' && exec $command_text"

if ((ANONYMOUS == 0)); then
  [[ -r /dev/tty ]] || {
    printf 'interactive terminal required for protected token delivery\n' >&2
    tmux send-keys -t "$SESSION" C-c
    exit 3
  }
  HF_TOKEN_INPUT=""
  IFS= read -r -s -p 'Hugging Face token: ' HF_TOKEN_INPUT </dev/tty
  printf '\n'
  if [[ "$HF_TOKEN_INPUT" != hf_* || "$HF_TOKEN_INPUT" =~ [[:space:]] ]]; then
    HF_TOKEN_INPUT=""
    unset HF_TOKEN_INPUT
    tmux send-keys -t "$SESSION" C-c
    printf 'invalid token input\n' >&2
    exit 3
  fi
  printf '%s\n' "$HF_TOKEN_INPUT" >"$TOKEN_FIFO"
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
fi
printf 'started session=%s run_root=%s mode=%s\n' \
  "$SESSION" "$RUN_ROOT" "$([[ $ANONYMOUS == 1 ]] && printf anonymous || printf protected-token-fifo)"
printf 'monitor once: %s --run-root %s --once\n' \
  "$FE_ROOT/scripts/before_we_act/monitor_r10_hf_assets.sh" "$RUN_ROOT"
