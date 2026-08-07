#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; OUTPUT_ROOT=""; EPISODES=1; START_SEED=5000; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python; SESSION=bwa-r15-expert-collect; ROBOFACTORY=/workspace/RoboFactory
while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT_ROOT" && "$EPISODES" =~ ^[1-9][0-9]*$ && "$EPISODES" -le 50 && "$START_SEED" =~ ^[1-9][0-9]*$ ]] || { printf 'output required; episodes must be 1..50\n' >&2; exit 2; }
for command in git tmux df; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
[[ "$(git -C "$ROOT" branch --show-current)" == bwa/r15-closed-loop-evolution && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires clean R15 orchestration branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$(git -C "$ROOT" rev-parse origin/bwa/r15-closed-loop-evolution)" ]] || { printf 'R15 orchestration differs from origin\n' >&2; exit 3; }
[[ "$(git -C "$ROBOFACTORY" rev-parse HEAD)" == 5868242322414a91454e22f1dd9641f613ba1bcf && -z "$(git -C "$ROBOFACTORY" status --porcelain)" ]] || { printf 'RoboFactory source identity differs\n' >&2; exit 3; }
[[ "$START_SEED" -gt 3149 && "$START_SEED" -lt 1000000000 ]] || { printf 'start seed overlaps original demonstrations or frozen eval range\n' >&2; exit 3; }
tmux has-session -t "$SESSION" 2>/dev/null && { printf 'session already exists: %s\n' "$SESSION" >&2; exit 3; }
[[ ! -e "$OUTPUT_ROOT/status.json" ]] || { printf 'output already has a status file: %s\n' "$OUTPUT_ROOT" >&2; exit 3; }
AVAILABLE_KB="$(df -Pk /workspace | awk 'NR==2 {print $4}')"; [[ "$AVAILABLE_KB" -ge 20971520 ]] || { printf 'expert smoke requires at least 20 GiB free\n' >&2; exit 3; }
printf 'R15 expert preflight output=%s episodes=%s seed=%s source=%s free_kB=%s\n' "$OUTPUT_ROOT" "$EPISODES" "$START_SEED" "$(git -C "$ROBOFACTORY" rev-parse HEAD)" "$AVAILABLE_KB"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi
mkdir -p "$OUTPUT_ROOT"
tmux new-session -d -s "$SESSION" -n collect \
  "cd '$ROOT' && exec env BWA_R15_EXPERT_OUTPUT='$OUTPUT_ROOT' '$ROOT/scripts/before_we_act/run_r15_expert_collection.sh' --output-root '$OUTPUT_ROOT' --episodes '$EPISODES' --start-seed '$START_SEED' --python '$PYTHON'"
printf 'started session=%s output=%s\n' "$SESSION" "$OUTPUT_ROOT"
printf 'monitor: %s/scripts/before_we_act/monitor_r15_expert_collection.sh --output-root %s --interval 30\n' "$ROOT" "$OUTPUT_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_expert_collection.sh --output-root %s\n' "$ROOT" "$OUTPUT_ROOT"
