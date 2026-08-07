#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; CACHE_ROOT=""; RAW_HDF5=""; RAW_JSON=""; EPISODES=""; GPU_INDEX=""; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python; SESSION=bwa-r15-expert-cache
while (($#)); do
  case "$1" in
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --raw-hdf5) RAW_HDF5="$2"; shift 2 ;;
    --raw-json) RAW_JSON="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$CACHE_ROOT" && -f "$RAW_HDF5" && -f "$RAW_JSON" && "$EPISODES" =~ ^[1-9][0-9]*$ && "$GPU_INDEX" =~ ^[0-3]$ ]] || { printf 'valid cache/raw/episodes/GPU required\n' >&2; exit 2; }
for command in git tmux nvidia-smi jq df; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
[[ "$(git -C "$ROOT" branch --show-current)" == bwa/r15-closed-loop-evolution && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires clean R15 branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune; [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$(git -C "$ROOT" rev-parse origin/bwa/r15-closed-loop-evolution)" ]] || { printf 'R15 branch differs from origin\n' >&2; exit 3; }
[[ ! -e "$CACHE_ROOT" ]] || { printf 'cache root already exists: %s\n' "$CACHE_ROOT" >&2; exit 3; }
tmux has-session -t "$SESSION" 2>/dev/null && { printf 'session already exists: %s\n' "$SESSION" >&2; exit 3; }
nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$GPU_INDEX" >&2; exit 3; } || true
AVAILABLE_KB="$(df -Pk /workspace | awk 'NR==2 {print $4}')"; [[ "$AVAILABLE_KB" -ge 20971520 ]] || { printf 'expert cache requires at least 20 GiB free\n' >&2; exit 3; }
env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/before_we_act/prepare_r15_expert_full_cache.py" --raw-hdf5 "$RAW_HDF5" --raw-json "$RAW_JSON" --base-index /workspace/bwa_runs/shared/r12r4_native_full_cache_v2/index.json --output-root "$CACHE_ROOT/dry-run" --vision-artifact /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m --action-codec "$ROOT/configs/action_codecs/robofactory_3panda_pd_joint_pos_24d.json" --episodes "$EPISODES" --device cuda:0 --dry-run
printf 'R15 expert cache preflight output=%s episodes=%s GPU=%s free_kB=%s\n' "$CACHE_ROOT" "$EPISODES" "$GPU_INDEX" "$AVAILABLE_KB"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi
tmux new-session -d -s "$SESSION" -n cache \
  "cd '$ROOT' && exec env BWA_R15_EXPERT_CACHE_OUTPUT='$CACHE_ROOT' '$ROOT/scripts/before_we_act/run_r15_expert_cache.sh' --cache-root '$CACHE_ROOT' --raw-hdf5 '$RAW_HDF5' --raw-json '$RAW_JSON' --episodes '$EPISODES' --gpu-index '$GPU_INDEX' --python '$PYTHON'"
printf 'started session=%s output=%s\n' "$SESSION" "$CACHE_ROOT"
printf 'monitor: %s/scripts/before_we_act/monitor_r15_expert_cache.sh --cache-root %s --interval 30\n' "$ROOT" "$CACHE_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_expert_cache.sh --cache-root %s\n' "$ROOT" "$CACHE_ROOT"
