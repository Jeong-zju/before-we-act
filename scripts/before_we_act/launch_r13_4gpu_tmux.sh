#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r13-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
ACTION_CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
FULL_INDEX=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2/index.json
CACHE=/workspace/bwa_runs/shared/r13/world_cache_v1.pt
WORKTREE_ROOT=/workspace/bwa_worktrees/r13
BELIEF_SHA=a453f3d0c8ab46b8d0874f74af5856050d5e9b57caaba9416c86fd8fd6f54c49
ACTION_SHA=4c85dcd30058912f4be375af04b65b0f39b365d885883eb29934552b14020e41
BRANCHES=(bwa/r13-p0-tdmpc2-world-component bwa/r13-p1-lpwm-world-component bwa/r13-p2-vjepa2ac-world-component bwa/r13-p3-dinowm-world-component)

while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --belief-checkpoint) BELIEF_CHECKPOINT="$2"; shift 2 ;;
    --action-checkpoint) ACTION_CHECKPOINT="$2"; shift 2 ;;
    --full-index) FULL_INDEX="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid run id\n' >&2; exit 2; }
declare -A ALIAS=( [A]=p0 [B]=p1 [C]=p2 [D]=p3 [a]=p0 [b]=p1 [c]=p2 [d]=p3 )
if [[ "$SELECTION" == all ]]; then
  SELECTED=(p0 p1 p2 p3)
else
  IFS=',' read -r -a RAW <<<"$SELECTION"
  SELECTED=()
  for item in "${RAW[@]}"; do
    item="${ALIAS[$item]:-$item}"
    [[ "$item" =~ ^p[0-3]$ ]] || { printf 'invalid candidate: %s\n' "$item" >&2; exit 2; }
    [[ " ${SELECTED[*]} " != *" $item "* ]] && SELECTED+=("$item")
  done
fi
[[ ${#SELECTED[@]} -gt 0 ]] || { printf 'at least one candidate is required\n' >&2; exit 2; }
for path in "$PYTHON" "$BELIEF_CHECKPOINT" "$ACTION_CHECKPOINT" "$FULL_INDEX"; do
  [[ -e "$path" ]] || { printf 'missing required path: %s\n' "$path" >&2; exit 3; }
done
[[ "$(sha256sum "$BELIEF_CHECKPOINT" | awk '{print $1}')" == "$BELIEF_SHA" ]] || { printf 'frozen W11 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(sha256sum "$ACTION_CHECKPOINT" | awk '{print $1}')" == "$ACTION_SHA" ]] || { printf 'frozen W12 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(git -C "$ROOT" branch --show-current)" == feat/model-improvements ]] || { printf 'launcher must run from feat/model-improvements\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$(git -C "$ROOT" rev-parse origin/feat/model-improvements)" ]] || { printf 'local model-improvements differs from origin\n' >&2; exit 3; }

PARENT_COMMIT=""
for index in 0 1 2 3; do
  branch="${BRANCHES[index]}"
  git -C "$ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote branch %s\n' "$branch" >&2; exit 3; }
  value="$(git -C "$ROOT" show "origin/$branch:configs/before_we_act/r13_world/p$index.yaml" | "$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(sys.stdin)["parent_commit"])')"
  if [[ -z "$PARENT_COMMIT" ]]; then PARENT_COMMIT="$value"; fi
  [[ "$value" == "$PARENT_COMMIT" ]] || { printf 'candidate parent commits differ\n' >&2; exit 3; }
  git -C "$ROOT" merge-base --is-ancestor "$PARENT_COMMIT" "origin/$branch" || { printf '%s does not descend from common R13 parent\n' "$branch" >&2; exit 3; }
done

for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  session="bwa-r13-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v target="$index" '$1==target {print $2}')"
  if [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is in use; refusing %s\n' "$index" "$candidate" >&2
    exit 3
  fi
done
printf 'R13 preflight: run=%s root=%s parent=%s selected=%s\n' "$RUN_ID" "$RUN_ROOT" "$PARENT_COMMIT" "${SELECTED[*]}"
for candidate in "${SELECTED[@]}"; do
  printf '  %s branch=%s GPU=%s session=bwa-r13-%s\n' "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, output, download or tmux session created\n'
  exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT/shared" "$(dirname "$CACHE")" /workspace/.cache/huggingface /workspace/bwa_upstream/r13
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"
  branch="${BRANCHES[index]}"
  worktree="$WORKTREE_ROOT/$candidate"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing worktree %s\n' "$worktree" >&2; exit 3; }
    git -C "$worktree" merge --ff-only "origin/$branch"
  elif git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$ROOT" worktree add "$worktree" "$branch"
    git -C "$worktree" merge --ff-only "origin/$branch"
  else
    git -C "$ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  [[ -f "$worktree/configs/before_we_act/r13_world/$candidate.yaml" && -f "$worktree/experiments/before_we_act/r13/$candidate/component_lock.yaml" ]] || { printf 'missing R13 contract for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$ROOT/scripts/before_we_act/r13_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --parent-commit "$PARENT_COMMIT" --belief-checkpoint "$BELIEF_CHECKPOINT" --action-checkpoint "$ACTION_CHECKPOINT" --cache "$CACHE" --index "$FULL_INDEX" "${WORKTREE_ARGS[@]}"

CACHE_ARGS=(--cache "$CACHE" --belief-sha256 "$BELIEF_SHA" --action-sha256 "$ACTION_SHA" --output "$RUN_ROOT/shared/cache.json")
if [[ -f "$CACHE" ]]; then
  "$PYTHON" "$ROOT/scripts/before_we_act/verify_r13_world_cache.py" "${CACHE_ARGS[@]}"
elif tmux has-session -t bwa-r13-prepare 2>/dev/null; then
  printf 'shared preparation session bwa-r13-prepare already exists; refusing an ambiguous duplicate\n' >&2
  exit 3
else
  PREPARE_LOG="$RUN_ROOT/shared/cache_prepare.log"
  PREPARE_CMD="set -o pipefail; env CUDA_VISIBLE_DEVICES=0 BWA_R13_RUN_ROOT='$RUN_ROOT' BWA_R13_CANDIDATE=prepare PYTHONPATH='$ROOT' '$PYTHON' '$ROOT/scripts/before_we_act/prepare_r13_world_cache.py' --index '$FULL_INDEX' --belief-config '$ROOT/configs/before_we_act/r11_belief/p0.yaml' --belief-checkpoint '$BELIEF_CHECKPOINT' --belief-sha256 '$BELIEF_SHA' --action-config '$ROOT/configs/before_we_act/r12_action/e1_p2.yaml' --action-checkpoint '$ACTION_CHECKPOINT' --action-sha256 '$ACTION_SHA' --output '$CACHE' --heartbeat '$RUN_ROOT/shared/cache_heartbeat.json' --device cuda:0 2>&1 | tee -a '$PREPARE_LOG' && exec '$PYTHON' '$ROOT/scripts/before_we_act/verify_r13_world_cache.py' --cache '$CACHE' --belief-sha256 '$BELIEF_SHA' --action-sha256 '$ACTION_SHA' --output '$RUN_ROOT/shared/cache.json'"
  tmux new-session -d -s bwa-r13-prepare -n cache "cd '$ROOT' && bash -lc \"$PREPARE_CMD\""
fi
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$ROOT/scripts/before_we_act/r13_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r13_candidate.sh --detail "not selected by this launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r13-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R13_RUN_ROOT='$RUN_ROOT' BWA_R13_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r13_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --action-checkpoint '$ACTION_CHECKPOINT' --cache '$CACHE' --python '$PYTHON'"
done
printf 'started R13 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$ROOT/scripts/before_we_act/monitor_r13.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$ROOT/scripts/before_we_act/monitor_r13.sh" "$RUN_ROOT"
printf 'safe stop: %s --run-root %s --candidate all\n' "$ROOT/scripts/before_we_act/stop_r13_4gpu_tmux.sh" "$RUN_ROOT"
