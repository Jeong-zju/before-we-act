#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r14-$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ROOT=""; SELECTION=all; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
ACTION_CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
WORLD_CHECKPOINT=/workspace/bwa_runs/shared/w13/checkpoint_010000.pt
BELIEF_SHA=a453f3d0c8ab46b8d0874f74af5856050d5e9b57caaba9416c86fd8fd6f54c49
ACTION_SHA=4c85dcd30058912f4be375af04b65b0f39b365d885883eb29934552b14020e41
WORLD_SHA=6f98120d087d0f93969c697b2a041d338bd9e235adf136a690bb10689cb19b64
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
W12_RUN=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2
WORKTREE_ROOT=/workspace/bwa_worktrees/r14
BRANCHES=(bwa/r14-p0-worldinworld-revision-component bwa/r14-p1-dinowm-cem-component bwa/r14-p2-tdmpc2-mpc-component bwa/r14-p3-mbrllib-optimizer-component)
while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'invalid run id\n' >&2; exit 2; }
declare -A ALIAS=( [A]=p0 [B]=p1 [C]=p2 [D]=p3 [a]=p0 [b]=p1 [c]=p2 [d]=p3 )
if [[ "$SELECTION" == all ]]; then SELECTED=(p0 p1 p2 p3)
else
  IFS=',' read -r -a RAW <<<"$SELECTION"; SELECTED=()
  for item in "${RAW[@]}"; do
    item="${ALIAS[$item]:-$item}"; [[ "$item" =~ ^p[0-3]$ ]] || { printf 'invalid candidate: %s\n' "$item" >&2; exit 2; }
    [[ " ${SELECTED[*]} " != *" $item "* ]] && SELECTED+=("$item")
  done
fi
for command in git tmux nvidia-smi jq sha256sum; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
for path in "$PYTHON" "$BELIEF_CHECKPOINT" "$ACTION_CHECKPOINT" "$WORLD_CHECKPOINT" "$VISION_ARTIFACT/config.json" "$VISION_ARTIFACT/model.safetensors" "$W12_RUN/acceptance.json"; do [[ -e "$path" ]] || { printf 'missing R14 input: %s\n' "$path" >&2; exit 3; }; done
[[ "$(sha256sum "$BELIEF_CHECKPOINT" | awk '{print $1}')" == "$BELIEF_SHA" ]] || { printf 'W11 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(sha256sum "$ACTION_CHECKPOINT" | awk '{print $1}')" == "$ACTION_SHA" ]] || { printf 'W12 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(sha256sum "$WORLD_CHECKPOINT" | awk '{print $1}')" == "$WORLD_SHA" ]] || { printf 'W13 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(git -C "$ROOT" branch --show-current)" == feat/model-improvements && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires clean feat/model-improvements\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
BASE_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$BASE_HEAD" == "$(git -C "$ROOT" rev-parse origin/feat/model-improvements)" ]] || { printf 'local base differs from origin\n' >&2; exit 3; }
for index in 0 1 2 3; do
  branch="${BRANCHES[index]}"; git -C "$ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote branch %s\n' "$branch" >&2; exit 3; }
  value="$(git -C "$ROOT" show "origin/$branch:configs/before_we_act/r14_decision/p$index.yaml" | "$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(sys.stdin)["parent_commit"])')"
  [[ "$value" == "$BASE_HEAD" ]] || { printf '%s has wrong parent %s\n' "$branch" "$value" >&2; exit 3; }
done
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  [[ -f "$PROTOCOL_ROOT/seeds/$task.json" && -f "$W12_RUN/validation/gate20/$task.json" ]] || { printf 'missing paired seed/W12 report for %s\n' "$task" >&2; exit 3; }
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"; session="bwa-r14-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$index" >&2; exit 3; } || true
done
printf 'R14 preflight run=%s root=%s parent=%s selected=%s baseline=W12-77/100\n' "$RUN_ID" "$RUN_ROOT" "$BASE_HEAD" "${SELECTED[*]}"
for candidate in "${SELECTED[@]}"; do index="${candidate#p}"; printf '  %s branch=%s GPU=%s session=bwa-r14-%s\n' "$candidate" "${BRANCHES[index]}" "$index" "$candidate"; done
if ((DRY_RUN)); then printf 'dry-run passed; no worktree/output/tmux created\n'; exit 0; fi
mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface /workspace/bwa_upstream/r14
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"; branch="${BRANCHES[index]}"; worktree="$WORKTREE_ROOT/$candidate"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing worktree %s\n' "$worktree" >&2; exit 3; }
    git -C "$worktree" merge --ff-only "origin/$branch"
  elif git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch"; then git -C "$ROOT" worktree add "$worktree" "$branch"; git -C "$worktree" merge --ff-only "origin/$branch"
  else git -C "$ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"; fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  [[ -f "$worktree/configs/before_we_act/r14_decision/$candidate.yaml" && -f "$worktree/experiments/before_we_act/r14/$candidate/component_lock.yaml" ]] || { printf 'missing R14 contract for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$ROOT/scripts/before_we_act/r14_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --parent-commit "$BASE_HEAD" --belief-checkpoint "$BELIEF_CHECKPOINT" --action-checkpoint "$ACTION_CHECKPOINT" --world-checkpoint "$WORLD_CHECKPOINT" "${WORKTREE_ARGS[@]}"
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$ROOT/scripts/before_we_act/r14_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r14_candidate.sh --detail "not selected by launch" --pid 0 --child-pid 0 --total-steps 100 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"; worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r14-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R14_RUN_ROOT='$RUN_ROOT' BWA_R14_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r14_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --action-checkpoint '$ACTION_CHECKPOINT' --world-checkpoint '$WORLD_CHECKPOINT' --vision-artifact '$VISION_ARTIFACT' --protocol-root '$PROTOCOL_ROOT' --w12-run '$W12_RUN' --python '$PYTHON'"
done
printf 'started R14 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$ROOT/scripts/before_we_act/monitor_r14.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$ROOT/scripts/before_we_act/monitor_r14.sh" "$RUN_ROOT"
printf 'safe stop: %s --run-root %s --candidate all\n' "$ROOT/scripts/before_we_act/stop_r14_4gpu_tmux.sh" "$RUN_ROOT"
