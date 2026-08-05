#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r12-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
BELIEF_SHA256=a453f3d0c8ab46b8d0874f74af5856050d5e9b57caaba9416c86fd8fd6f54c49
NORMALIZATION_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
NORMALIZATION_SHA256=061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d
PARENT_COMMIT=fdc228189c7fc8556acba9ab9462998ffb967c71
BASE_BRANCH=feat/model-improvements
WORKTREE_ROOT=/workspace/bwa_worktrees/r12r1
DATA_ROOT=/workspace/datasets/robofactory_multitask
R11_CACHE=/workspace/bwa_runs/shared/r11_observation_cache.pt
ACTION_CACHE=/workspace/bwa_runs/shared/r12_causal_coldstart_action_cache.pt
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
BRANCHES=(bwa/r12r1-p0-openpi-causal-coldstart-20k bwa/r12r1-p1-smolvla-causal-coldstart-20k bwa/r12r1-p2-rdt-causal-coldstart-20k bwa/r12r1-p3-consistency-causal-coldstart-20k)

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
for command in git tmux nvidia-smi jq sha256sum find grep awk; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }
done
[[ -x "$PYTHON" && -f "$BELIEF_CHECKPOINT" && -f "$NORMALIZATION_CHECKPOINT" && -f "$R11_CACHE" ]] || { printf 'missing Python, W11/W10 checkpoint, or R11 cache\n' >&2; exit 3; }
[[ "$(sha256sum "$BELIEF_CHECKPOINT" | awk '{print $1}')" == "$BELIEF_SHA256" ]] || { printf 'W11 belief checkpoint hash differs\n' >&2; exit 3; }
[[ "$(sha256sum "$NORMALIZATION_CHECKPOINT" | awk '{print $1}')" == "$NORMALIZATION_SHA256" ]] || { printf 'W10 normalization checkpoint hash differs\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == "$BASE_BRANCH" ]] || { printf 'R12 launcher must run from feat/model-improvements\n' >&2; exit 3; }
[[ -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$FE_ROOT" fetch origin --prune
BASE_HEAD="$(git -C "$FE_ROOT" rev-parse HEAD)"
[[ "$BASE_HEAD" == "$(git -C "$FE_ROOT" rev-parse "origin/$BASE_BRANCH")" ]] || { printf 'local R12 engineering base differs from origin\n' >&2; exit 3; }
git -C "$FE_ROOT" merge-base --is-ancestor "$PARENT_COMMIT" HEAD || { printf 'R12 engineering base does not descend from frozen W11 parent\n' >&2; exit 3; }
for branch in "${BRANCHES[@]}"; do
  git -C "$FE_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote R12 branch %s\n' "$branch" >&2; exit 3; }
done
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  [[ -f "$DATA_ROOT/$task/training_manifest.json" && -f "/workspace/bwa_runs/shared/frozen100/$task.json" && -f "$PROTOCOL_ROOT/seeds/$task.json" ]] || { printf 'missing data/baseline/protocol for %s\n' "$task" >&2; exit 3; }
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  session="bwa-r12r1-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v target="$index" '$1==target {print $2}')"
  if [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is in use; refusing %s\n' "$index" "$candidate" >&2
    exit 3
  fi
done
printf 'R12 preflight: run=%s root=%s selected=%s engineering_base=%s@%s W11_parent=%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}" "$BASE_BRANCH" "$BASE_HEAD" "$PARENT_COMMIT"
for candidate in "${SELECTED[@]}"; do
  printf '  %s branch=%s GPU=%s session=bwa-r12r1-%s\n' "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, cache, artifact, download or tmux session created\n'
  exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface /workspace/bwa_upstream/r12
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"
  branch="${BRANCHES[index]}"
  worktree="$WORKTREE_ROOT/$candidate"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing R12 worktree %s\n' "$worktree" >&2; exit 3; }
    git -C "$worktree" merge --ff-only "origin/$branch"
  elif git -C "$FE_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$FE_ROOT" worktree add "$worktree" "$branch"
    git -C "$worktree" merge --ff-only "origin/$branch"
  else
    git -C "$FE_ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  git -C "$worktree" merge-base --is-ancestor "$PARENT_COMMIT" HEAD || { printf '%s does not descend from W11 parent\n' "$candidate" >&2; exit 3; }
  [[ -f "$worktree/configs/before_we_act/r12_action/$candidate.yaml" && -f "$worktree/experiments/before_we_act/r12/$candidate/component_lock.yaml" ]] || { printf 'missing R12 contract for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --parent-commit "$PARENT_COMMIT" --belief-checkpoint "$BELIEF_CHECKPOINT" --belief-checkpoint-sha256 "$BELIEF_SHA256" --normalization-checkpoint "$NORMALIZATION_CHECKPOINT" "${WORKTREE_ARGS[@]}"

MANIFEST_COUNT="$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type f -name training_manifest.json | wc -l)"
EPISODE_COUNT="$(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'episode_*.hdf5' | wc -l)"
[[ "$MANIFEST_COUNT" == 5 && "$EPISODE_COUNT" == 750 ]] || { printf 'dataset incomplete: %s manifests %s episodes\n' "$MANIFEST_COUNT" "$EPISODE_COUNT" >&2; exit 3; }
if [[ ! -f "$ACTION_CACHE" ]]; then
  if tmux has-session -t bwa-r12r1-prepare 2>/dev/null; then
    printf 'reusing active shared cache preparation session: bwa-r12r1-prepare\n'
  else
    tmux new-session -d -s bwa-r12r1-prepare -n cache \
      "cd '$FE_ROOT' && exec env PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_action_cache.py' --r11-cache '$R11_CACHE' --parent-checkpoint '$NORMALIZATION_CHECKPOINT' --data-root '$DATA_ROOT' --output '$ACTION_CACHE' --state '/workspace/bwa_runs/shared/r12_causal_coldstart_action_cache_state.json' --heartbeat '/workspace/bwa_runs/shared/r12_causal_coldstart_action_cache_heartbeat.json'"
  fi
fi
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r12_candidate.sh --detail "not selected by this launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r12r1-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R12_RUN_ROOT='$RUN_ROOT' BWA_R12_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r12_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --normalization-checkpoint '$NORMALIZATION_CHECKPOINT' --action-cache '$ACTION_CACHE' --protocol-root '$PROTOCOL_ROOT' --data-root '$DATA_ROOT' --python '$PYTHON'"
done
printf 'started R12 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r12.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r12.sh" "$RUN_ROOT"
