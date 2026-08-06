#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r12r4-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
BELIEF_SHA256=a453f3d0c8ab46b8d0874f74af5856050d5e9b57caaba9416c86fd8fd6f54c49
NORMALIZATION_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
NORMALIZATION_SHA256=061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d
W11_PARENT=fdc228189c7fc8556acba9ab9462998ffb967c71
BASE_BRANCH=feat/model-improvements
WORKTREE_ROOT=/workspace/bwa_worktrees/r12r4
DATA_ROOT=/workspace/datasets/robofactory_multitask
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
FULL_CACHE_ROOT=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2
FULL_INDEX="$FULL_CACHE_ROOT/index.json"
BRANCHES=(bwa/r12r4-p0-openpi-full-query-130k bwa/r12r4-p1-smolvla-full-query-130k bwa/r12r4-p2-act-plan-prior-full-130k bwa/r12r4-p3-diffusion-full-query-130k)

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
for command in git tmux nvidia-smi jq sha256sum find grep awk df; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }
done
[[ -x "$PYTHON" && -f "$BELIEF_CHECKPOINT" && -f "$NORMALIZATION_CHECKPOINT" && -f "$VISION_ARTIFACT/config.json" && -f "$VISION_ARTIFACT/model.safetensors" ]] || { printf 'missing Python, W11/W10 checkpoint, or DINOv3 artifact\n' >&2; exit 3; }
[[ "$(sha256sum "$BELIEF_CHECKPOINT" | awk '{print $1}')" == "$BELIEF_SHA256" ]] || { printf 'W11 belief checkpoint hash differs\n' >&2; exit 3; }
[[ "$(sha256sum "$NORMALIZATION_CHECKPOINT" | awk '{print $1}')" == "$NORMALIZATION_SHA256" ]] || { printf 'W10 normalization checkpoint hash differs\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == "$BASE_BRANCH" ]] || { printf 'R12-R4 launcher must run from feat/model-improvements\n' >&2; exit 3; }
[[ -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$FE_ROOT" fetch origin --prune
BASE_HEAD="$(git -C "$FE_ROOT" rev-parse HEAD)"
[[ "$BASE_HEAD" == "$(git -C "$FE_ROOT" rev-parse "origin/$BASE_BRANCH")" ]] || { printf 'local R12-R4 engineering base differs from origin\n' >&2; exit 3; }
git -C "$FE_ROOT" merge-base --is-ancestor "$W11_PARENT" HEAD || { printf 'engineering base does not descend from W11 parent\n' >&2; exit 3; }
for branch in "${BRANCHES[@]}"; do
  git -C "$FE_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote R12-R4 branch %s\n' "$branch" >&2; exit 3; }
  git -C "$FE_ROOT" merge-base --is-ancestor "$BASE_HEAD" "origin/$branch" || { printf '%s does not descend from current R4 engineering base\n' "$branch" >&2; exit 3; }
done
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  [[ -f "$DATA_ROOT/$task/training_manifest.json" && -f "/workspace/bwa_runs/shared/frozen100/$task.json" && -f "$PROTOCOL_ROOT/seeds/$task.json" ]] || { printf 'missing data/baseline/protocol for %s\n' "$task" >&2; exit 3; }
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  session="bwa-r12r4-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v target="$index" '$1==target {print $2}')"
  if [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is in use; refusing %s\n' "$index" "$candidate" >&2; exit 3
  fi
done
available_kb="$(df -Pk /workspace | awk 'NR==2 {print $4}')"
if [[ ! -f "$FULL_INDEX" && "$available_kb" -lt 104857600 ]]; then
  printf 'native-resolution feature cache requires at least 100 GiB free under /workspace; available_kB=%s\n' "$available_kb" >&2; exit 3
fi
printf 'R12-R4 preflight: run=%s root=%s selected=%s base=%s@%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}" "$BASE_BRANCH" "$BASE_HEAD"
printf '  visual contract: every native 480x640 RGB -> frozen DINO 30x40 -> post-encoder 6x8 cache\n'
for candidate in "${SELECTED[@]}"; do
  printf '  %s branch=%s GPU=%s session=bwa-r12r4-%s\n' "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, cache, artifact or tmux session created\n'; exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface /workspace/bwa_upstream/r12r4 "$FULL_CACHE_ROOT"
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"; branch="${BRANCHES[index]}"; worktree="$WORKTREE_ROOT/$candidate"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing R12-R4 worktree %s\n' "$worktree" >&2; exit 3; }
    git -C "$worktree" merge --ff-only "origin/$branch"
  elif git -C "$FE_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$FE_ROOT" worktree add "$worktree" "$branch"
    git -C "$worktree" merge --ff-only "origin/$branch"
  else
    git -C "$FE_ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  [[ -f "$worktree/configs/before_we_act/r12_action/$candidate.yaml" && -f "$worktree/experiments/before_we_act/r12/$candidate/component_lock.yaml" ]] || { printf 'missing R12-R4 contract for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --round R12-R4 --session-prefix bwa-r12r4 --formal-updates 130000 --shared-spatial-cache "$FULL_CACHE_ROOT" --protocol-variant r12_full_episode_native_480x640_dinov3_30x40_to_6x8_v2 --parent-commit "$BASE_HEAD" --belief-checkpoint "$BELIEF_CHECKPOINT" --belief-checkpoint-sha256 "$BELIEF_SHA256" --normalization-checkpoint "$NORMALIZATION_CHECKPOINT" "${WORKTREE_ARGS[@]}"

MANIFEST_COUNT="$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type f -name training_manifest.json | wc -l)"
EPISODE_COUNT="$(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'episode_*.hdf5' | wc -l)"
[[ "$MANIFEST_COUNT" == 5 && "$EPISODE_COUNT" == 750 ]] || { printf 'dataset incomplete: %s manifests %s episodes\n' "$MANIFEST_COUNT" "$EPISODE_COUNT" >&2; exit 3; }
if [[ ! -f "$FULL_INDEX" ]]; then
  for index in 0 1 2 3; do
    session="bwa-r12r4-cache-rank$index"
    if ! tmux has-session -t "$session" 2>/dev/null; then
      tmux new-session -d -s "$session" -n cache \
        "cd '$FE_ROOT' && exec env CUDA_VISIBLE_DEVICES='$index' PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_full_episode_cache.py' --mode shard --rank '$index' --world-size 4 --data-root '$DATA_ROOT' --vision-artifact '$VISION_ARTIFACT' --output-root '$FULL_CACHE_ROOT' --state '$FULL_CACHE_ROOT/rank_${index}_state.json' --heartbeat '$FULL_CACHE_ROOT/rank_${index}_heartbeat.json' --frame-batch-size 1 --image-batch-size 5 --device cuda:0"
    fi
  done
  if ! tmux has-session -t bwa-r12r4-cache-index 2>/dev/null; then
    tmux new-session -d -s bwa-r12r4-cache-index -n index \
      "cd '$FE_ROOT' && exec env PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_full_episode_cache.py' --mode index --world-size 4 --output-root '$FULL_CACHE_ROOT' --index '$FULL_INDEX' --state '$FULL_CACHE_ROOT/index_state.json' --heartbeat '$FULL_CACHE_ROOT/index_heartbeat.json'"
  fi
fi
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r12_r4_candidate.sh --detail "not selected by this launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"; worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r12r4-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R12_RUN_ROOT='$RUN_ROOT' BWA_R12_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r12_r4_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --normalization-checkpoint '$NORMALIZATION_CHECKPOINT' --full-index '$FULL_INDEX' --vision-artifact '$VISION_ARTIFACT' --protocol-root '$PROTOCOL_ROOT' --data-root '$DATA_ROOT' --python '$PYTHON'"
done
printf 'started R12-R4 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r12_r4.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r12_r4.sh" "$RUN_ROOT"
