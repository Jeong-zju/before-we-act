#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r12r3-$(date -u +%Y%m%dT%H%M%SZ)"
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
WORKTREE_ROOT=/workspace/bwa_worktrees/r12r3
DATA_ROOT=/workspace/datasets/robofactory_multitask
R11_CACHE=/workspace/bwa_runs/shared/r11_observation_cache.pt
ACTION_CACHE=/workspace/bwa_runs/shared/r12_dense_causal_history_action_cache_v2.pt
SPATIAL_CACHE=/workspace/bwa_runs/shared/r12r3_dinov3_spatial_cache_v1.pt
SPATIAL_SHARDS=/workspace/bwa_runs/shared/r12r3_dinov3_spatial_cache_v1_shards
RECOVERY_CACHE=/workspace/bwa_runs/shared/r12r3_on_policy_recovery_cache_v1.pt
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
BRANCHES=(bwa/r12r3-p0-openpi-spatial-fusion-60k bwa/r12r3-p1-smolvla-spatial-fusion-60k bwa/r12r3-p2-act-spatial-fusion-60k bwa/r12r3-p3-diffusion-spatial-fusion-60k)

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
[[ -x "$PYTHON" && -f "$BELIEF_CHECKPOINT" && -f "$NORMALIZATION_CHECKPOINT" && -f "$R11_CACHE" && -f "$VISION_ARTIFACT/config.json" && -f "$VISION_ARTIFACT/model.safetensors" ]] || { printf 'missing Python, W11/W10 checkpoint, R11 cache, or DINOv3 artifact\n' >&2; exit 3; }
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
  session="bwa-r12r3-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v target="$index" '$1==target {print $2}')"
  if [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is in use; refusing %s\n' "$index" "$candidate" >&2
    exit 3
  fi
done
printf 'R12 preflight: run=%s root=%s selected=%s engineering_base=%s@%s W11_parent=%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}" "$BASE_BRANCH" "$BASE_HEAD" "$PARENT_COMMIT"
for candidate in "${SELECTED[@]}"; do
  printf '  %s branch=%s GPU=%s session=bwa-r12r3-%s\n' "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, cache, artifact, download or tmux session created\n'
  exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface /workspace/bwa_upstream/r12r3 "$SPATIAL_SHARDS"
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
  if tmux has-session -t bwa-r12r2-prepare 2>/dev/null; then
    printf 'reusing active shared cache preparation session: bwa-r12r2-prepare\n'
  else
    tmux new-session -d -s bwa-r12r2-prepare -n cache \
      "cd '$FE_ROOT' && exec env PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_action_cache.py' --r11-cache '$R11_CACHE' --parent-checkpoint '$NORMALIZATION_CHECKPOINT' --data-root '$DATA_ROOT' --output '$ACTION_CACHE' --state '/workspace/bwa_runs/shared/r12_dense_causal_history_action_cache_v2_state.json' --heartbeat '/workspace/bwa_runs/shared/r12_dense_causal_history_action_cache_v2_heartbeat.json'"
  fi
fi
if [[ ! -f "$SPATIAL_CACHE" ]]; then
  for index in 0 1 2 3; do
    session="bwa-r12r3-spatial-rank$index"
    if ! tmux has-session -t "$session" 2>/dev/null; then
      tmux new-session -d -s "$session" -n spatial \
        "while [[ ! -f '$ACTION_CACHE' ]]; do sleep 10; done; cd '$FE_ROOT' && exec env CUDA_VISIBLE_DEVICES='$index' PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_spatial_cache.py' --mode shard --rank '$index' --world-size 4 --action-cache '$ACTION_CACHE' --data-root '$DATA_ROOT' --vision-artifact '$VISION_ARTIFACT' --shard-dir '$SPATIAL_SHARDS' --state '$SPATIAL_SHARDS/rank_${index}_state.json' --heartbeat '$SPATIAL_SHARDS/rank_${index}_heartbeat.json' --device cuda:0"
    fi
  done
  if ! tmux has-session -t bwa-r12r3-spatial-consolidate 2>/dev/null; then
    tmux new-session -d -s bwa-r12r3-spatial-consolidate -n consolidate \
      "while [[ ! -f '$ACTION_CACHE' ]]; do sleep 10; done; cd '$FE_ROOT' && exec env PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r12_spatial_cache.py' --mode consolidate --world-size 4 --action-cache '$ACTION_CACHE' --shard-dir '$SPATIAL_SHARDS' --output '$SPATIAL_CACHE' --state '/workspace/bwa_runs/shared/r12r3_dinov3_spatial_cache_v1_state.json' --heartbeat '/workspace/bwa_runs/shared/r12r3_dinov3_spatial_cache_v1_heartbeat.json'"
  fi
fi
PROBE="$RUN_ROOT/representation_sufficiency.json"
if [[ ! -f "$PROBE" ]] && ! tmux has-session -t bwa-r12r3-representation-probe 2>/dev/null; then
  tmux new-session -d -s bwa-r12r3-representation-probe -n probe \
    "while [[ ! -f '$SPATIAL_CACHE' ]]; do sleep 10; done; cd '$FE_ROOT' && exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/probe_r12_representation.py' --action-cache '$ACTION_CACHE' --spatial-cache '$SPATIAL_CACHE' --belief-config '$FE_ROOT/configs/before_we_act/r11_belief/p0.yaml' --belief-checkpoint '$BELIEF_CHECKPOINT' --output '$PROBE' --device cuda:0"
fi
RECOVERY_ROOT="$RUN_ROOT/recovery"
RECOVERY_SEEDS="$RECOVERY_ROOT/training_seeds.json"
RECOVERY_RECEIPT="$RECOVERY_ROOT/recovery_receipt.json"
mkdir -p "$RECOVERY_ROOT"
"$PYTHON" "$FE_ROOT/scripts/before_we_act/prepare_r12_recovery_seeds.py" \
  --gate20-root "$PROTOCOL_ROOT" --output "$RECOVERY_SEEDS" --per-task 4
if [[ ! -f "$RECOVERY_CACHE" ]]; then
  for candidate in p0 p2; do
    index="${candidate#p}"
    session="bwa-r12r3-recovery-$candidate"
    checkpoint="$RUN_ROOT/unused"
    if [[ "$candidate" == p0 ]]; then
      checkpoint=/workspace/bwa_runs/r12r2-20260805-dense-history-act-dp/candidates/p0/train/formal/checkpoints/checkpoint_120000.pt
    else
      checkpoint=/workspace/bwa_runs/r12r2-20260805-dense-history-act-dp/candidates/p2/train/formal/checkpoints/checkpoint_120000.pt
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      tmux new-session -d -s "$session" -n recovery \
        "while [[ ! -f '$PROBE' ]]; do sleep 10; done; '$PYTHON' -c 'import json; assert json.load(open(\"$PROBE\"))[\"passed\"]'; cd '$WORKTREE_ROOT/$candidate' && exec env CUDA_VISIBLE_DEVICES='$index' PYTHONPATH='$WORKTREE_ROOT/$candidate' '$PYTHON' '$WORKTREE_ROOT/$candidate/scripts/before_we_act/collect_r12_recovery.py' --candidate '$candidate' --student-config '$WORKTREE_ROOT/$candidate/configs/before_we_act/r12_action/$candidate.yaml' --student-checkpoint '$checkpoint' --belief-config '$WORKTREE_ROOT/$candidate/configs/before_we_act/r11_belief/p0.yaml' --belief-checkpoint '$BELIEF_CHECKPOINT' --teacher-checkpoint '$NORMALIZATION_CHECKPOINT' --vision-artifact '$VISION_ARTIFACT' --seed-manifest '$RECOVERY_SEEDS' --output '$RECOVERY_ROOT/${candidate}_recovery.pt' --state '$RECOVERY_ROOT/${candidate}_state.json' --heartbeat '$RECOVERY_ROOT/${candidate}_heartbeat.json' --device cuda:0"
    fi
  done
  if ! tmux has-session -t bwa-r12r3-recovery-consolidate 2>/dev/null; then
    tmux new-session -d -s bwa-r12r3-recovery-consolidate -n consolidate \
      "while [[ ! -f '$RECOVERY_ROOT/p0_recovery.pt' || ! -f '$RECOVERY_ROOT/p2_recovery.pt' ]]; do sleep 10; done; cd '$FE_ROOT' && exec env PYTHONPATH='$FE_ROOT' '$PYTHON' '$FE_ROOT/scripts/before_we_act/consolidate_r12_recovery.py' --p0 '$RECOVERY_ROOT/p0_recovery.pt' --p2 '$RECOVERY_ROOT/p2_recovery.pt' --seed-manifest '$RECOVERY_SEEDS' --output '$RECOVERY_CACHE' --receipt '$RECOVERY_RECEIPT'"
  fi
elif [[ ! -f "$RECOVERY_RECEIPT" ]]; then
  printf 'shared recovery cache exists but this run has no verified receipt; refusing implicit reuse\n' >&2
  exit 3
fi
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r12_candidate.sh --detail "not selected by this launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r12r3-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R12_RUN_ROOT='$RUN_ROOT' BWA_R12_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r12_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --normalization-checkpoint '$NORMALIZATION_CHECKPOINT' --action-cache '$ACTION_CACHE' --spatial-cache '$SPATIAL_CACHE' --recovery-cache '$RECOVERY_CACHE' --recovery-receipt '$RECOVERY_RECEIPT' --vision-artifact '$VISION_ARTIFACT' --representation-probe '$PROBE' --protocol-root '$PROTOCOL_ROOT' --data-root '$DATA_ROOT' --python '$PYTHON'"
done
printf 'started R12 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r12.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r12.sh" "$RUN_ROOT"
