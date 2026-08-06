#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r12e1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
BELIEF_CHECKPOINT=/workspace/bwa_runs/shared/w11/checkpoint_010000.pt
BELIEF_SHA256=a453f3d0c8ab46b8d0874f74af5856050d5e9b57caaba9416c86fd8fd6f54c49
NORMALIZATION_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
NORMALIZATION_SHA256=061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d
BASE_BRANCH=feat/model-improvements
WORKTREE_ROOT=/workspace/bwa_worktrees/r12r4
DATA_ROOT=/workspace/datasets/robofactory_multitask
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20
FULL_CACHE_ROOT=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2
FULL_INDEX="$FULL_CACHE_ROOT/index.json"
BRANCHES=(bwa/r12e1-p0-openpi-stack-specialist bwa/r12e1-p1-smolvla-stack-specialist bwa/r12e1-p2-act-stack-specialist bwa/r12e1-p3-diffusion-stack-specialist)

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
[[ "$(sha256sum "$NORMALIZATION_CHECKPOINT" | awk '{print $1}')" == "$NORMALIZATION_SHA256" ]] || { printf 'W10 checkpoint hash differs\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == "$BASE_BRANCH" ]] || { printf 'R12-E1 launcher must run from feat/model-improvements\n' >&2; exit 3; }
[[ -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$FE_ROOT" fetch origin --prune
BASE_HEAD="$(git -C "$FE_ROOT" rev-parse HEAD)"
[[ "$BASE_HEAD" == "$(git -C "$FE_ROOT" rev-parse "origin/$BASE_BRANCH")" ]] || { printf 'local R12-E1 engineering base differs from origin\n' >&2; exit 3; }
for branch in "${BRANCHES[@]}"; do
  git -C "$FE_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote R12-E1 branch %s\n' "$branch" >&2; exit 3; }
  git -C "$FE_ROOT" merge-base --is-ancestor "$BASE_HEAD" "origin/$branch" || { printf '%s does not descend from current E1 base\n' "$branch" >&2; exit 3; }
done
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  [[ -f "$DATA_ROOT/$task/training_manifest.json" && -f "/workspace/bwa_runs/shared/frozen100/$task.json" && -f "$PROTOCOL_ROOT/seeds/$task.json" ]] || { printf 'missing data/baseline/protocol for %s\n' "$task" >&2; exit 3; }
done
for candidate in "${SELECTED[@]}"; do
  session="bwa-r12e1-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
done
printf 'R12-E1 preflight: run=%s root=%s selected=%s base=%s@%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}" "$BASE_BRANCH" "$BASE_HEAD"
printf '  policy: native 480x640 multi-view RGB primary; W11+task ID supplemental; exact W10 fallback except Stack\n'
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  usage="$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' || true)"
  printf '  %s branch=%s GPU=%s current_pids=%s session=bwa-r12e1-%s\n' "$candidate" "${BRANCHES[index]}" "$index" "${usage:-none}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, cache, artifact or tmux session created\n'; exit 0
fi

mkdir -p "$RUN_ROOT" /workspace/.cache/huggingface "$FULL_CACHE_ROOT"
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"; branch="${BRANCHES[index]}"; worktree="$WORKTREE_ROOT/$candidate"
  [[ -e "$worktree/.git" ]] || { printf 'missing prepared worktree: %s\n' "$worktree" >&2; exit 3; }
  [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing R12-E1 worktree %s\n' "$worktree" >&2; exit 3; }
  git -C "$worktree" merge --ff-only "origin/$branch"
  commit="$(git -C "$worktree" rev-parse HEAD)"
  [[ -f "$worktree/configs/before_we_act/r12_action/e1_$candidate.yaml" ]] || { printf 'missing R12-E1 config for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --round R12-E1 --session-prefix bwa-r12e1 --formal-updates 130000 --shared-spatial-cache "$FULL_CACHE_ROOT" --protocol-variant r12_e1_native_image_task_film_exact_w10_fallback_v1 --parent-commit "$BASE_HEAD" --belief-checkpoint "$BELIEF_CHECKPOINT" --belief-checkpoint-sha256 "$BELIEF_SHA256" --normalization-checkpoint "$NORMALIZATION_CHECKPOINT" "${WORKTREE_ARGS[@]}"

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
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r12_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r12_evolution_candidate.sh --detail "not selected by launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"; worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r12e1-$candidate" -n pipeline \
    "cd '$worktree' && exec env BWA_R12_RUN_ROOT='$RUN_ROOT' BWA_R12_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r12_evolution_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --belief-checkpoint '$BELIEF_CHECKPOINT' --normalization-checkpoint '$NORMALIZATION_CHECKPOINT' --full-index '$FULL_INDEX' --vision-artifact '$VISION_ARTIFACT' --protocol-root '$PROTOCOL_ROOT' --python '$PYTHON'"
done
printf 'started R12-E1 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r12_evolution.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r12_evolution.sh" "$RUN_ROOT"
printf 'safe stop: %s --run-root %s --candidate all\n' "$FE_ROOT/scripts/before_we_act/stop_r12_evolution_4gpu_tmux.sh" "$RUN_ROOT"
