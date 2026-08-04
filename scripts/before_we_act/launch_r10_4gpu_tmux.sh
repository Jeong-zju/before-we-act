#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r10-20260804"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
WORKTREE_ROOT=/workspace/bwa-r10-worktrees
PARENT_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r10_gate20

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
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { printf 'invalid run id\n' >&2; exit 2; }
[[ -n "$RUN_ROOT" ]] || RUN_ROOT="/workspace/bwa_runs/$RUN_ID"

declare -A ALIAS=( [A]=p0 [B]=p1 [C]=p2 [D]=p3 [a]=p0 [b]=p1 [c]=p2 [d]=p3 )
if [[ "$SELECTION" == all ]]; then
  SELECTED=(p0 p1 p2 p3)
else
  IFS=',' read -r -a RAW <<<"$SELECTION"
  SELECTED=()
  for item in "${RAW[@]}"; do
    item="${ALIAS[$item]:-$item}"
    [[ "$item" =~ ^p[0-3]$ ]] || { printf 'invalid candidate: %s\n' "$item" >&2; exit 2; }
    [[ " ${SELECTED[*]} " == *" $item "* ]] || SELECTED+=("$item")
  done
fi
BRANCHES=(
  bwa/r10-p0-calibrated-crossview
  bwa/r10-p1-object-slots
  bwa/r10-p2-recurrent-predictive-state
  bwa/r10-p3-jepa-future-feature
)

for command in git tmux nvidia-smi jq sha256sum find grep awk sed; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }
done
[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 3; }
[[ -f "$PARENT_CHECKPOINT" ]] || { printf 'missing parent checkpoint\n' >&2; exit 3; }
[[ "$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')" == 061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d ]] || {
  printf 'parent checkpoint hash mismatch\n' >&2; exit 3;
}
[[ "$(git -C "$FE_ROOT" branch --show-current)" == bwa/r9-core-native ]] || {
  printf 'launcher base repository must be on bwa/r9-core-native\n' >&2; exit 3;
}
[[ -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$FE_ROOT" fetch --no-tags origin bwa/r9-core-native "${BRANCHES[@]}"
for branch in "${BRANCHES[@]}"; do
  git -C "$FE_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || {
    printf 'missing remote branch: %s\n' "$branch" >&2; exit 3;
  }
done
ASSET_STATUS="$(jq -r '.status // "UNKNOWN"' /workspace/bwa_runs/shared/r10_hf_assets/state.json 2>/dev/null || true)"
if [[ "$ASSET_STATUS" != PASSED ]]; then
  printf 'shared S10 Hugging Face assets are not PASSED: %s\n' "$ASSET_STATUS" >&2
  printf 'start/resume with scripts/before_we_act/launch_r10_hf_assets_tmux.sh\n' >&2
  exit 3
fi
for task in lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo; do
  [[ -f "/workspace/datasets/robofactory_multitask/$task/training_manifest.json" ]] || {
    printf 'missing training manifest for %s\n' "$task" >&2; exit 3;
  }
  [[ -f "/workspace/bwa_runs/shared/frozen100/$task.json" ]] || {
    printf 'missing frozen100 baseline for %s\n' "$task" >&2; exit 3;
  }
done

for candidate in "${SELECTED[@]}"; do
  gpu="${candidate#p}"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q .; then
    uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v index="$gpu" '$1==index {print $2}')"
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
      printf 'GPU %s is already in use; refusing candidate %s\n' "$gpu" "$candidate" >&2
      exit 3
    fi
  fi
  if tmux has-session -t "bwa-r10-$candidate" 2>/dev/null; then
    printf 'tmux session already exists: bwa-r10-%s\n' "$candidate" >&2
    exit 3
  fi
done

printf 'R10 launch preflight passed\n'
printf 'run=%s root=%s selected=%s parent=%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}" "$(git -C "$FE_ROOT" rev-parse HEAD)"
for candidate in "${SELECTED[@]}"; do
  printf '%s branch=%s GPU=%s session=bwa-r10-%s\n' \
    "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run: no worktree, run artifact or tmux session created\n'
  exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface
WORKTREE_ARGS=()
PARENT_COMMIT=""
for index in 0 1 2 3; do
  candidate="p$index"
  branch="${BRANCHES[index]}"
  worktree="$WORKTREE_ROOT/$candidate"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" ]] || {
      printf 'existing worktree has wrong branch: %s\n' "$worktree" >&2; exit 3;
    }
    [[ -z "$(git -C "$worktree" status --porcelain)" ]] || {
      printf 'candidate worktree is dirty: %s\n' "$worktree" >&2; exit 3;
    }
    git -C "$worktree" merge --ff-only "origin/$branch"
  else
    if git -C "$FE_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$FE_ROOT" worktree add "$worktree" "$branch"
      git -C "$worktree" merge --ff-only "origin/$branch"
    else
      git -C "$FE_ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"
    fi
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  config="$worktree/configs/before_we_act/r10_perception/$candidate.yaml"
  card="$worktree/experiments/before_we_act/r10/$candidate/implementation_card.yaml"
  [[ -f "$config" && -f "$card" ]] || { printf 'candidate config/card missing: %s\n' "$candidate" >&2; exit 3; }
  candidate_parent="$($PYTHON - "$config" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["parent_commit"])
PY
)"
  if [[ -z "$PARENT_COMMIT" ]]; then PARENT_COMMIT="$candidate_parent"; fi
  [[ "$candidate_parent" == "$PARENT_COMMIT" ]] || { printf 'candidate parent drift\n' >&2; exit 3; }
  git -C "$worktree" merge-base --is-ancestor "$PARENT_COMMIT" HEAD || {
    printf '%s does not descend from frozen parent\n' "$candidate" >&2; exit 3;
  }
  "$PYTHON" "$worktree/scripts/before_we_act/validate_implementation_card.py" \
    "$card" --expected-parent "$PARENT_COMMIT" >/dev/null
  (
    cd "$worktree"
    "$PYTHON" scripts/before_we_act/audit_candidate_diff.py \
      --card "$card" --parent "$PARENT_COMMIT" --head HEAD
  ) >/dev/null
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done

"$PYTHON" "$FE_ROOT/scripts/before_we_act/prepare_r10_gate20.py" \
  --baseline-dir /workspace/bwa_runs/shared/frozen100 --output "$PROTOCOL_ROOT"
"$PYTHON" "$FE_ROOT/scripts/before_we_act/r10_runtime.py" init \
  --run-root "$RUN_ROOT" --run-id "$RUN_ID" --parent-commit "$PARENT_COMMIT" \
  --parent-checkpoint "$PARENT_CHECKPOINT" "${WORKTREE_ARGS[@]}"

for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r10_runtime.py" status \
      --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED \
      --stage pending --program run_r10_candidate.sh --detail "not selected by this launch" \
      --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r10-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R10_RUN_ROOT='$RUN_ROOT' BWA_R10_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r10_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --parent-checkpoint '$PARENT_CHECKPOINT' --protocol-root '$PROTOCOL_ROOT' --python '$PYTHON'"
done
printf 'started: %s\n' "${SELECTED[*]}"
printf 'monitor once: %q %q monitor --run-root %q --candidate all --once\n' \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/r10_runtime.py" "$RUN_ROOT"
printf 'monitor live: %q %q monitor --run-root %q --candidate all\n' \
  "$PYTHON" "$FE_ROOT/scripts/before_we_act/r10_runtime.py" "$RUN_ROOT"
