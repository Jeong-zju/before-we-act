#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r11-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT=""
SELECTION=all
DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
PARENT_CHECKPOINT=/workspace/bwa_runs/shared/parent/checkpoint_120000.pt
PARENT_COMMIT=06ba780a4617b4aa92b5a103864f0ca28f79aaa6
WORKTREE_ROOT=/workspace/bwa_worktrees/r11
DATA_ROOT=/workspace/datasets/robofactory_multitask
CACHE=/workspace/bwa_runs/shared/r11_observation_cache.pt
BRANCHES=(bwa/r11-p0-vjepa2-component bwa/r11-p1-lpwm-particle-component bwa/r11-p2-dinowm-feature-component bwa/r11-p3-lerobot-vlajepa-component)

while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --parent-checkpoint) PARENT_CHECKPOINT="$2"; shift 2 ;;
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
[[ -x "$PYTHON" && -f "$PARENT_CHECKPOINT" ]] || { printf 'missing Python or parent checkpoint\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == bwa/main ]] || { printf 'launcher must run from bwa/main\n' >&2; exit 3; }
[[ -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'base repository is dirty\n' >&2; exit 3; }
git -C "$FE_ROOT" fetch origin --prune
[[ "$(git -C "$FE_ROOT" rev-parse HEAD)" == "$(git -C "$FE_ROOT" rev-parse origin/bwa/main)" ]] || { printf 'local bwa/main differs from origin/bwa/main\n' >&2; exit 3; }
[[ "$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')" == 061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d ]] || { printf 'frozen W10 checkpoint hash differs\n' >&2; exit 3; }

for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  session="bwa-r11-$candidate"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v target="$index" '$1==target {print $2}')"
  if [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is in use; refusing %s\n' "$index" "$candidate" >&2
    exit 3
  fi
done
printf 'R11 preflight: run=%s root=%s selected=%s\n' "$RUN_ID" "$RUN_ROOT" "${SELECTED[*]}"
for candidate in "${SELECTED[@]}"; do
  printf '  %s branch=%s GPU=%s session=bwa-r11-%s\n' "$candidate" "${BRANCHES[${candidate#p}]}" "${candidate#p}" "$candidate"
done
if ((DRY_RUN)); then
  printf 'dry-run passed; no worktree, artifact, download or tmux session created\n'
  exit 0
fi

mkdir -p "$WORKTREE_ROOT" "$RUN_ROOT" /workspace/.cache/huggingface /workspace/bwa_upstream/r11
WORKTREE_ARGS=()
for index in 0 1 2 3; do
  candidate="p$index"
  branch="${BRANCHES[index]}"
  worktree="$WORKTREE_ROOT/$candidate"
  git -C "$FE_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch" || { printf 'missing remote branch %s\n' "$branch" >&2; exit 3; }
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf 'invalid existing worktree %s\n' "$worktree" >&2; exit 3; }
    git -C "$worktree" merge --ff-only "origin/$branch"
  elif git -C "$FE_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$FE_ROOT" worktree add "$worktree" "$branch"
    git -C "$worktree" merge --ff-only "origin/$branch"
  else
    git -C "$FE_ROOT" worktree add -b "$branch" "$worktree" "origin/$branch"
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  git -C "$worktree" merge-base --is-ancestor "$PARENT_COMMIT" HEAD || { printf '%s does not descend from W10 parent\n' "$candidate" >&2; exit 3; }
  [[ -f "$worktree/configs/before_we_act/r11_belief/$candidate.yaml" && -f "$worktree/experiments/before_we_act/r11/$candidate/component_lock.yaml" ]] || { printf 'missing R11 contract for %s\n' "$candidate" >&2; exit 3; }
  WORKTREE_ARGS+=(--worktree "$candidate=$branch=$commit=$worktree")
done
"$PYTHON" "$FE_ROOT/scripts/before_we_act/r11_runtime.py" init --run-root "$RUN_ROOT" --run-id "$RUN_ID" --parent-commit "$PARENT_COMMIT" --parent-checkpoint "$PARENT_CHECKPOINT" "${WORKTREE_ARGS[@]}"

MANIFEST_COUNT="$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type f -name training_manifest.json 2>/dev/null | wc -l)"
EPISODE_COUNT="$(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l)"
if [[ "$MANIFEST_COUNT" != 5 || "$EPISODE_COUNT" != 750 ]]; then
  printf 'dataset incomplete (%s manifests, %s episodes); starting S0 fixed-revision Hugging Face download anonymously\n' "$MANIFEST_COUNT" "$EPISODE_COUNT"
  "$FE_ROOT/scripts/before_we_act/launch_r10_hf_assets_tmux.sh" --anonymous
fi
if [[ ! -f "$CACHE" ]]; then
  if tmux has-session -t bwa-r11-prepare 2>/dev/null; then
    printf 'reusing active shared cache preparation session: bwa-r11-prepare\n'
  else
    PREPARE_CMD="while [[ \$(find '$DATA_ROOT' -mindepth 2 -maxdepth 2 -type f -name training_manifest.json 2>/dev/null | wc -l) != 5 || \$(find '$DATA_ROOT' -mindepth 3 -maxdepth 3 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l) != 750 ]]; do sleep 30; done; exec '$PYTHON' '$FE_ROOT/scripts/before_we_act/prepare_r11_observation_cache.py' --data-root '$DATA_ROOT' --output '$CACHE'"
    tmux new-session -d -s bwa-r11-prepare -n cache "cd '$FE_ROOT' && $PREPARE_CMD"
  fi
fi
for candidate in p0 p1 p2 p3; do
  if [[ " ${SELECTED[*]} " != *" $candidate "* ]]; then
    "$PYTHON" "$FE_ROOT/scripts/before_we_act/r11_runtime.py" status --run-root "$RUN_ROOT" --candidate "$candidate" --state NOT_STARTED --stage pending --program run_r11_candidate.sh --detail "not selected by this launch" --pid 0 --child-pid 0 --log "$RUN_ROOT/candidates/$candidate/logs/candidate.log"
  fi
done
for candidate in "${SELECTED[@]}"; do
  index="${candidate#p}"
  worktree="$WORKTREE_ROOT/$candidate"
  tmux new-session -d -s "bwa-r11-$candidate" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$index' BWA_R11_RUN_ROOT='$RUN_ROOT' BWA_R11_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r11_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$index' --parent-checkpoint '$PARENT_CHECKPOINT' --cache '$CACHE' --data-root '$DATA_ROOT' --python '$PYTHON'"
done
printf 'started R11 candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s --run-root %s --candidate all --once\n' "$FE_ROOT/scripts/before_we_act/monitor_r11.sh" "$RUN_ROOT"
printf 'monitor live: %s --run-root %s --candidate all --interval 30\n' "$FE_ROOT/scripts/before_we_act/monitor_r11.sh" "$RUN_ROOT"
