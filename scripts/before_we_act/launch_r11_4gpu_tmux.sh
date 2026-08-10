#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=/workspace/bwa_runs/r11-four-way-v1
REMOTE_REPO=/workspace/fe-pc-wam
WORKTREE_ROOT=/workspace/r11_worktrees
BASE_PYTHON=/venv/robofactory-act/bin/python
SELECTION=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --all) SELECTION=all; shift ;;
    --candidate) SELECTION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --repo) REMOTE_REPO="$2"; shift 2 ;;
    --worktree-root) WORKTREE_ROOT="$2"; shift 2 ;;
    --base-python) BASE_PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SELECTION" ]] || { printf 'use --all or --candidate A|B|C|D\n' >&2; exit 2; }
SELECTION="${SELECTION^^}"
if [[ "$SELECTION" == ALL ]]; then
  SELECTED=(A B C D)
else
  [[ "$SELECTION" =~ ^[A-D]$ ]] || { printf 'invalid candidate: %s\n' "$SELECTION" >&2; exit 2; }
  SELECTED=("$SELECTION")
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
LOCAL_MANIFEST="$ROOT/docs/experiments/r11/run_manifest.json"
PROVENANCE="$RUN_ROOT/preflight/baseline_provenance.json"
PROGRESS="$RUN_ROOT/preflight/baseline_provenance_progress.jsonl"
STAT_VERIFY="$ROOT/scripts/before_we_act/verify_r11_stat_bound_inputs.py"
PREFLIGHT="$ROOT/scripts/before_we_act/preflight_r11_candidate.py"
RECORDER="$ROOT/scripts/before_we_act/record_r11_deployment.py"

declare -A BRANCHES=(
  [A]=feat/r11-vjepa21-ac-refine
  [B]=feat/r11-dreamzero-wan22-wam
  [C]=feat/r11-cosmos-policy-latent
  [D]=feat/r11-lawam-latent-subgoal
)
declare -A WORKTREES=(
  [A]="$WORKTREE_ROOT/a-vjepa"
  [B]="$WORKTREE_ROOT/b-dreamzero"
  [C]="$WORKTREE_ROOT/c-cosmos"
  [D]="$WORKTREE_ROOT/d-lawam"
)
declare -A CONFIGS=(
  [A]=configs/before_we_act/r11/a-vjepa21-ac-refine.json
  [B]=configs/before_we_act/r11/b-dreamzero-wan22-wam.json
  [C]=configs/before_we_act/r11/c-cosmos-policy-latent.json
  [D]=configs/before_we_act/r11/d-lawam-latent-subgoal.json
)
declare -A SESSIONS=(
  [A]=bwa-r11-a-vjepa
  [B]=bwa-r11-b-dreamzero
  [C]=bwa-r11-c-cosmos
  [D]=bwa-r11-d-lawam
)
declare -A UPSTREAMS=(
  [A]=204698b45b3712590f06245fbfba32d3be539812
  [B]=ab790c198fbce33503358efbbd4187ce9a89adf3
  [C]=a2c298b0a3df3778b973fe65e9e58877b292d8a7
  [D]=4ea6fdadce6c9b8746028307a246b79ee2c4fd55
)

for command in git tmux nvidia-smi jq sha256sum find awk grep; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }
done
[[ -x "$BASE_PYTHON" && -d "$REMOTE_REPO/.git" ]] || {
  printf 'remote repository or base Python is missing\n' >&2
  exit 3
}
for path in "$RUN_MANIFEST" "$LOCAL_MANIFEST" "$PROVENANCE" "$PROGRESS" "$STAT_VERIFY" "$PREFLIGHT"; do
  [[ -f "$path" ]] || { printf 'missing frozen preflight path: %s\n' "$path" >&2; exit 3; }
done
[[ "$(sha256sum "$RUN_MANIFEST" | awk '{print $1}')" == "$(sha256sum "$LOCAL_MANIFEST" | awk '{print $1}')" ]] || {
  printf 'local/remote immutable run manifests differ\n' >&2
  exit 3
}
BASE_COMMIT="$(jq -r '.base.commit' "$RUN_MANIFEST")"
[[ "$BASE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'invalid frozen base commit\n' >&2; exit 3; }

git -C "$REMOTE_REPO" fetch --prune --no-tags origin \
  feat/model-improvements feat/r11-four-way-integration \
  "${BRANCHES[A]}" "${BRANCHES[B]}" "${BRANCHES[C]}" "${BRANCHES[D]}"
[[ "$(git -C "$REMOTE_REPO" rev-parse origin/feat/model-improvements)" == "$BASE_COMMIT" ]] || {
  printf 'origin baseline branch moved from immutable R11 base\n' >&2
  exit 3
}

if ((DRY_RUN == 0)); then mkdir -p "$WORKTREE_ROOT"; fi
for candidate in "${SELECTED[@]}"; do
  branch="${BRANCHES[$candidate]}"
  worktree="${WORKTREES[$candidate]}"
  if [[ -e "$worktree/.git" ]]; then
    [[ "$(git -C "$worktree" branch --show-current)" == "$branch" ]] || {
      printf 'existing worktree has wrong branch: %s\n' "$worktree" >&2; exit 3;
    }
    [[ -z "$(git -C "$worktree" status --porcelain)" ]] || {
      printf 'existing candidate worktree is dirty: %s\n' "$worktree" >&2; exit 3;
    }
    if ((DRY_RUN == 0)); then git -C "$worktree" merge --ff-only "origin/$branch"; fi
  else
    if ((DRY_RUN)); then
      printf 'dry-run requires predeployed worktree: %s\n' "$worktree" >&2
      exit 3
    fi
    if git -C "$REMOTE_REPO" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$REMOTE_REPO" worktree add "$worktree" "$branch"
      git -C "$worktree" merge --ff-only "origin/$branch"
    else
      git -C "$REMOTE_REPO" worktree add -b "$branch" "$worktree" "origin/$branch"
    fi
  fi
  commit="$(git -C "$worktree" rev-parse HEAD)"
  [[ "$commit" == "$(git -C "$REMOTE_REPO" rev-parse "origin/$branch")" ]] || {
    printf '%s worktree is not at the pushed branch head\n' "$candidate" >&2; exit 3;
  }
  git -C "$worktree" merge-base --is-ancestor "$BASE_COMMIT" "$commit" || {
    printf '%s does not descend from frozen base\n' "$candidate" >&2; exit 3;
  }
  preflight_temporary="$(mktemp "/tmp/r11-preflight-$candidate-XXXXXX.json")"
  env PYTHONPATH="$worktree" "$BASE_PYTHON" "$worktree/scripts/before_we_act/preflight_r11_candidate.py" \
    --candidate "$candidate" --worktree "$worktree" \
    --config "$worktree/${CONFIGS[$candidate]}" --run-manifest "$RUN_MANIFEST" \
    --skip-assets --output "$preflight_temporary" >/dev/null
  rm -f "$preflight_temporary"
done

env PYTHONPATH="$ROOT" "$BASE_PYTHON" "$STAT_VERIFY" \
  --run-manifest "$RUN_MANIFEST" --baseline-provenance "$PROVENANCE" --progress "$PROGRESS"

AVAILABLE_GIB="$(df -Pk /workspace | awk 'NR==2 {print int($4/1024/1024)}')"
((AVAILABLE_GIB >= 60)) || { printf 'less than 60 GiB free under /workspace\n' >&2; exit 3; }
for candidate in "${SELECTED[@]}"; do
  gpu=$(( $(printf '%d' "'$candidate") - 65 ))
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v index="$gpu" '$1==index {print $2}')"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU %s is already in use; refusing candidate %s\n' "$gpu" "$candidate" >&2
    exit 3
  fi
  if tmux has-session -t "${SESSIONS[$candidate]}" 2>/dev/null; then
    printf 'owned tmux session already exists: %s\n' "${SESSIONS[$candidate]}" >&2
    exit 3
  fi
done

ASSET_PENDING=0
for candidate in "${SELECTED[@]}"; do
  worktree="${WORKTREES[$candidate]}"
  config="$worktree/${CONFIGS[$candidate]}"
  asset_status="$($BASE_PYTHON - "$config" <<'PY'
import json, sys
from pathlib import Path
c = json.load(open(sys.argv[1]))
paths = [Path(v["receipt"]) for v in c.get("assets", {}).values()]
if "foundation" in c:
    paths.append(Path(c["foundation"]["receipt"]))
if not paths or any(not p.is_file() for p in paths):
    print("PENDING_DOWNLOAD")
else:
    try:
        values = [json.load(open(p)).get("status") for p in paths]
    except Exception:
        print("INVALID")
    else:
        print("PASSED" if all(v == "PASSED" for v in values) else "INVALID")
PY
)"
  printf '%s branch=%s commit=%s upstream=%s GPU=%s tmux=%s assets=%s\n' \
    "$candidate" "${BRANCHES[$candidate]}" "$(git -C "$worktree" rev-parse HEAD)" \
    "${UPSTREAMS[$candidate]}" "$(( $(printf '%d' "'$candidate") - 65 ))" \
    "${SESSIONS[$candidate]}" "$asset_status"
  [[ "$asset_status" == PASSED ]] || ASSET_PENDING=1
  if [[ "$asset_status" == PASSED ]]; then
    preflight_temporary="$(mktemp "/tmp/r11-assets-$candidate-XXXXXX.json")"
    env PYTHONPATH="$worktree" "$BASE_PYTHON" \
      "$worktree/scripts/before_we_act/preflight_r11_candidate.py" \
      --candidate "$candidate" --worktree "$worktree" --config "$config" \
      --run-manifest "$RUN_MANIFEST" --output "$preflight_temporary" >/dev/null
    rm -f "$preflight_temporary"
  fi
done
printf 'R11 read-only preflight passed: base/data/seeds/checkpoint/source/branch/GPU/disk\n'
if ((DRY_RUN)); then
  if ((ASSET_PENDING)); then
    printf 'dry-run: one or more pinned foundation bundles still require the candidate download gate\n'
    exit 4
  fi
  printf 'dry-run: all foundation receipts exist; no process/session/run artifact was created\n'
  exit 0
fi

LAUNCHER_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
for candidate in "${SELECTED[@]}"; do
  worktree="${WORKTREES[$candidate]}"
  branch="${BRANCHES[$candidate]}"
  commit="$(git -C "$worktree" rev-parse HEAD)"
  gpu=$(( $(printf '%d' "'$candidate") - 65 ))
  session="${SESSIONS[$candidate]}"
  "$BASE_PYTHON" "$RECORDER" --run-root "$RUN_ROOT" --candidate "$candidate" \
    --branch "$branch" --commit "$commit" --base-commit "$BASE_COMMIT" \
    --upstream-commit "${UPSTREAMS[$candidate]}" --worktree "$worktree" \
    --gpu "$gpu" --tmux "$session" --launcher-commit "$LAUNCHER_COMMIT" >/dev/null
  tmux new-session -d -s "$session" -n pipeline \
    "cd '$worktree' && exec env CUDA_VISIBLE_DEVICES='$gpu' BWA_R11_RUN_ROOT='$RUN_ROOT' BWA_R11_CANDIDATE='$candidate' '$worktree/scripts/before_we_act/run_r11_candidate.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '$gpu' --base-python '$BASE_PYTHON'"
done
printf 'started independent R11 sessions: %s\n' "${SELECTED[*]}"
printf 'monitor: %s/scripts/before_we_act/monitor_r11.sh --all --once\n' "$ROOT"
