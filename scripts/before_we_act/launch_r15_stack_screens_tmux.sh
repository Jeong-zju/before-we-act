#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r15-screen-$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ROOT=""; SELECTION="p0,p1"; SPLIT=discovery20; GPU_OVERRIDE=""; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r15_stack_protocol_v1
while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --gpu-index) GPU_OVERRIDE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$SPLIT" =~ ^(discovery20|validation20|reserve20|final20)$ ]] || { printf 'invalid run id or split\n' >&2; exit 2; }
IFS=',' read -r -a SELECTED <<<"$SELECTION"
for candidate in "${SELECTED[@]}"; do [[ "$candidate" =~ ^p[0-2]$ ]] || { printf 'available screen candidates are p0,p1,p2\n' >&2; exit 2; }; done
if [[ -n "$GPU_OVERRIDE" ]]; then
  [[ "${#SELECTED[@]}" -eq 1 && "$GPU_OVERRIDE" =~ ^[0-3]$ ]] || { printf -- '--gpu-index requires one candidate and GPU 0..3\n' >&2; exit 2; }
fi
for command in git tmux nvidia-smi sha256sum; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
[[ "$(git -C "$ROOT" branch --show-current)" == bwa/r15-closed-loop-evolution && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires clean R15 orchestration branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$(git -C "$ROOT" rev-parse origin/bwa/r15-closed-loop-evolution)" ]] || { printf 'R15 orchestration branch differs from origin\n' >&2; exit 3; }
SEED_FILE="$PROTOCOL_ROOT/$SPLIT.json"
[[ -f "$SEED_FILE" ]] || { printf 'missing frozen R15 split: %s\n' "$SEED_FILE" >&2; exit 3; }
SEED_SHA="$(sha256sum "$SEED_FILE" | awk '{print $1}')"

declare -A LABEL GPU WORKTREE BRANCH CONFIG CHECKPOINT REFERENCE
LABEL[p0]=w12_control; GPU[p0]=0; WORKTREE[p0]=/workspace/bwa_worktrees/r12r4/p2; BRANCH[p0]=bwa/r12e1-p2-act-stack-specialist; CONFIG[p0]="${WORKTREE[p0]}/configs/before_we_act/r12_action/e1_p2.yaml"; CHECKPOINT[p0]=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2/train/formal/checkpoints/checkpoint_130000.pt; REFERENCE[p0]=1
LABEL[p1]=e2_p0_causal_phase; GPU[p1]=1; WORKTREE[p1]=/workspace/bwa_worktrees/r12e2/p0; BRANCH[p1]=bwa/r12e2-p0-causal-phase-stack-specialist; CONFIG[p1]="${WORKTREE[p1]}/configs/before_we_act/r12_action/e1_p0.yaml"; CHECKPOINT[p1]=/workspace/bwa_runs/r12e2-20260806-causal-phase-p0-v2/candidates/p0/train/formal/checkpoints/checkpoint_phase_030000.pt; REFERENCE[p1]=0
LABEL[p2]=e3_p2_act_causal_phase; GPU[p2]=3; WORKTREE[p2]=/workspace/bwa_worktrees/r12e3/p2; BRANCH[p2]=bwa/r12e3-p2-act-causal-phase-stack-specialist; CONFIG[p2]="${WORKTREE[p2]}/configs/before_we_act/r12_action/e1_p2.yaml"; CHECKPOINT[p2]=/workspace/bwa_runs/r15e0-20260807-causal-phase-p2-resume/candidates/p2/train/formal/checkpoints/checkpoint_phase_030000.pt; REFERENCE[p2]=0
[[ -n "$GPU_OVERRIDE" ]] && GPU["${SELECTED[0]}"]="$GPU_OVERRIDE"

for candidate in "${SELECTED[@]}"; do
  worktree="${WORKTREE[$candidate]}"; branch="${BRANCH[$candidate]}"; session="bwa-r15s-$candidate"; gpu="${GPU[$candidate]}"
  for path in "$worktree/.git" "${CONFIG[$candidate]}" "${CHECKPOINT[$candidate]}"; do [[ -e "$path" ]] || { printf 'missing %s input: %s\n' "$candidate" "$path" >&2; exit 3; }; done
  [[ "$(git -C "$worktree" branch --show-current)" == "$branch" && -z "$(git -C "$worktree" status --porcelain)" ]] || { printf '%s worktree is not clean/expected branch\n' "$candidate" >&2; exit 3; }
  commit="$(git -C "$worktree" rev-parse HEAD)"; [[ "$commit" == "$(git -C "$worktree" rev-parse "origin/$branch")" ]] || { printf '%s differs from origin\n' "$candidate" >&2; exit 3; }
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$gpu" >&2; exit 3; } || true
  printf '%s label=%s branch=%s commit=%s GPU=%s checkpoint=%s\n' "$candidate" "${LABEL[$candidate]}" "$branch" "$commit" "$gpu" "${CHECKPOINT[$candidate]}"
done
printf 'R15 screen preflight run=%s split=%s seeds=%s root=%s\n' "$RUN_ID" "$SPLIT" "$SEED_SHA" "$RUN_ROOT"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi
mkdir -p "$RUN_ROOT"
for candidate in "${SELECTED[@]}"; do
  worktree="${WORKTREE[$candidate]}"; branch="${BRANCH[$candidate]}"; commit="$(git -C "$worktree" rev-parse HEAD)"
  REGISTER=(register --run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --candidate "$candidate" --label "${LABEL[$candidate]}" --gpu "${GPU[$candidate]}" --worktree "$worktree" --branch "$branch" --commit "$commit" --config "${CONFIG[$candidate]}" --checkpoint "${CHECKPOINT[$candidate]}")
  ((REFERENCE[$candidate])) && REGISTER+=(--reference)
  "$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" "${REGISTER[@]}"
done
for candidate in "${SELECTED[@]}"; do
  tmux new-session -d -s "bwa-r15s-$candidate" -n screen \
    "cd '$ROOT' && exec env BWA_R15_RUN_ROOT='$RUN_ROOT' BWA_R15_CANDIDATE='$candidate' '$ROOT/scripts/before_we_act/run_r15_stack_screen.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '${GPU[$candidate]}' --python '$PYTHON'"
done
printf 'started R15 screen candidates: %s\n' "${SELECTED[*]}"
printf 'monitor once: %s/scripts/before_we_act/monitor_r15_stack_screens.sh --run-root %s --candidate all --once\n' "$ROOT" "$RUN_ROOT"
printf 'monitor live: %s/scripts/before_we_act/monitor_r15_stack_screens.sh --run-root %s --candidate all --interval 30\n' "$ROOT" "$RUN_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_stack_screens.sh --run-root %s --candidate all\n' "$ROOT" "$RUN_ROOT"
