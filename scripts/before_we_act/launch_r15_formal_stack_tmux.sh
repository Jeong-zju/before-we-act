#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r15-formal-$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ROOT=""; CANDIDATE=p1
GPU_INDEX=""; MODE=recent_temporal_ensemble; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
SEED_FILE=/workspace/bwa_runs/shared/r10_gate20/seeds/three_robots_stack_cube.json
W12_GATE=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2/validation/gate20
while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --execution-mode) MODE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$CANDIDATE" =~ ^p[1-3]$ && "$GPU_INDEX" =~ ^[0-3]$ && "$MODE" =~ ^(act_temporal_ensemble|mild_temporal_ensemble|balanced_temporal_ensemble|recent_temporal_ensemble|responsive_temporal_ensemble|cogact_adaptive_ensemble|aac_entropy_chunk|latest_chunk)$ ]] || { printf 'valid run/candidate/GPU/mode required\n' >&2; exit 2; }
for command in git tmux nvidia-smi sha256sum jq; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$BRANCH" =~ ^bwa/r15-(closed-loop-evolution|aac-entropy-chunk)$ && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires a clean R15 evolution branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune; COMMIT="$(git -C "$ROOT" rev-parse HEAD)"; [[ "$COMMIT" == "$(git -C "$ROOT" rev-parse "origin/$BRANCH")" ]] || { printf 'R15 branch differs from origin\n' >&2; exit 3; }
REFERENCE="$W12_GATE/three_robots_stack_cube.json"
PROTECTED_FILES=("$W12_GATE/lift_barrier.json" "$W12_GATE/camera_alignment.json" "$W12_GATE/long_pipeline_delivery.json" "$W12_GATE/take_photo.json")
for path in "$CHECKPOINT" "$SEED_FILE" "$REFERENCE" "${PROTECTED_FILES[@]}"; do [[ -f "$path" ]] || { printf 'missing formal input: %s\n' "$path" >&2; exit 3; }; done
STACK_CONTROL="$(jq -r .successes "$REFERENCE")"; PROTECTED="$(jq -s '[.[].successes] | add' "${PROTECTED_FILES[@]}")"; BASELINE_TOTAL=$((STACK_CONTROL + PROTECTED))
[[ "$(jq -r .episodes "$REFERENCE")" == 20 && "$BASELINE_TOTAL" == 77 ]] || { printf 'frozen W12 formal identity differs\n' >&2; exit 3; }
SEED_SHA="$(sha256sum "$SEED_FILE" | awk '{print $1}')"; REFERENCE_SHA="$(jq -r .seed_protocol.sha256 "$REFERENCE")"; [[ "$SEED_SHA" == "$REFERENCE_SHA" ]] || { printf 'formal Stack seed hash differs\n' >&2; exit 3; }
[[ ! -e "$RUN_ROOT" ]] || { printf 'formal run root exists: %s\n' "$RUN_ROOT" >&2; exit 3; }
SESSION="bwa-r15s-$CANDIDATE"; tmux has-session -t "$SESSION" 2>/dev/null && { printf 'session already exists: %s\n' "$SESSION" >&2; exit 3; }
nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$GPU_INDEX" >&2; exit 3; } || true
printf 'R15 formal preflight run=%s candidate=%s GPU=%s mode=%s baseline=%s protected=%s commit=%s\n' "$RUN_ID" "$CANDIDATE" "$GPU_INDEX" "$MODE" "$BASELINE_TOTAL" "$PROTECTED" "$COMMIT"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi

RUNTIME="$ROOT/scripts/before_we_act/r15_runtime.py"; SPLIT=formal_gate20
P0_WORKTREE=/workspace/bwa_worktrees/r12r4/p2; P0_BRANCH=bwa/r12e1-p2-act-stack-specialist; P0_COMMIT="$(git -C "$P0_WORKTREE" rev-parse HEAD)"
P0_CONFIG="$P0_WORKTREE/configs/before_we_act/r12_action/e1_p2.yaml"; P0_CHECKPOINT=/workspace/bwa_runs/shared/w12/checkpoint_130000.pt
COMMON=(--run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --formal --protected-successes "$PROTECTED" --baseline-total "$BASELINE_TOTAL")
"$PYTHON" "$RUNTIME" register "${COMMON[@]}" --candidate p0 --label w12_control --gpu 0 --worktree "$P0_WORKTREE" --branch "$P0_BRANCH" --commit "$P0_COMMIT" --config "$P0_CONFIG" --checkpoint "$P0_CHECKPOINT" --reference
mkdir -p "$RUN_ROOT/candidates/p0/validation"; ln -s "$REFERENCE" "$RUN_ROOT/candidates/p0/validation/$SPLIT.json"
"$PYTHON" "$RUNTIME" accept --run-root "$RUN_ROOT" --candidate p0
"$PYTHON" "$RUNTIME" status --run-root "$RUN_ROOT" --candidate p0 --state REFERENCE --stage complete --program r15_runtime.py --detail 'frozen W12 formal reference' --pid 0 --child-pid 0 --log "$REFERENCE"
case "$MODE" in mild_temporal_ensemble) LABEL=w12_mild_decay_0p02 ;; balanced_temporal_ensemble) LABEL=w12_balanced_decay_0p05 ;; recent_temporal_ensemble) LABEL=w12_recent_decay_0p10 ;; responsive_temporal_ensemble) LABEL=w12_responsive_decay_0p20 ;; cogact_adaptive_ensemble) LABEL=cogact_adaptive_alpha0p1_h2 ;; aac_entropy_chunk) LABEL=aac_entropy20_h16 ;; latest_chunk) LABEL=w12_latest_chunk ;; *) LABEL=checkpoint_act_temporal_ensemble ;; esac
CONFIG="$ROOT/configs/before_we_act/r12_action/e1_p2.yaml"
"$PYTHON" "$RUNTIME" register "${COMMON[@]}" --candidate "$CANDIDATE" --label "$LABEL" --gpu "$GPU_INDEX" --worktree "$ROOT" --branch "$BRANCH" --commit "$COMMIT" --config "$CONFIG" --checkpoint "$CHECKPOINT"
tmux new-session -d -s "$SESSION" -n formal \
  "cd '$ROOT' && exec env BWA_R15_RUN_ROOT='$RUN_ROOT' BWA_R15_CANDIDATE='$CANDIDATE' '$ROOT/scripts/before_we_act/run_r15_stack_screen.sh' --run-root '$RUN_ROOT' --candidate '$CANDIDATE' --gpu-index '$GPU_INDEX' --python '$PYTHON'"
printf 'started session=%s output=%s\n' "$SESSION" "$RUN_ROOT"
printf 'monitor: %s/scripts/before_we_act/monitor_r15_stack_screens.sh --run-root %s --candidate all --interval 30\n' "$ROOT" "$RUN_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_stack_screens.sh --run-root %s --candidate %s\n' "$ROOT" "$RUN_ROOT" "$CANDIDATE"
