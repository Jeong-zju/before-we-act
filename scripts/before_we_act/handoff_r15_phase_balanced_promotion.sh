#!/usr/bin/env bash
set -Eeuo pipefail

VALIDATION_ROOT=/workspace/bwa_worktrees/r15/expert-validation
EXPERT_ROOT=/workspace/bwa_worktrees/r15/expert-evolution
DISCOVERY=/workspace/bwa_runs/r15e30-20260807-phase-balanced-e9-ft5k-discovery20
VALIDATION=/workspace/bwa_runs/r15e31-20260807-phase-balanced-e9-ft5k-validation20
FORMAL=/workspace/bwa_runs/r15e32-20260807-phase-balanced-e9-ft5k-formal20
CHECKPOINT="$DISCOVERY/candidates/p1/train/stack_expert/checkpoints/checkpoint_005000.pt"
EXPERT_INDEX=/workspace/bwa_runs/r15_stack_expert20_cache_20260807-v1-physical/features/index.json
VALIDATION_REFERENCE=/workspace/bwa_runs/r15e7-20260807-w12-validation20-control
DISCOVERY_REFERENCE=/workspace/bwa_runs/r15e1-20260807-discovery20-v2
PYTHON=/venv/robofactory-act/bin/python

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do sleep 20; done
}

wait_for_gpu0() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null || nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; do
    sleep 20
  done
}

acceptance_status() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

launch_e21_search() {
  cd "$EXPERT_ROOT"
  exec ./scripts/before_we_act/launch_r15_expert_finetune_tmux.sh \
    --run-id r15e21-20260807-expert20-e6-ft5k-discovery20 \
    --candidate p2 --gpu-index 0 --session bwa-r15s-expert-e21 \
    --updates 5000 --batch-size 12 --expert-rows 6 --learning-rate 2e-5 \
    --warmup 500 --expert-index "$EXPERT_INDEX" --split discovery20 \
    --reference-run-root "$DISCOVERY_REFERENCE"
}

DISCOVERY_ACCEPTANCE="$DISCOVERY/candidates/p1/acceptance.json"
wait_for_file "$DISCOVERY_ACCEPTANCE"
if [[ "$(acceptance_status "$DISCOVERY_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] e30 discovery failed; launching e21 search\n' "$(date -u +%FT%TZ)"
  wait_for_gpu0 bwa-r15s-phase-e30
  launch_e21_search
fi

printf '[%s] e30 discovery passed; starting independent validation\n' "$(date -u +%FT%TZ)"
wait_for_gpu0 bwa-r15s-phase-e30
cd "$VALIDATION_ROOT"
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate p1 --split validation20 \
  --reference-run-root "$VALIDATION_REFERENCE" --checkpoint "$CHECKPOINT" \
  --gpu-index 0 --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate p1 --split validation20 \
  --reference-run-root "$VALIDATION_REFERENCE" --checkpoint "$CHECKPOINT" \
  --gpu-index 0 --execution-mode act_temporal_ensemble

VALIDATION_ACCEPTANCE="$VALIDATION/candidates/p1/acceptance.json"
wait_for_file "$VALIDATION_ACCEPTANCE"
if [[ "$(acceptance_status "$VALIDATION_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] e30 validation failed; launching e21 search\n' "$(date -u +%FT%TZ)"
  wait_for_gpu0 bwa-r15s-p1
  launch_e21_search
fi

printf '[%s] e30 validation passed; starting original formal Gate20\n' "$(date -u +%FT%TZ)"
wait_for_gpu0 bwa-r15s-p1
cd "$VALIDATION_ROOT"
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate p1 --gpu-index 0 \
  --checkpoint "$CHECKPOINT" --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate p1 --gpu-index 0 \
  --checkpoint "$CHECKPOINT" --execution-mode act_temporal_ensemble

FORMAL_ACCEPTANCE="$FORMAL/candidates/p1/acceptance.json"
wait_for_file "$FORMAL_ACCEPTANCE"
if [[ "$(acceptance_status "$FORMAL_ACCEPTANCE")" == PASSED ]]; then
  printf '[%s] e30 passed discovery, validation, and original Gate20; stopping promotion chain\n' "$(date -u +%FT%TZ)"
  exit 0
fi
printf '[%s] e30 formal Gate20 failed; launching e21 search\n' "$(date -u +%FT%TZ)"
wait_for_gpu0 bwa-r15s-p1
launch_e21_search
