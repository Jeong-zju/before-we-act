#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/bwa_worktrees/r15/role-query-specialist
DISCOVERY=/workspace/bwa_runs/r15e51-20260808-role-query-phase-e9-ft5k-discovery20
VALIDATION=/workspace/bwa_runs/r15e52-20260808-role-query-phase-e9-ft5k-validation20
FORMAL=/workspace/bwa_runs/r15e53-20260808-role-query-phase-e9-ft5k-formal20
DISCOVERY_REFERENCE=/workspace/bwa_runs/r15e1-20260807-discovery20-v2
VALIDATION_REFERENCE=/workspace/bwa_runs/r15e7-20260807-w12-validation20-control
EXPERT_INDEX=/workspace/bwa_runs/r15_stack_expert20_cache_20260807-v1-physical/features/index.json
PHASE_MANIFEST=/workspace/bwa_runs/shared/r15_stack_phase_manifest_v1.json
PYTHON=/venv/robofactory-act/bin/python
GPU=3
CANDIDATE=p3

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do sleep 20; done
}

wait_for_gpu() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null || nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; do
    sleep 20
  done
}

acceptance_status() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

for path in "$DISCOVERY" "$VALIDATION" "$FORMAL"; do
  [[ ! -e "$path" ]] || { printf 'role-query promotion root already exists: %s\n' "$path" >&2; exit 3; }
done

printf '[%s] waiting for GPU%s before role-query phase-balanced continuation\n' "$(date -u +%FT%TZ)" "$GPU"
wait_for_gpu bwa-r15s-role-e51
cd "$ROOT"
./scripts/before_we_act/launch_r15_expert_finetune_tmux.sh \
  --run-id "$(basename "$DISCOVERY")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --session bwa-r15s-role-e51 --updates 5000 \
  --batch-size 12 --expert-rows 9 --learning-rate 2e-5 --warmup 500 \
  --expert-index "$EXPERT_INDEX" --phase-manifest "$PHASE_MANIFEST" \
  --split discovery20 --reference-run-root "$DISCOVERY_REFERENCE" --dry-run
./scripts/before_we_act/launch_r15_expert_finetune_tmux.sh \
  --run-id "$(basename "$DISCOVERY")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --session bwa-r15s-role-e51 --updates 5000 \
  --batch-size 12 --expert-rows 9 --learning-rate 2e-5 --warmup 500 \
  --expert-index "$EXPERT_INDEX" --phase-manifest "$PHASE_MANIFEST" \
  --split discovery20 --reference-run-root "$DISCOVERY_REFERENCE"

DISCOVERY_ACCEPTANCE="$DISCOVERY/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$DISCOVERY_ACCEPTANCE"
if [[ "$(acceptance_status "$DISCOVERY_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] role-query discovery did not strictly beat W12; promotion stops\n' "$(date -u +%FT%TZ)"
  exit 0
fi

CHECKPOINT="$DISCOVERY/candidates/$CANDIDATE/train/stack_expert/checkpoints/checkpoint_005000.pt"
printf '[%s] role-query discovery passed; starting independent validation\n' "$(date -u +%FT%TZ)"
wait_for_gpu bwa-r15s-role-e51
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate "$CANDIDATE" \
  --split validation20 --reference-run-root "$VALIDATION_REFERENCE" \
  --checkpoint "$CHECKPOINT" --gpu-index "$GPU" \
  --session bwa-r15s-role-e52 --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate "$CANDIDATE" \
  --split validation20 --reference-run-root "$VALIDATION_REFERENCE" \
  --checkpoint "$CHECKPOINT" --gpu-index "$GPU" \
  --session bwa-r15s-role-e52 --execution-mode act_temporal_ensemble

VALIDATION_ACCEPTANCE="$VALIDATION/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$VALIDATION_ACCEPTANCE"
if [[ "$(acceptance_status "$VALIDATION_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] role-query validation did not strictly beat W12; promotion stops\n' "$(date -u +%FT%TZ)"
  exit 0
fi

printf '[%s] role-query validation passed; starting original formal Gate20\n' "$(date -u +%FT%TZ)"
wait_for_gpu bwa-r15s-role-e52
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --checkpoint "$CHECKPOINT" \
  --session bwa-r15s-role-e53 --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --checkpoint "$CHECKPOINT" \
  --session bwa-r15s-role-e53 --execution-mode act_temporal_ensemble

FORMAL_ACCEPTANCE="$FORMAL/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$FORMAL_ACCEPTANCE"
printf '[%s] role-query formal status=%s\n' \
  "$(date -u +%FT%TZ)" "$(acceptance_status "$FORMAL_ACCEPTANCE")"
