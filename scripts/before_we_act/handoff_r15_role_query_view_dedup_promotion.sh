#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/bwa_worktrees/r15/role-query-view-dedup
DISCOVERY=/workspace/bwa_runs/r15e54-20260808-role-query-view-dedup-e9-ft5k-discovery20
VALIDATION=/workspace/bwa_runs/r15e55-20260808-role-query-view-dedup-e9-ft5k-validation20
FORMAL=/workspace/bwa_runs/r15e56-20260808-role-query-view-dedup-e9-ft5k-formal20
DISCOVERY_REFERENCE=/workspace/bwa_runs/r15e1-20260807-discovery20-v2
VALIDATION_REFERENCE=/workspace/bwa_runs/r15e7-20260807-w12-validation20-control
EXPERT_INDEX=/workspace/bwa_runs/r15_stack_expert20_cache_20260807-v1-physical/features/index.json
PHASE_MANIFEST=/workspace/bwa_runs/shared/r15_stack_phase_manifest_v1.json
PYTHON=/venv/robofactory-act/bin/python
GPU=3
CANDIDATE=p3
PREDECESSOR=bwa-r15-handoff-role-query-e51
UPSTREAM_PRODUCER=bwa-r15s-expert-e22

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
  [[ ! -e "$path" ]] || { printf 'role-query view-dedup promotion root already exists: %s\n' "$path" >&2; exit 3; }
done

printf '[%s] waiting for predecessor %s before exact-view-dedup ablation\n' \
  "$(date -u +%FT%TZ)" "$PREDECESSOR"
while tmux has-session -t "$PREDECESSOR" 2>/dev/null; do sleep 20; done
while tmux has-session -t "$UPSTREAM_PRODUCER" 2>/dev/null; do sleep 20; done
wait_for_gpu bwa-r15s-dedup-e54
cd "$ROOT"
./scripts/before_we_act/launch_r15_expert_finetune_tmux.sh \
  --run-id "$(basename "$DISCOVERY")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --session bwa-r15s-dedup-e54 --updates 5000 \
  --batch-size 12 --expert-rows 9 --learning-rate 2e-5 --warmup 500 \
  --expert-index "$EXPERT_INDEX" --phase-manifest "$PHASE_MANIFEST" \
  --split discovery20 --reference-run-root "$DISCOVERY_REFERENCE" --dry-run
./scripts/before_we_act/launch_r15_expert_finetune_tmux.sh \
  --run-id "$(basename "$DISCOVERY")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --session bwa-r15s-dedup-e54 --updates 5000 \
  --batch-size 12 --expert-rows 9 --learning-rate 2e-5 --warmup 500 \
  --expert-index "$EXPERT_INDEX" --phase-manifest "$PHASE_MANIFEST" \
  --split discovery20 --reference-run-root "$DISCOVERY_REFERENCE"

DISCOVERY_ACCEPTANCE="$DISCOVERY/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$DISCOVERY_ACCEPTANCE"
if [[ "$(acceptance_status "$DISCOVERY_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] role-query view-dedup discovery did not strictly beat W12; promotion stops\n' "$(date -u +%FT%TZ)"
  exit 0
fi

CHECKPOINT="$DISCOVERY/candidates/$CANDIDATE/train/stack_expert/checkpoints/checkpoint_005000.pt"
printf '[%s] role-query view-dedup discovery passed; starting independent validation\n' "$(date -u +%FT%TZ)"
wait_for_gpu bwa-r15s-dedup-e54
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate "$CANDIDATE" \
  --split validation20 --reference-run-root "$VALIDATION_REFERENCE" \
  --checkpoint "$CHECKPOINT" --gpu-index "$GPU" \
  --session bwa-r15s-dedup-e55 --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_temporal_screens_tmux.sh \
  --run-id "$(basename "$VALIDATION")" --candidate "$CANDIDATE" \
  --split validation20 --reference-run-root "$VALIDATION_REFERENCE" \
  --checkpoint "$CHECKPOINT" --gpu-index "$GPU" \
  --session bwa-r15s-dedup-e55 --execution-mode act_temporal_ensemble

VALIDATION_ACCEPTANCE="$VALIDATION/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$VALIDATION_ACCEPTANCE"
if [[ "$(acceptance_status "$VALIDATION_ACCEPTANCE")" != PASSED ]]; then
  printf '[%s] role-query view-dedup validation did not strictly beat W12; promotion stops\n' "$(date -u +%FT%TZ)"
  exit 0
fi

printf '[%s] role-query view-dedup validation passed; starting original formal Gate20\n' "$(date -u +%FT%TZ)"
wait_for_gpu bwa-r15s-dedup-e55
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --checkpoint "$CHECKPOINT" \
  --session bwa-r15s-dedup-e56 --execution-mode act_temporal_ensemble --dry-run
./scripts/before_we_act/launch_r15_formal_stack_tmux.sh \
  --run-id "$(basename "$FORMAL")" --candidate "$CANDIDATE" \
  --gpu-index "$GPU" --checkpoint "$CHECKPOINT" \
  --session bwa-r15s-dedup-e56 --execution-mode act_temporal_ensemble

FORMAL_ACCEPTANCE="$FORMAL/candidates/$CANDIDATE/acceptance.json"
wait_for_file "$FORMAL_ACCEPTANCE"
printf '[%s] role-query view-dedup formal status=%s\n' \
  "$(date -u +%FT%TZ)" "$(acceptance_status "$FORMAL_ACCEPTANCE")"
