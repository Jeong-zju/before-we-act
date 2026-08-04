#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_HOST=""
SOURCE_PORT=""
DEST_ROOT=/workspace
DRY_RUN=0
EXPECTED_CHECKPOINT=061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d

while (($#)); do
  case "$1" in
    --source-host) SOURCE_HOST="$2"; shift 2 ;;
    --source-port) SOURCE_PORT="$2"; shift 2 ;;
    --dest-root) DEST_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$SOURCE_HOST" || -z "$SOURCE_PORT" ]]; then
  printf 'usage: %s --source-host user@host --source-port PORT [--dest-root PATH] [--dry-run]\n' "$0" >&2
  exit 2
fi

RUN_ROOT="$DEST_ROOT/bwa_runs/shared/s10_asset_sync"
STATE="$RUN_ROOT/state.json"
HEARTBEAT="$RUN_ROOT/heartbeat"
LOG="$RUN_ROOT/sync.log"
KNOWN_HOSTS="$RUN_ROOT/known_hosts"
mkdir -p "$RUN_ROOT" "$DEST_ROOT/datasets/robofactory_multitask" \
  "$DEST_ROOT/artifacts/dinov3-vitb16-pretrain-lvd1689m" \
  "$DEST_ROOT/bwa_runs/shared/parent"
touch "$KNOWN_HOSTS"
chmod 600 "$KNOWN_HOSTS"

STARTED_AT="$(date -u +%FT%TZ)"
write_state() {
  local status="$1"
  local stage="$2"
  local detail="$3"
  local temporary="$STATE.tmp"
  printf '{"schema_version":1,"experiment":"r10-shared-assets","status":"%s","stage":"%s","detail":"%s","pid":%d,"started_at":"%s","updated_at":"%s","log":"%s"}\n' \
    "$status" "$stage" "$detail" "$$" "$STARTED_AT" "$(date -u +%FT%TZ)" "$LOG" >"$temporary"
  mv "$temporary" "$STATE"
}
heartbeat() {
  while kill -0 "$$" 2>/dev/null; do
    touch "$HEARTBEAT"
    sleep 20
  done
}
heartbeat &
HEARTBEAT_PID=$!
cleanup() {
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'write_state FAILED asset_sync "line_${LINENO}_exit_${?}"' ERR

SSH_COMMAND="ssh -p $SOURCE_PORT -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=accept-new"
RSYNC_OPTIONS=(--archive --partial --append-verify --human-readable --info=progress2)
if ((DRY_RUN)); then
  RSYNC_OPTIONS+=(--dry-run)
fi

run_sync() {
  local stage="$1"
  local source="$2"
  local destination="$3"
  write_state DOWNLOADING "$stage" "$source"
  printf '[%s] stage=%s source=%s destination=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$source" "$destination" | tee -a "$LOG"
  rsync "${RSYNC_OPTIONS[@]}" -e "$SSH_COMMAND" \
    "$SOURCE_HOST:$source" "$destination" 2>&1 | tee -a "$LOG"
}

write_state PREPARING asset_sync preflight
run_sync parent_checkpoint \
  /workspace/runs/no_wrist_stereo_core_120k/checkpoint_120000.pt \
  "$DEST_ROOT/bwa_runs/shared/parent/"
run_sync parent_metadata \
  /workspace/runs/no_wrist_stereo_core_120k/config.json \
  "$DEST_ROOT/bwa_runs/shared/parent/"
run_sync parent_normalization \
  /workspace/runs/no_wrist_stereo_core_120k/normalization.pt \
  "$DEST_ROOT/bwa_runs/shared/parent/"
run_sync baseline_frozen100 \
  /workspace/runs/no_wrist_stereo_core_120k/frozen100/ \
  "$DEST_ROOT/bwa_runs/shared/frozen100/"
run_sync dino \
  /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m/ \
  "$DEST_ROOT/artifacts/dinov3-vitb16-pretrain-lvd1689m/"
run_sync robofactory \
  /workspace/RoboFactory/ \
  "$DEST_ROOT/RoboFactory/"

if ((!DRY_RUN)); then
  actual_checkpoint="$(sha256sum "$DEST_ROOT/bwa_runs/shared/parent/checkpoint_120000.pt" | awk '{print $1}')"
  if [[ "$actual_checkpoint" != "$EXPECTED_CHECKPOINT" ]]; then
    printf 'parent checkpoint hash mismatch: %s\n' "$actual_checkpoint" >&2
    exit 3
  fi
fi

write_state DOWNLOADING five_task_dataset parallel_five_tasks
printf '[%s] stage=five_task_dataset mode=parallel-five-task\n' \
  "$(date -u +%FT%TZ)" | tee -a "$LOG"
DATASET_PIDS=()
for task in \
  lift_barrier \
  camera_alignment \
  three_robots_stack_cube \
  long_pipeline_delivery \
  take_photo; do
  mkdir -p "$DEST_ROOT/datasets/robofactory_multitask/$task"
  rsync "${RSYNC_OPTIONS[@]}" -e "$SSH_COMMAND" \
    "$SOURCE_HOST:/workspace/datasets/robofactory_multitask/$task/" \
    "$DEST_ROOT/datasets/robofactory_multitask/$task/" \
    >"$RUN_ROOT/${task}.log" 2>&1 &
  DATASET_PIDS+=("$!")
done
for pid in "${DATASET_PIDS[@]}"; do
  wait "$pid"
done
printf '[%s] five-task parallel transfer complete\n' \
  "$(date -u +%FT%TZ)" | tee -a "$LOG"
if ((DRY_RUN)); then
  write_state STOPPED asset_sync dry_run_complete
else
  write_state PASSED asset_sync complete
fi
