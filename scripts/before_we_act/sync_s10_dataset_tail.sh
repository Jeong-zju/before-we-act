#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_HOST=""
SOURCE_PORT=""
DEST_ROOT=/workspace
START_EPISODE=125
END_EPISODE=149
DRY_RUN=0

while (($#)); do
  case "$1" in
    --source-host) SOURCE_HOST="$2"; shift 2 ;;
    --source-port) SOURCE_PORT="$2"; shift 2 ;;
    --dest-root) DEST_ROOT="$2"; shift 2 ;;
    --start) START_EPISODE="$2"; shift 2 ;;
    --end) END_EPISODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$SOURCE_HOST" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$ ]] ||
   [[ ! "$SOURCE_PORT" =~ ^[0-9]+$ ]] ||
   [[ ! "$START_EPISODE" =~ ^[0-9]+$ ]] ||
   [[ ! "$END_EPISODE" =~ ^[0-9]+$ ]] ||
   ((START_EPISODE < 0 || END_EPISODE > 149 || START_EPISODE > END_EPISODE)); then
  printf 'usage: %s --source-host user@host --source-port PORT '
  printf '[--dest-root PATH] [--start 0..149] [--end 0..149] [--dry-run]\n'
  exit 2
fi

SHARD_ID="$(printf '%03d_%03d' "$START_EPISODE" "$END_EPISODE")"
RUN_ROOT="$DEST_ROOT/bwa_runs/shared/s10_asset_tail_$SHARD_ID"
STATE="$RUN_ROOT/state.json"
HEARTBEAT="$RUN_ROOT/heartbeat"
LOG="$RUN_ROOT/sync.log"
KNOWN_HOSTS="$DEST_ROOT/bwa_runs/shared/s10_asset_sync/known_hosts"
mkdir -p "$RUN_ROOT" "$DEST_ROOT/bwa_runs/shared/s10_asset_sync" \
  "$DEST_ROOT/datasets/robofactory_multitask"
touch "$KNOWN_HOSTS"
chmod 600 "$KNOWN_HOSTS"

if [[ -f "$STATE" ]]; then
  previous_pid="$(python3 - "$STATE" <<'PY'
import json, sys
try:
    print(int(json.load(open(sys.argv[1]))["pid"]))
except Exception:
    print(0)
PY
)"
  previous_status="$(python3 - "$STATE" <<'PY'
import json, sys
try:
    print(str(json.load(open(sys.argv[1]))["status"]))
except Exception:
    print("UNKNOWN")
PY
)"
  if [[ "$previous_pid" =~ ^[1-9][0-9]*$ ]] &&
     kill -0 "$previous_pid" 2>/dev/null &&
     [[ "$previous_status" =~ ^(PREPARING|DOWNLOADING)$ ]]; then
    printf 'tail shard %s is already active as PID %s\n' "$SHARD_ID" "$previous_pid" >&2
    exit 3
  fi
fi

STARTED_AT="$(date -u +%FT%TZ)"
write_state() {
  local status="$1"
  local detail="$2"
  local temporary="$STATE.tmp"
  printf '{"schema_version":1,"experiment":"r10-shared-assets-tail-%s",' "$SHARD_ID" >"$temporary"
  printf '"status":"%s","stage":"five_task_dataset_tail","detail":"%s",' \
    "$status" "$detail" >>"$temporary"
  printf '"start_episode":%d,"end_episode":%d,"pid":%d,' \
    "$START_EPISODE" "$END_EPISODE" "$$" >>"$temporary"
  printf '"started_at":"%s","updated_at":"%s","log":"%s"}\n' \
    "$STARTED_AT" "$(date -u +%FT%TZ)" "$LOG" >>"$temporary"
  mv "$temporary" "$STATE"
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    touch "$HEARTBEAT"
    sleep 20
  done
}
heartbeat_loop &
HEARTBEAT_PID=$!
DATASET_PIDS=()
STOP_REQUESTED=0
on_signal() {
  STOP_REQUESTED=1
  for pid in "${DATASET_PIDS[@]:-}"; do
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
}
cleanup() {
  local code=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then
    write_state STOPPED signal_requested
  elif ((code != 0)); then
    write_state FAILED "exit_$code"
  fi
}
trap on_signal INT TERM
trap cleanup EXIT

SSH_COMMAND="ssh -p $SOURCE_PORT -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=accept-new"
RSYNC_OPTIONS=(--archive --partial --append-verify --human-readable --info=progress2)
FILTERS=(--include=/hdf5/)
for ((episode=START_EPISODE; episode<=END_EPISODE; episode++)); do
  FILTERS+=("--include=/hdf5/episode_$(printf '%06d' "$episode").hdf5")
done
FILTERS+=(--exclude='*')
if ((DRY_RUN)); then
  RSYNC_OPTIONS+=(--dry-run)
fi

TASKS=(
  lift_barrier
  camera_alignment
  three_robots_stack_cube
  long_pipeline_delivery
  take_photo
)
write_state DOWNLOADING parallel_nonoverlapping_tail
printf '[%s] shard=%s episodes=%03d..%03d mode=parallel-five-task\n' \
  "$STARTED_AT" "$SHARD_ID" "$START_EPISODE" "$END_EPISODE" | tee -a "$LOG"
for task in "${TASKS[@]}"; do
  destination="$DEST_ROOT/datasets/robofactory_multitask/$task/"
  mkdir -p "$destination"
  rsync "${RSYNC_OPTIONS[@]}" "${FILTERS[@]}" -e "$SSH_COMMAND" \
    "$SOURCE_HOST:/workspace/datasets/robofactory_multitask/$task/" \
    "$destination" >"$RUN_ROOT/$task.log" 2>&1 &
  DATASET_PIDS+=("$!")
done

failed=0
for pid in "${DATASET_PIDS[@]}"; do
  wait "$pid" || failed=1
done
if ((STOP_REQUESTED)); then
  exit 130
fi
if ((failed)); then
  printf 'one or more tail transfers failed; inspect %s\n' "$RUN_ROOT" >&2
  exit 3
fi

if ((DRY_RUN)); then
  write_state STOPPED dry_run_complete
else
  expected=$((END_EPISODE - START_EPISODE + 1))
  for task in "${TASKS[@]}"; do
    observed=0
    for ((episode=START_EPISODE; episode<=END_EPISODE; episode++)); do
      path="$DEST_ROOT/datasets/robofactory_multitask/$task/hdf5/episode_$(printf '%06d' "$episode").hdf5"
      [[ -s "$path" ]] && observed=$((observed + 1))
    done
    if ((observed != expected)); then
      printf '%s tail verification failed: %d/%d\n' "$task" "$observed" "$expected" >&2
      exit 3
    fi
  done
  write_state PASSED complete
fi
printf '[%s] tail shard %s complete\n' "$(date -u +%FT%TZ)" "$SHARD_ID" | tee -a "$LOG"
