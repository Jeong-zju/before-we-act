#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_HOST=""
SOURCE_PORT=""
DEST_ROOT=/workspace
TASK=""
START_EPISODE=""
END_EPISODE=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --source-host) SOURCE_HOST="$2"; shift 2 ;;
    --source-port) SOURCE_PORT="$2"; shift 2 ;;
    --dest-root) DEST_ROOT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --start) START_EPISODE="$2"; shift 2 ;;
    --end) END_EPISODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$TASK" in
  lift_barrier|camera_alignment|three_robots_stack_cube|long_pipeline_delivery|take_photo) ;;
  *) printf 'unsupported task: %s\n' "$TASK" >&2; exit 2 ;;
esac
if [[ ! "$SOURCE_HOST" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$ ]] ||
   [[ ! "$SOURCE_PORT" =~ ^[0-9]+$ ]] ||
   [[ ! "$START_EPISODE" =~ ^[0-9]+$ ]] ||
   [[ ! "$END_EPISODE" =~ ^[0-9]+$ ]] ||
   ((START_EPISODE < 0 || END_EPISODE > 149 || START_EPISODE > END_EPISODE)); then
  printf 'usage: %s --source-host user@host --source-port PORT --task TASK ' "$0" >&2
  printf -- '--start 0..149 --end 0..149 [--dest-root PATH] [--dry-run]\n' >&2
  exit 2
fi

SHARD_ID="${TASK}_$(printf '%03d_%03d' "$START_EPISODE" "$END_EPISODE")"
RUN_ROOT="$DEST_ROOT/bwa_runs/shared/s10_asset_shard_$SHARD_ID"
STATE="$RUN_ROOT/state.json"
HEARTBEAT="$RUN_ROOT/heartbeat"
LOG="$RUN_ROOT/sync.log"
KNOWN_HOSTS="$DEST_ROOT/bwa_runs/shared/s10_asset_sync/known_hosts"
mkdir -p "$RUN_ROOT" "$DEST_ROOT/bwa_runs/shared/s10_asset_sync" \
  "$DEST_ROOT/datasets/robofactory_multitask/$TASK"
touch "$KNOWN_HOSTS"
chmod 600 "$KNOWN_HOSTS"

if [[ -f "$STATE" ]]; then
  read -r previous_pid previous_status < <(python3 - "$STATE" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
    print(int(payload["pid"]), str(payload["status"]))
except Exception:
    print(0, "UNKNOWN")
PY
)
  if [[ "$previous_pid" =~ ^[1-9][0-9]*$ ]] &&
     kill -0 "$previous_pid" 2>/dev/null &&
     [[ "$previous_status" =~ ^(PREPARING|DOWNLOADING)$ ]]; then
    printf 'task shard %s is already active as PID %s\n' "$SHARD_ID" "$previous_pid" >&2
    exit 3
  fi
fi

STARTED_AT="$(date -u +%FT%TZ)"
write_state() {
  local status="$1" detail="$2" temporary="$STATE.tmp"
  python3 - "$temporary" "$status" "$detail" "$STARTED_AT" "$LOG" \
    "$TASK" "$START_EPISODE" "$END_EPISODE" "$$" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, status, detail, started, log, task, first, last, pid = sys.argv[1:]
payload = {
    "schema_version": 1,
    "experiment": f"r10-shared-assets-shard-{task}-{int(first):03d}_{int(last):03d}",
    "status": status,
    "stage": "task_dataset_shard",
    "detail": detail,
    "task": task,
    "start_episode": int(first),
    "end_episode": int(last),
    "pid": int(pid),
    "started_at": started,
    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "log": log,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(path, path.removesuffix(".tmp"))
PY
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    touch "$HEARTBEAT"
    sleep 20
  done
}
CHILD_PID=0
STOP_REQUESTED=0
on_signal() {
  STOP_REQUESTED=1
  if [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT "$CHILD_PID" 2>/dev/null || true
  fi
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
heartbeat_loop &
HEARTBEAT_PID=$!

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

write_state DOWNLOADING nonoverlapping_task_shard
printf '[%s] task=%s episodes=%03d..%03d\n' \
  "$STARTED_AT" "$TASK" "$START_EPISODE" "$END_EPISODE" | tee -a "$LOG"
destination="$DEST_ROOT/datasets/robofactory_multitask/$TASK/"
rsync "${RSYNC_OPTIONS[@]}" "${FILTERS[@]}" -e "$SSH_COMMAND" \
  "$SOURCE_HOST:/workspace/datasets/robofactory_multitask/$TASK/" \
  "$destination" >>"$LOG" 2>&1 &
CHILD_PID=$!
wait "$CHILD_PID"
CHILD_PID=0

if ((STOP_REQUESTED)); then
  exit 130
elif ((DRY_RUN)); then
  write_state STOPPED dry_run_complete
else
  expected=$((END_EPISODE - START_EPISODE + 1))
  observed=0
  for ((episode=START_EPISODE; episode<=END_EPISODE; episode++)); do
    path="$destination/hdf5/episode_$(printf '%06d' "$episode").hdf5"
    [[ -s "$path" ]] && observed=$((observed + 1))
  done
  if ((observed != expected)); then
    printf '%s verification failed: %d/%d\n' "$TASK" "$observed" "$expected" >&2
    exit 3
  fi
  write_state PASSED complete
fi
printf '[%s] task shard %s complete\n' "$(date -u +%FT%TZ)" "$SHARD_ID" | tee -a "$LOG"
