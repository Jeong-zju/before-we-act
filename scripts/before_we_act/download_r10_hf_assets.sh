#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/shared/r10_hf_assets
DATA_ROOT=/workspace/datasets/robofactory_multitask
HF_HOME_PATH=/workspace/.cache/huggingface
HF_CLI=/venv/robofactory-act/bin/hf
TOKEN_FIFO=""
ANONYMOUS=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --hf-home) HF_HOME_PATH="$2"; shift 2 ;;
    --hf-cli) HF_CLI="$2"; shift 2 ;;
    --token-fifo) TOKEN_FIFO="$2"; shift 2 ;;
    --anonymous) ANONYMOUS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -x "$HF_CLI" ]] || { printf 'missing hf CLI: %s\n' "$HF_CLI" >&2; exit 3; }
if ((ANONYMOUS == 0)) && [[ ! -p "$TOKEN_FIFO" ]]; then
  printf 'protected token FIFO is required unless --anonymous is used\n' >&2
  exit 3
fi

STATE="$RUN_ROOT/state.json"
HEARTBEAT="$RUN_ROOT/heartbeat"
LOG="$RUN_ROOT/download.log"
mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$HF_HOME_PATH"
if [[ -f "$STATE" ]]; then
  read -r old_pid old_status < <(python3 - "$STATE" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
    print(int(payload.get("pid", 0)), str(payload.get("status", "UNKNOWN")))
except Exception:
    print(0, "UNKNOWN")
PY
)
  if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null &&
     [[ "$old_status" =~ ^(PREPARING|DOWNLOADING|VERIFYING)$ ]]; then
    printf 'R10 Hugging Face asset download is already active as PID %s\n' "$old_pid" >&2
    exit 3
  fi
fi

STARTED_AT="$(date -u +%FT%TZ)"
CURRENT_TASK=none
CURRENT_REPO=none
CHILD_PID=0
HEARTBEAT_PID=""
TOKEN_INPUT=""
write_state() {
  local status="$1" detail="$2" completed="${3:-0}" total="${4:-750}"
  python3 - "$STATE.tmp" "$status" "$detail" "$STARTED_AT" "$LOG" \
    "$CURRENT_TASK" "$CURRENT_REPO" "$$" "$CHILD_PID" "$completed" "$total" \
    "$DATA_ROOT" "$HF_HOME_PATH" <<'PY'
import json, os, sys
from datetime import datetime, timezone
(
    path, status, detail, started, log, task, repo, pid, child_pid,
    completed, total, data_root, hf_home,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "experiment": "r10-s0-huggingface-assets",
    "status": status,
    "stage": "five_task_hf_download",
    "detail": detail,
    "task": task,
    "repo": repo,
    "pid": int(pid),
    "child_pid": int(child_pid),
    "completed_episode_files": int(completed),
    "total_episode_files": int(total),
    "started_at": started,
    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "log": log,
    "data_root": data_root,
    "hf_home": hf_home,
    "download_contract": {
        "command": "official hf download",
        "repo_type": "dataset",
        "xet": "enabled",
        "workers": "CLI default (8)",
        "download_timeout_seconds": 600,
        "etag_timeout_seconds": 60,
        "max_attempts": 5,
        "resume": "same local-dir, HF cache and .incomplete files",
    },
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(path, path.removesuffix(".tmp"))
PY
}
episode_count() {
  find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'episode_*.hdf5' \
    2>/dev/null | wc -l | tr -d '[:space:]'
}
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    touch "$HEARTBEAT"
    sleep 20
  done
}
STOP_REQUESTED=0
on_signal() {
  STOP_REQUESTED=1
  if [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT "$CHILD_PID" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  if [[ -n "$HEARTBEAT_PID" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  TOKEN_INPUT=""
  unset TOKEN_INPUT HF_TOKEN
  [[ -n "$TOKEN_FIFO" ]] && unlink "$TOKEN_FIFO" 2>/dev/null || true
  if ((STOP_REQUESTED)); then
    write_state STOPPED "operator stop; Hub cache and partial files preserved" "$(episode_count)" 750
  elif ((code != 0)); then
    write_state FAILED "download pipeline exited with code $code" "$(episode_count)" 750
  fi
}
trap on_signal INT TERM
trap cleanup EXIT

if ((ANONYMOUS == 0)); then
  IFS= read -r TOKEN_INPUT <"$TOKEN_FIFO"
  unlink "$TOKEN_FIFO"
  TOKEN_FIFO=""
  if [[ "$TOKEN_INPUT" != hf_* || "$TOKEN_INPUT" =~ [[:space:]] ]]; then
    printf 'invalid protected Hugging Face token input\n' >&2
    exit 3
  fi
fi

exec > >(tee -a "$LOG") 2>&1
export HF_HOME="$HF_HOME_PATH"
export HF_HUB_DISABLE_XET=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export HF_DOWNLOAD_ATTEMPTS=5
export HF_DOWNLOAD_INITIAL_BACKOFF_SECONDS=15

DATASET_SPECS=(
  "lift_barrier|zeno-ai/robofactory-lift-barrier-multiview|6ab620091677e69370412f08cd7adecacc28c146|2"
  "long_pipeline_delivery|zeno-ai/robofactory-long-pipeline-delivery-multiview|fee628311ff52a3ae0ddfddf82379c63d28f7533|4"
  "take_photo|zeno-ai/robofactory-take-photo-multiview|3966385a4c688a5610d4b6cde044150f6b73d320|4"
  "three_robots_stack_cube|zeno-ai/robofactory-three-robots-stack-cube-multiview|d0ae346bf2ce63ec801af1f036c08a4a91faf366|3"
  "camera_alignment|zeno-ai/robofactory-camera-alignment-multiview|e204af13f7191dfd86dab3da529316a51558f479|3"
)

verify_task() {
  local task="$1" agents="$2"
  python3 "$FE_ROOT/scripts/verify_s2_r3_dataset_local.py" \
    --manifest "$DATA_ROOT/$task/training_manifest.json" \
    --expected-task "$task" --expected-episodes 150 --expected-agent-count "$agents"
}

run_download() {
  local task="$1" repo="$2" revision="$3" attempt=1 delay=15 code
  local command=(
    "$HF_CLI" download "$repo" --repo-type dataset --revision "$revision"
    --local-dir "$DATA_ROOT/$task"
  )
  ((DRY_RUN)) && command+=(--dry-run)
  while ((attempt <= 5)); do
    printf 'Hugging Face %s: attempt %d/5 (Xet enabled, CLI default workers).\n' \
      "$task" "$attempt"
    set +e
    if [[ -n "$TOKEN_INPUT" ]]; then
      HF_TOKEN="$TOKEN_INPUT" "${command[@]}" &
    else
      env -u HF_TOKEN "${command[@]}" &
    fi
    CHILD_PID=$!
    write_state DOWNLOADING \
      "official Hub transfer active; Xet enabled; partial files preserved" \
      "$(episode_count)" 750
    wait "$CHILD_PID"
    code=$?
    CHILD_PID=0
    set -e
    ((STOP_REQUESTED)) && return 130
    ((code == 0)) && return 0
    ((attempt == 5)) && return "$code"
    printf 'Hugging Face %s failed; retrying in %d seconds; cache retained.\n' \
      "$task" "$delay" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
    ((delay > 300)) && delay=300
  done
}

write_state PREPARING "S0 fixed-revision Hugging Face preflight" "$(episode_count)" 750
heartbeat_loop &
HEARTBEAT_PID=$!
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r task repo revision agents <<<"$spec"
  CURRENT_TASK="$task"
  CURRENT_REPO="$repo@$revision"
  if verify_task "$task" "$agents" >/dev/null 2>&1; then
    printf 'Verified complete local dataset: %s\n' "$task"
    continue
  fi
  write_state DOWNLOADING "S0 fixed revision; Xet enabled; default workers; retry<=5" \
    "$(episode_count)" 750
  run_download "$task" "$repo" "$revision"
  if ((DRY_RUN == 0)); then
    write_state VERIFYING "verifying 150 episodes and camera contract" "$(episode_count)" 750
    verify_task "$task" "$agents"
  fi
done

TOKEN_INPUT=""
unset TOKEN_INPUT HF_TOKEN
if ((DRY_RUN)); then
  write_state STOPPED "dry-run complete; no dataset payload requested" "$(episode_count)" 750
else
  CURRENT_TASK=complete
  CURRENT_REPO=fixed-five-revisions
  write_state PASSED "five fixed-revision datasets verified" 750 750
fi
printf 'R10 Hugging Face asset pipeline complete.\n'
