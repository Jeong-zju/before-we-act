#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1/assets
DATA_ROOT=/workspace/datasets/robofactory_multitask
HF_HOME_PATH=/workspace/.cache/huggingface
HF_CLI=/venv/robofactory-act/bin/hf
PYTHON=/venv/robofactory-act/bin/python
DRY_RUN=0

while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --hf-home) HF_HOME_PATH="$2"; shift 2 ;;
    --hf-cli) HF_CLI="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -x "$HF_CLI" ]] || { printf 'missing hf CLI: %s\n' "$HF_CLI" >&2; exit 3; }
[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 3; }
mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$HF_HOME_PATH"
STATE="$RUN_ROOT/state.json"
HEARTBEAT="$RUN_ROOT/heartbeat.json"
LOG="$RUN_ROOT/download.log"
STARTED_AT="$(date -u +%FT%TZ)"
CURRENT_TASK=none
CURRENT_REPO=none
CHILD_PID=0
HEARTBEAT_PID=0
STOP_REQUESTED=0

write_state() {
  local status="$1" detail="$2"
  PYTHONPATH="$FE_ROOT" "$PYTHON" - "$STATE" "$status" "$detail" "$STARTED_AT" \
    "$CURRENT_TASK" "$CURRENT_REPO" "$$" "$CHILD_PID" "$DATA_ROOT" "$HF_HOME_PATH" "$LOG" <<'PY'
import json, os, sys, time
from pathlib import Path
path, status, detail, started, task, repo, pid, child, data, cache, log = sys.argv[1:]
target = Path(path)
tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
payload = {
    "schema_version": 1, "round": "R13N", "experiment": "six-task-assets",
    "status": status, "stage": "fixed_revision_hf_download", "detail": detail,
    "task": task, "repo": repo, "pid": int(pid), "child_pid": int(child),
    "started_at": started, "updated_at_epoch": time.time(), "data_root": data,
    "hf_home": cache, "log": log,
}
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(tmp, target)
PY
}

heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    PYTHONPATH="$FE_ROOT" "$PYTHON" - "$HEARTBEAT" "$$" "$CHILD_PID" "$CURRENT_TASK" <<'PY'
import json, os, sys, time
from pathlib import Path
path, pid, child, task = sys.argv[1:]
target = Path(path); tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps({"producer":"download_r13n_hf_assets","pid":int(pid),"child_pid":int(child),"task":task,"updated_at_epoch":time.time()}, sort_keys=True)+"\n")
os.replace(tmp, target)
PY
    sleep 20
  done
}

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
    write_state STOPPED "graceful stop; Hub cache and incomplete files preserved"
  elif ((code != 0)); then
    write_state FAILED "asset pipeline exited with code $code"
  fi
}
trap on_signal INT TERM
trap cleanup EXIT

exec > >(tee -a "$LOG") 2>&1
export HF_HOME="$HF_HOME_PATH"
export HF_HUB_DISABLE_XET=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export HF_DOWNLOAD_ATTEMPTS=5
unset HF_TOKEN

verify_task() {
  PYTHONPATH="$FE_ROOT" "$PYTHON" - "$DATA_ROOT" "$1" "$2" <<'PY'
import json, sys
from before_we_act.r13n import validate_manifest
print(json.dumps(validate_manifest(sys.argv[1], sys.argv[2], require_files=sys.argv[3] == "1"), sort_keys=True))
PY
}

download_task() {
  local task="$1" repo="$2" revision="$3" attempt=1 delay=15 code=0
  local command=("$HF_CLI" download "$repo" --repo-type dataset --revision "$revision" --local-dir "$DATA_ROOT/$task")
  ((DRY_RUN)) && command+=(--dry-run)
  while ((attempt <= 5)); do
    printf 'R13N %s: hf download attempt %d/5\n' "$task" "$attempt"
    set +e
    env -u HF_TOKEN "${command[@]}" &
    CHILD_PID=$!
    write_state DOWNLOADING "fixed revision transfer; Xet/cache/resume enabled"
    wait "$CHILD_PID"; code=$?; CHILD_PID=0
    set -e
    ((STOP_REQUESTED)) && return 130
    ((code == 0)) && return 0
    ((attempt == 5)) && return "$code"
    sleep "$delay"
    attempt=$((attempt + 1)); delay=$((delay * 2)); ((delay > 300)) && delay=300
  done
}

write_state PREPARING "checking six pinned manifests"
heartbeat_loop & HEARTBEAT_PID=$!

SPECS=(
  "pass_shoe|zeno-ai/robofactory-pass-shoe-multiview|646bbfec792ed46c78e452acfc06b423ca1410af"
  "place_food|zeno-ai/robofactory-place-food-multiview|2237d907f0b28d3f2e19fa4ea03b4048be2de27d"
)
for spec in "${SPECS[@]}"; do
  IFS='|' read -r task repo revision <<<"$spec"
  CURRENT_TASK="$task"; CURRENT_REPO="$repo@$revision"
  if verify_task "$task" 1 >/dev/null 2>&1; then
    printf 'verified existing R13N dataset: %s\n' "$task"
    continue
  fi
  download_task "$task" "$repo" "$revision"
  ((DRY_RUN)) || verify_task "$task" 1
done

if ((DRY_RUN)); then
  write_state STOPPED "dry-run complete"
else
  write_state VERIFYING "validating all six manifests and local files"
  RECEIPT="$RUN_ROOT/dataset_receipt.json"
  PYTHONPATH="$FE_ROOT" "$PYTHON" - "$DATA_ROOT" "$RECEIPT" <<'PY'
import json, os, sys, time
from pathlib import Path
from before_we_act.r13n import TASKS, validate_manifest
root, output = sys.argv[1:]
payload = {"schema_version":1,"round":"R13N","created_at_epoch":time.time(),"tasks":{task:validate_manifest(root,task,require_files=True) for task in TASKS}}
target=Path(output); tmp=target.with_name(f".{target.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
print(json.dumps({"receipt":str(target),"tasks":list(payload["tasks"])},sort_keys=True))
PY
  CURRENT_TASK=complete; CURRENT_REPO=six-pinned-revisions
  write_state PASSED "six fixed-revision datasets verified"
fi
printf 'R13N asset pipeline complete\n'
