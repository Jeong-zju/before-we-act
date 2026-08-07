#!/usr/bin/env bash
set -Eeuo pipefail

CACHE_ROOT=""; RAW_HDF5=""; RAW_JSON=""; EPISODES=""; GPU_INDEX=""; PYTHON=/venv/robofactory-act/bin/python
BASE_INDEX=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2/index.json
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
while (($#)); do
  case "$1" in
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --raw-hdf5) RAW_HDF5="$2"; shift 2 ;;
    --raw-json) RAW_JSON="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$CACHE_ROOT" && -n "$RAW_HDF5" && -n "$RAW_JSON" && "$EPISODES" =~ ^[1-9][0-9]*$ && "$GPU_INDEX" =~ ^[0-3]$ ]] || { printf 'valid cache/raw/episodes/GPU required\n' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$CACHE_ROOT/cache.log"; STATE="$CACHE_ROOT/state.json"; HEARTBEAT="$CACHE_ROOT/heartbeat.json"; CHILD_FILE="$CACHE_ROOT/child.pid"
mkdir -p "$CACHE_ROOT"; exec > >(tee -a "$LOG") 2>&1
CHILD_PID=0; STOP_REQUESTED=0; TERMINAL_WRITTEN=0; printf '0\n' >"$CHILD_FILE"
write_terminal() {
  "$PYTHON" - "$STATE" "$1" "$2" "$3" <<'PY'
import datetime,json,os,sys
path,state,stage,detail=sys.argv[1:]
try: current=json.load(open(path))
except (FileNotFoundError,json.JSONDecodeError): current={}
now=datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
d={**current,"state":state,"stage":stage,"detail":detail,"updated_at":now}
tmp=path+f'.{os.getpid()}.tmp'; open(tmp,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
PY
}
on_signal() { STOP_REQUESTED=1; [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; }
cleanup() {
  local code=$?
  if ((STOP_REQUESTED)); then write_terminal STOPPED stopped "graceful stop; completed immutable expert shards preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then write_terminal FAILED failed "expert cache exit=$code; inspect cache.log" || true; fi
}
trap on_signal INT TERM; trap cleanup EXIT
"$PYTHON" - "$CACHE_ROOT/process.json" "$GPU_INDEX" "$(git -C "$ROOT" branch --show-current)" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import datetime,json,os,sys
path,gpu,branch,commit=sys.argv[1:]
d={"schema_version":1,"program":"prepare_r15_expert_full_cache.py","pid":os.getppid(),"gpu":int(gpu),"session":"bwa-r15-expert-cache","branch":branch,"commit":commit,"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}
tmp=path+f'.{os.getpid()}.tmp'; open(tmp,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
PY
(
  cd "$ROOT"
  exec env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$ROOT" "$PYTHON" scripts/before_we_act/prepare_r15_expert_full_cache.py \
    --raw-hdf5 "$RAW_HDF5" --raw-json "$RAW_JSON" --base-index "$BASE_INDEX" \
    --output-root "$CACHE_ROOT/features" --vision-artifact "$VISION_ARTIFACT" \
    --episodes "$EPISODES" --device cuda:0 --frame-batch-size 1 --image-batch-size 4 \
    --state "$STATE" --heartbeat "$HEARTBEAT"
) &
CHILD_PID=$!; printf '%s\n' "$CHILD_PID" >"$CHILD_FILE"; wait "$CHILD_PID"
CHILD_PID=0; printf '0\n' >"$CHILD_FILE"
[[ -f "$CACHE_ROOT/features/index.json" && "$(jq -r .state "$STATE")" == PASSED ]] || { printf 'expert cache completion receipt differs\n' >&2; exit 3; }
TERMINAL_WRITTEN=1
printf 'expert cache complete index=%s\n' "$CACHE_ROOT/features/index.json"
