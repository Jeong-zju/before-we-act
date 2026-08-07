#!/usr/bin/env bash
set -Eeuo pipefail

OUTPUT_ROOT=""; EPISODES=1; START_SEED=5000; PYTHON=/venv/robofactory-act/bin/python
ROBOFACTORY=/workspace/RoboFactory
while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT_ROOT" && "$EPISODES" =~ ^[1-9][0-9]*$ && "$START_SEED" =~ ^[1-9][0-9]*$ ]] || { printf 'valid output/episodes/seed required\n' >&2; exit 2; }
STATUS="$OUTPUT_ROOT/status.json"; HEARTBEAT="$OUTPUT_ROOT/heartbeat.json"; CHILD_FILE="$OUTPUT_ROOT/child.pid"
LOG="$OUTPUT_ROOT/collector.log"; RAW_ROOT="$OUTPUT_ROOT/raw"
mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
exec > >(tee -a "$LOG") 2>&1

atomic_state() {
  "$PYTHON" - "$STATUS" "$1" "$2" "$3" "$$" "$CHILD_PID" <<'PY'
import datetime,json,os,sys
path,state,stage,detail,pid,child=sys.argv[1:]
try: current=json.load(open(path))
except (FileNotFoundError,json.JSONDecodeError): current={}
now=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")
d={**current,"schema_version":1,"state":state,"stage":stage,"detail":detail,"pid":int(pid),"child_pid":int(child),"created_at":current.get("created_at",now),"updated_at":now}
tmp=path+f".{os.getpid()}.tmp"; open(tmp,"w").write(json.dumps(d,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}
CHILD_PID=0; HEARTBEAT_PID=0; STOP_REQUESTED=0; TERMINAL_WRITTEN=0
printf '0\n' >"$CHILD_FILE"
heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    local observed=0; [[ -f "$CHILD_FILE" ]] && observed="$(<"$CHILD_FILE")"
    "$PYTHON" - "$HEARTBEAT" "$$" "$observed" <<'PY' >/dev/null 2>&1 || true
import datetime,json,os,sys
path,pid,child=sys.argv[1:]; now=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")
d={"schema_version":1,"producer":"run_r15_expert_collection.sh","pid":int(pid),"child_pid":int(child),"updated_at":now}
tmp=path+f".{os.getpid()}.tmp"; open(tmp,"w").write(json.dumps(d,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
    sleep 20
  done
}
on_signal() { STOP_REQUESTED=1; [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; }
cleanup() {
  local code=$?; kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then atomic_state STOPPED stopped "graceful stop; raw outputs preserved" || true
  elif ((code != 0 && TERMINAL_WRITTEN == 0)); then atomic_state FAILED failed "collector exit=$code; inspect log" || true; fi
}
trap on_signal INT TERM; trap cleanup EXIT
heartbeat_loop & HEARTBEAT_PID=$!
atomic_state PREPARING source_audit "verifying pinned RoboFactory oracle"
EXPECTED_COMMIT=5868242322414a91454e22f1dd9641f613ba1bcf
[[ "$(git -C "$ROBOFACTORY" rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git -C "$ROBOFACTORY" status --porcelain)" ]] || { printf 'RoboFactory source identity differs\n' >&2; exit 3; }
CONFIG="$ROBOFACTORY/robofactory/configs/table/three_robots_stack_cube.yaml"
SOLVER="$ROBOFACTORY/robofactory/planner/solutions/three_robots_stack_cube.py"
for path in "$PYTHON" "$CONFIG" "$SOLVER"; do [[ -e "$path" ]] || { printf 'missing expert source: %s\n' "$path" >&2; exit 3; }; done
"$PYTHON" - "$OUTPUT_ROOT/identity.json" "$EXPECTED_COMMIT" "$CONFIG" "$SOLVER" "$EPISODES" "$START_SEED" <<'PY'
import datetime,hashlib,json,os,sys
target,commit,config,solver,episodes,seed=sys.argv[1:]
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
d={"schema_version":1,"round":"R15-Evolution","source":"RoboFactory motion-planning oracle","source_commit":commit,"config":config,"config_sha256":digest(config),"solver":solver,"solver_sha256":digest(solver),"requested_success_episodes":int(episodes),"start_seed":int(seed),"seed_exclusions":"original demonstrations 3000:3149 and all frozen evaluation seeds","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")}
tmp=target+f".{os.getpid()}.tmp"; open(tmp,"w").write(json.dumps(d,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
PY
atomic_state PREPARING collecting "motion-planning oracle; success-only; raw RGB HDF5"
(
  cd "$ROBOFACTORY"
  exec env PYTHONPATH="$ROBOFACTORY" "$PYTHON" -m robofactory.planner.run --env-id ThreeRobotsStackCube-rf --config "$CONFIG" --obs-mode rgb --num-traj "$EPISODES" --seed "$START_SEED" --only-count-success --sim-backend cpu --render-mode sensors --traj-name "r15_stack_expert_seed_${START_SEED}" --record-dir "$RAW_ROOT"
) &
CHILD_PID=$!; printf '%s\n' "$CHILD_PID" >"$CHILD_FILE"; atomic_state PREPARING collecting "motion-planning oracle; success-only; raw RGB HDF5"
wait "$CHILD_PID"; CHILD_PID=0; printf '0\n' >"$CHILD_FILE"
mapfile -t H5_FILES < <(find "$RAW_ROOT" -type f -name '*.h5' -print)
mapfile -t JSON_FILES < <(find "$RAW_ROOT" -type f -name '*.json' -print)
[[ "${#H5_FILES[@]}" == 1 && "${#JSON_FILES[@]}" == 1 ]] || { printf 'expert collector output pair is incomplete\n' >&2; exit 3; }
"$PYTHON" - "${JSON_FILES[0]}" "$EPISODES" "$OUTPUT_ROOT/receipt.json" <<'PY'
import datetime,hashlib,json,os,sys
source,expected,target=sys.argv[1:]; d=json.load(open(source)); episodes=d.get("episodes",[])
if len(episodes)!=int(expected) or not all(bool(row.get("success")) for row in episodes.values() if isinstance(episodes,dict)):
    if not isinstance(episodes,list) or len(episodes)!=int(expected): raise SystemExit("expert episode receipt differs")
h=hashlib.sha256(open(source,"rb").read()).hexdigest()
out={"schema_version":1,"status":"PASSED","episodes":int(expected),"metadata_json":source,"metadata_sha256":h,"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")}
tmp=target+f".{os.getpid()}.tmp"; open(tmp,"w").write(json.dumps(out,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
PY
TERMINAL_WRITTEN=1; atomic_state PASSED complete "expert raw collection complete; schema conversion remains separate"
printf 'expert collection complete output=%s\n' "$OUTPUT_ROOT"
