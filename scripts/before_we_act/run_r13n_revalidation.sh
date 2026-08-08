#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v2-clipfix
SOURCE_RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1
CACHE_ROOT=/workspace/bwa_runs/shared/r13n_native_full_cache_v1
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PYTHON=/venv/robofactory-act/bin/python
SESSION=bwa-r13n-v2
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --source-run-root) SOURCE_RUN_ROOT="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --vision-artifact) VISION_ARTIFACT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

CONFIG="$FE_ROOT/configs/before_we_act/r13n/b6.yaml"
INDEX="$CACHE_ROOT/index.json"
CHECKPOINT="$SOURCE_RUN_ROOT/train/formal/checkpoints/checkpoint_130000.pt"
SOURCE_ACCEPTANCE="$SOURCE_RUN_ROOT/acceptance.json"
STATE="$RUN_ROOT/status.json"
HEARTBEAT="$RUN_ROOT/heartbeat.json"
LOG_ROOT="$RUN_ROOT/logs"
OFFLINE="$RUN_ROOT/validation/offline.json"
SEEDS="$RUN_ROOT/seeds"
[[ "$RUN_ROOT" != "$SOURCE_RUN_ROOT" ]] || { printf 'revalidation run root must differ from source\n' >&2; exit 3; }
[[ -x "$PYTHON" && -f "$CONFIG" && -f "$INDEX" && -f "$CHECKPOINT" && -f "$SOURCE_ACCEPTANCE" ]] || { printf 'R13N revalidation input is missing\n' >&2; exit 3; }
[[ "$(jq -r '.passed // false' "$SOURCE_ACCEPTANCE")" == true ]] || { printf 'source R13N acceptance is not PASSED\n' >&2; exit 3; }
[[ -f "$VISION_ARTIFACT/config.json" && -f "$VISION_ARTIFACT/model.safetensors" ]] || { printf 'R13N vision artifact is missing\n' >&2; exit 3; }
[[ "$(git -C "$FE_ROOT" branch --show-current)" == feat/model-improvements && -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'R13N revalidation requires clean feat/model-improvements\n' >&2; exit 3; }

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
if [[ ! -d "$SEEDS" ]]; then cp -a "$SOURCE_RUN_ROOT/seeds" "$SEEDS"; fi
[[ -f "$SEEDS/protocol.json" ]] || { printf 'R13N seed protocol is missing\n' >&2; exit 3; }

STARTED_AT="$(date -u +%FT%TZ)"
BRANCH="$(git -C "$FE_ROOT" branch --show-current)"
COMMIT="$(git -C "$FE_ROOT" rev-parse HEAD)"
CHILD_PID=0
CHILDREN=()
HEARTBEAT_PID=0
STOP_REQUESTED=0

write_state() {
  local status="$1" stage="$2" program="$3" detail="$4"
  "$PYTHON" - "$STATE" "$status" "$stage" "$program" "$detail" "$STARTED_AT" "$BRANCH" "$COMMIT" "$SESSION" "$$" "$CHILD_PID" "$RUN_ROOT" "$CACHE_ROOT" <<'PY'
import json, os, sys, time
from pathlib import Path
(path,status,stage,program,detail,started,branch,commit,session,pid,child,run_root,cache_root)=sys.argv[1:]
target=Path(path); temporary=target.with_name(f'.{target.name}.{os.getpid()}.tmp')
payload={"schema_version":1,"round":"R13N","experiment":"b6_act_six_task_clipfix","status":status,"stage":stage,"program":program,"detail":detail,"started_at":started,"updated_at_epoch":time.time(),"branch":branch,"commit":commit,"tmux_session":session,"pid":int(pid),"child_pid":int(child),"run_root":run_root,"cache_root":cache_root,"gpu_assignment":{"offline":0,"closed_loop":"task waves across 0,1,2,3"}}
temporary.write_text(json.dumps(payload,sort_keys=True)+'\n'); os.replace(temporary,target)
PY
}

heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" - "$HEARTBEAT" "$$" <<'PY'
import json, os, sys, time
from pathlib import Path
path=Path(sys.argv[1]); temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
temporary.write_text(json.dumps({"producer":"run_r13n_revalidation","pid":int(sys.argv[2]),"updated_at_epoch":time.time()},sort_keys=True)+'\n'); os.replace(temporary,path)
PY
    sleep 20
  done
}

stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -INT "$pid" 2>/dev/null || true
  done
}
on_signal() { STOP_REQUESTED=1; stop_children; }
cleanup() {
  local code=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  if ((STOP_REQUESTED)); then write_state STOPPED stopped run_r13n_revalidation.sh "graceful stop requested; partial outputs preserved"
  elif ((code!=0)); then write_state FAILED failed run_r13n_revalidation.sh "revalidation exited with code $code; inspect logs"
  fi
}
trap on_signal INT TERM
trap cleanup EXIT

run_child() {
  local status="$1" stage="$2" program="$3" detail="$4" log="$5"; shift 5
  write_state "$status" "$stage" "$program" "$detail"
  "$@" >>"$log" 2>&1 & CHILD_PID=$!; CHILDREN=("$CHILD_PID")
  write_state "$status" "$stage" "$program" "$detail"
  set +e; wait "$CHILD_PID"; local code=$?; set -e
  CHILD_PID=0; CHILDREN=(); ((code==0)) || return "$code"
}

"$PYTHON" - "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/code_update_receipt.json" "$STARTED_AT" "$BRANCH" "$COMMIT" "$SESSION" "$RUN_ROOT" "$SOURCE_RUN_ROOT" "$CACHE_ROOT" "$INDEX" "$VISION_ARTIFACT" "$CONFIG" "$CHECKPOINT" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
(manifest,receipt,started,branch,commit,session,run_root,source_run,cache,index,vision,config,checkpoint)=sys.argv[1:]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload={"schema_version":1,"round":"R13N","run_id":"r13n-no-stack-v2-clipfix","run_variant":"normalized_clip_physical_bounds_fix_v2","normalized_clip":96.0,"started_at":started,"branch":branch,"commit":commit,"tmux_session":session,"run_root":run_root,"source_run_root":source_run,"source_checkpoint":checkpoint,"source_checkpoint_sha256":sha(checkpoint),"feature_cache":cache,"feature_index":index,"vision_artifact":vision,"config":config,"config_sha256":sha(config),"tasks":["lift_barrier","camera_alignment","long_pipeline_delivery","take_photo","pass_shoe","place_food"],"created_at_epoch":time.time()}
for target,value in ((manifest,payload),(receipt,{"schema_version":1,"round":"R13N","fix":"raise normalized guard from 5 to 96 and clamp denormalized actions to simulator physical bounds","training_reused":True,"checkpoint_weights_changed":False,"source_run_root":source_run,"source_checkpoint_sha256":sha(checkpoint),"effective_evaluation_commit":commit,"created_at_epoch":time.time()})):
 path=Path(target); temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp'); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(temporary,path)
PY

write_state PREPARING revalidation_prepare run_r13n_revalidation.sh "reusing immutable 130k weights and paired seed protocol"
heartbeat_loop & HEARTBEAT_PID=$!

if [[ ! -f "$OFFLINE" ]]; then
  mkdir -p "$(dirname "$OFFLINE")"
  run_child VALIDATING offline evaluate_r13n_offline.py "full held-out validation without destructive normalized clipping" "$LOG_ROOT/offline.log" \
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.evaluate_r13n_offline --config "$CONFIG" --checkpoint "$CHECKPOINT" --full-index "$INDEX" --output "$OFFLINE" --heartbeat "$RUN_ROOT/offline_heartbeat.json" --device cuda:0 --batch-size 10 --workers 2
fi

evaluate_task() {
  local stage="$1" task="$2" gpu="$3" video_args=()
  local output="$RUN_ROOT/evaluation/$stage/$task.json" log="$LOG_ROOT/${stage}_${task}.log" heartbeat="$RUN_ROOT/evaluation/$stage/${task}_heartbeat.json"
  [[ -f "$output" ]] && { EVAL_PID=0; return 0; }
  [[ "$stage" == discovery ]] && video_args=(--video-dir "$RUN_ROOT/videos")
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$FE_ROOT:/workspace/RoboFactory" "$PYTHON" -m before_we_act.evaluate_r13n \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --vision-artifact "$VISION_ARTIFACT" --task "$task" --stage "$stage" \
    --seed-file "$SEEDS/$stage/$task.json" --episodes 20 --device cuda:0 --output "$output" --heartbeat "$heartbeat" --resume-log "$log" "${video_args[@]}" >>"$log" 2>&1 &
  EVAL_PID=$!
}

for stage in discovery validation formal; do
  write_state VALIDATING "$stage" evaluate_r13n.py "clip-fixed candidate-native six-task ${stage^}20"
  ORDER=(camera_alignment long_pipeline_delivery take_photo lift_barrier pass_shoe place_food)
  for wave_start in 0 4; do
    EVAL_PIDS=(); wave_end=$((wave_start+4)); ((wave_end>6)) && wave_end=6
    for ((index_position=wave_start; index_position<wave_end; index_position++)); do
      task="${ORDER[$index_position]}"; gpu=$((index_position-wave_start)); EVAL_PID=0; evaluate_task "$stage" "$task" "$gpu"; ((EVAL_PID>0)) && EVAL_PIDS+=("$EVAL_PID")
    done
    CHILDREN=("${EVAL_PIDS[@]}"); codes=0
    for pid in "${EVAL_PIDS[@]}"; do set +e; wait "$pid"; code=$?; set -e; ((code==0)) || codes=$code; done
    CHILDREN=(); ((codes==0)) || exit "$codes"
  done
done

ACCEPTANCE="$RUN_ROOT/acceptance.json"
run_child ACCEPTING acceptance accept_r13n.py "checking clip-fixed offline validation, 360 native rollouts and runtime health" "$LOG_ROOT/acceptance.log" \
  env PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/accept_r13n.py" --run-root "$RUN_ROOT" --checkpoint "$CHECKPOINT" --offline "$OFFLINE" --seed-protocol "$SEEDS/protocol.json" --output "$ACCEPTANCE"
write_state PASSED complete accept_r13n.py "R13N clip-fixed baseline established; see acceptance.json"
printf 'R13N clip-fixed revalidation complete: %s\n' "$ACCEPTANCE"
