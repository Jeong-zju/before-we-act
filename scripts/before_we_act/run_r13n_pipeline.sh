#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT=/workspace/bwa_runs/r13n-no-stack-v1
DATA_ROOT=/workspace/datasets/robofactory_multitask
CACHE_ROOT=/workspace/bwa_runs/shared/r13n_native_full_cache_v1
REUSE_INDEX=/workspace/bwa_runs/shared/r12r4_native_full_cache_v2/index.json
VISION_ARTIFACT=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
PYTHON=/venv/robofactory-act/bin/python
SESSION=bwa-r13n
while (($#)); do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --reuse-index) REUSE_INDEX="$2"; shift 2 ;;
    --vision-artifact) VISION_ARTIFACT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

CONFIG="$FE_ROOT/configs/before_we_act/r13n/b6.yaml"
INDEX="$CACHE_ROOT/index.json"
STATE="$RUN_ROOT/status.json"
HEARTBEAT="$RUN_ROOT/heartbeat.json"
ASSET_STATE="$RUN_ROOT/assets/state.json"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
[[ -x "$PYTHON" && -f "$CONFIG" && -f "$REUSE_INDEX" && -f "$VISION_ARTIFACT/config.json" && -f "$VISION_ARTIFACT/model.safetensors" ]] || { printf 'R13N required environment/artifact is missing\n' >&2; exit 3; }
[[ "$(jq -r '.status // ""' "$ASSET_STATE" 2>/dev/null)" == PASSED ]] || { printf 'R13N assets are not PASSED: %s\n' "$ASSET_STATE" >&2; exit 3; }

STARTED_AT="$(date -u +%FT%TZ)"
BRANCH="$(git -C "$FE_ROOT" branch --show-current)"
COMMIT="$(git -C "$FE_ROOT" rev-parse HEAD)"
[[ "$BRANCH" == feat/model-improvements && -z "$(git -C "$FE_ROOT" status --porcelain)" ]] || { printf 'R13N requires clean feat/model-improvements\n' >&2; exit 3; }
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
target=Path(path); tmp=target.with_name(f'.{target.name}.{os.getpid()}.tmp')
payload={"schema_version":1,"round":"R13N","experiment":"b6_act_six_task","status":status,"stage":stage,"program":program,"detail":detail,"started_at":started,"updated_at_epoch":time.time(),"branch":branch,"commit":commit,"tmux_session":session,"pid":int(pid),"child_pid":int(child),"run_root":run_root,"cache_root":cache_root,"gpu_assignment":{"cache":"0,1,2,3","training":0,"offline":0,"closed_loop":"task waves across 0,1,2,3"}}
tmp.write_text(json.dumps(payload,sort_keys=True)+'\n'); os.replace(tmp,target)
PY
}

heartbeat_loop() {
  while kill -0 "$$" 2>/dev/null; do
    "$PYTHON" - "$HEARTBEAT" "$$" <<'PY'
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_name(f'.{p.name}.{os.getpid()}.tmp'); t.write_text(json.dumps({"producer":"run_r13n_pipeline","pid":int(sys.argv[2]),"updated_at_epoch":time.time()},sort_keys=True)+'\n'); os.replace(t,p)
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
  if ((STOP_REQUESTED)); then write_state STOPPED stopped run_r13n_pipeline.sh "graceful stop requested; latest checkpoint and partial outputs preserved"
  elif ((code!=0)); then write_state FAILED failed run_r13n_pipeline.sh "pipeline exited with code $code; inspect logs"
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
  CHILD_PID=0; CHILDREN=()
  ((code==0)) || return "$code"
}

atomic_manifest() {
  "$PYTHON" - "$RUN_ROOT/run_manifest.json" "$STARTED_AT" "$BRANCH" "$COMMIT" "$SESSION" "$RUN_ROOT" "$DATA_ROOT" "$CACHE_ROOT" "$INDEX" "$VISION_ARTIFACT" <<'PY'
import json, os, sys, time
from pathlib import Path
path,started,branch,commit,session,run_root,data,cache,index,vision=sys.argv[1:]
p=Path(path); t=p.with_name(f'.{p.name}.{os.getpid()}.tmp'); payload={"schema_version":1,"round":"R13N","run_id":"r13n-no-stack-v1","started_at":started,"branch":branch,"commit":commit,"tmux_session":session,"run_root":run_root,"data_root":data,"hf_cache":"/workspace/.cache/huggingface","feature_cache":cache,"feature_index":index,"vision_artifact":vision,"config":"configs/before_we_act/r13n/b6.yaml","tasks":["lift_barrier","camera_alignment","long_pipeline_delivery","take_photo","pass_shoe","place_food"],"gpu_assignment":{"cache":[0,1,2,3],"training":0,"offline":0,"evaluation_waves":[["camera_alignment","long_pipeline_delivery","take_photo","lift_barrier"],["pass_shoe","place_food"]]},"created_at_epoch":time.time()}; t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

atomic_manifest
write_state PREPARING cache_prepare prepare_r13n_full_cache.py "building only two new-task DINO caches; reusing four hash-pinned caches"
heartbeat_loop & HEARTBEAT_PID=$!

if [[ ! -f "$INDEX" ]]; then
  CACHE_PIDS=()
  for rank in 0 1 2 3; do
    env CUDA_VISIBLE_DEVICES="$rank" PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/prepare_r13n_full_cache.py" \
      --mode shard --rank "$rank" --world-size 4 --data-root "$DATA_ROOT" --vision-artifact "$VISION_ARTIFACT" \
      --output-root "$CACHE_ROOT" --state "$CACHE_ROOT/rank_${rank}_state.json" --heartbeat "$CACHE_ROOT/rank_${rank}_heartbeat.json" \
      --frame-batch-size 1 --image-batch-size 5 --device cuda:0 >>"$LOG_ROOT/cache_rank_${rank}.log" 2>&1 &
    CACHE_PIDS+=("$!")
  done
  CHILDREN=("${CACHE_PIDS[@]}")
  codes=0
  for pid in "${CACHE_PIDS[@]}"; do set +e; wait "$pid"; code=$?; set -e; ((code==0)) || codes=$code; done
  CHILDREN=(); ((codes==0)) || exit "$codes"
  run_child PREPARING cache_index prepare_r13n_full_cache.py "combining four reusable tasks with two new tasks" "$LOG_ROOT/cache_index.log" \
    env PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/prepare_r13n_full_cache.py" --mode index --world-size 4 \
      --data-root "$DATA_ROOT" --output-root "$CACHE_ROOT" --state "$CACHE_ROOT/index_state.json" --heartbeat "$CACHE_ROOT/index_heartbeat.json" \
      --reuse-index "$REUSE_INDEX" --index "$INDEX"
fi

PREFLIGHT="$RUN_ROOT/preflight"
if [[ ! -f "$PREFLIGHT/receipt.json" ]]; then
  run_child TRAINING preflight_train train_r13n_baseline.py "two-update from-scratch preflight" "$LOG_ROOT/preflight_train.log" \
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_r13n_baseline --config "$CONFIG" --full-index "$INDEX" --output "$PREFLIGHT/train" --updates 2 --workers 2 --device cuda:0 --heartbeat "$PREFLIGHT/train_heartbeat.json"
  run_child VALIDATING preflight_restore verify_r13n_preflight.py "strict restore and causal action effects" "$LOG_ROOT/preflight_verify.log" \
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/verify_r13n_preflight.py" --config "$CONFIG" --full-index "$INDEX" --checkpoint "$PREFLIGHT/train/checkpoints/checkpoint_000002.pt" --device cuda:0 --output "$PREFLIGHT/receipt.json"
fi

TRAIN_ROOT="$RUN_ROOT/train/formal"
CHECKPOINT="$TRAIN_ROOT/checkpoints/checkpoint_130000.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  TRAIN_ARGS=(--config "$CONFIG" --full-index "$INDEX" --output "$TRAIN_ROOT" --workers 2 --device cuda:0 --heartbeat "$RUN_ROOT/train_heartbeat.json")
  if [[ -f "$TRAIN_ROOT/checkpoints/checkpoint_latest.pt" ]]; then TRAIN_ARGS+=(--resume "$TRAIN_ROOT/checkpoints/checkpoint_latest.pt"); fi
  run_child TRAINING train train_r13n_baseline.py "from-scratch task-balanced 130k ACT training on GPU0" "$LOG_ROOT/train.log" \
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.train_r13n_baseline "${TRAIN_ARGS[@]}"
fi

OFFLINE="$RUN_ROOT/validation/offline.json"
if [[ ! -f "$OFFLINE" ]]; then
  run_child VALIDATING offline evaluate_r13n_offline.py "full six-task held-out timestep validation" "$LOG_ROOT/offline.log" \
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FE_ROOT" "$PYTHON" -m before_we_act.evaluate_r13n_offline --config "$CONFIG" --checkpoint "$CHECKPOINT" --full-index "$INDEX" --output "$OFFLINE" --heartbeat "$RUN_ROOT/offline_heartbeat.json" --device cuda:0 --batch-size 10 --workers 2
fi

SEEDS="$RUN_ROOT/seeds"
if [[ ! -f "$SEEDS/protocol.json" ]]; then
  run_child PREPARING seed_protocol prepare_r13n_seed_protocol.py "freezing 360 disjoint seeds" "$LOG_ROOT/seeds.log" \
    env PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/prepare_r13n_seed_protocol.py" --output-root "$SEEDS"
fi

evaluate_task() {
  local stage="$1" task="$2" gpu="$3" video_args=()
  local output="$RUN_ROOT/evaluation/$stage/$task.json" log="$LOG_ROOT/${stage}_${task}.log" heartbeat="$RUN_ROOT/evaluation/$stage/${task}_heartbeat.json"
  [[ -f "$output" ]] && return 0
  [[ "$stage" == discovery ]] && video_args=(--video-dir "$RUN_ROOT/videos")
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$FE_ROOT:/workspace/RoboFactory" "$PYTHON" -m before_we_act.evaluate_r13n \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --vision-artifact "$VISION_ARTIFACT" --task "$task" --stage "$stage" \
    --seed-file "$SEEDS/$stage/$task.json" --episodes 20 --device cuda:0 --output "$output" --heartbeat "$heartbeat" --resume-log "$log" "${video_args[@]}" >>"$log" 2>&1 &
  EVAL_PID=$!
}

for stage in discovery validation formal; do
  write_state VALIDATING "$stage" evaluate_r13n.py "candidate-native six-task ${stage^}20"
  ORDER=(camera_alignment long_pipeline_delivery take_photo lift_barrier pass_shoe place_food)
  for wave_start in 0 4; do
    EVAL_PIDS=()
    wave_end=$((wave_start+4)); ((wave_end>6)) && wave_end=6
    for ((index=wave_start; index<wave_end; index++)); do
      task="${ORDER[$index]}"; gpu=$((index-wave_start)); EVAL_PID=0; evaluate_task "$stage" "$task" "$gpu"; ((EVAL_PID>0)) && EVAL_PIDS+=("$EVAL_PID")
    done
    CHILDREN=("${EVAL_PIDS[@]}"); codes=0
    for pid in "${EVAL_PIDS[@]}"; do set +e; wait "$pid"; code=$?; set -e; ((code==0)) || codes=$code; done
    CHILDREN=(); ((codes==0)) || exit "$codes"
  done
done

ACCEPTANCE="$RUN_ROOT/acceptance.json"
run_child ACCEPTING acceptance accept_r13n.py "checking checkpoint, validation, 360 native rollouts and runtime health" "$LOG_ROOT/acceptance.log" \
  env PYTHONPATH="$FE_ROOT" "$PYTHON" "$FE_ROOT/scripts/before_we_act/accept_r13n.py" --run-root "$RUN_ROOT" --checkpoint "$CHECKPOINT" --offline "$OFFLINE" --seed-protocol "$SEEDS/protocol.json" --output "$ACCEPTANCE"
write_state PASSED complete accept_r13n.py "R13N B6 baseline established; see acceptance.json"
printf 'R13N pipeline complete: %s\n' "$ACCEPTANCE"
