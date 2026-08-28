#!/usr/bin/env bash
set -Eeuo pipefail

RUN_NAME=${RDT_RUN_NAME:-formal}
ROOT=/workspace/bwa_rdt_runs/$RUN_NAME/rdt_1b
RDT_REPO=/workspace/repos/rdt-1b
RDT_PYTHON=/workspace/venvs/rdt/bin/python
OUTPUT_DIR=${RDT_OUTPUT_DIR:-$ROOT/checkpoints}
TARGET_STEPS=${RDT_MAX_TRAIN_STEPS:-300000}
mkdir -p "$ROOT" "$OUTPUT_DIR"
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled DS_SKIP_CUDA_CHECK=1
export RDT_ROBOFACTORY_DATASET=/workspace/datasets/robofactory_multitask
export RDT_OUTPUT_DIR="$OUTPUT_DIR" RDT_GC_PARENT_PID=$$

status() {
  "$RDT_PYTHON" - "$ROOT/status.json" "$1" "$TARGET_STEPS" <<'PY'
import json,os,sys,tempfile
p,s,n=sys.argv[1:]; os.makedirs(os.path.dirname(p),exist_ok=True)
fd,t=tempfile.mkstemp(dir=os.path.dirname(p),prefix='status.')
with os.fdopen(fd,'w') as f:
 json.dump({'schema':'bwa.rdt.train.v1','baseline':'rdt_1b','status':s,'target_steps':int(n),
            'gpus':[0,1,2,3],'optimizer_scope':'all_rdt_parameters',
            'protocol':'decentralized_local_rgb_qpos_action'},f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
}

cd "$RDT_REPO"
status training
resume=()
latest=""
for candidate in "$OUTPUT_DIR"/checkpoint-*; do
  [[ -d "$candidate" && "${candidate##*-}" =~ ^[0-9]+$ ]] || continue
  [[ -s "$candidate/pytorch_model.bin" && -s "$candidate/ema/model.safetensors" ]] || continue
  if [[ -z "$latest" || "${candidate##*-}" -gt "${latest##*-}" ]]; then latest=$candidate; fi
done
[[ -z "$latest" ]] || resume=(--resume_from_checkpoint="$latest")

"$RDT_PYTHON" /workspace/repos/before-we-act/deployment/rdt_local/rdt_checkpoint_gc.py \
  >> "$ROOT/checkpoint_gc.log" 2>&1 &
gc_pid=$!
cleanup() { kill "$gc_pid" 2>/dev/null || true; wait "$gc_pid" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

set +e
"$RDT_PYTHON" -m deepspeed.launcher.runner --num_gpus=4 main.py \
  --deepspeed=./configs/zero2.json \
  --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b \
  --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 \
  --output_dir="$OUTPUT_DIR" --train_batch_size=${RDT_MICRO_BATCH:-4} --sample_batch_size=1 \
  --max_train_steps="$TARGET_STEPS" --checkpointing_period=${RDT_CHECKPOINT_PERIOD:-2500} \
  --checkpoints_total_limit=2 --lr_scheduler=constant --learning_rate=1e-4 --mixed_precision=bf16 \
  --dataloader_num_workers=${RDT_NUM_WORKERS:-4} --dataset_type=finetune --state_noise_snr=40 \
  --load_from_hdf5 --precomp_lang_embed --set_grads_to_none --allow_tf32 --report_to=tensorboard \
  "${resume[@]}" 2>&1 | tee -a "$ROOT/train.log"
train_rc=${PIPESTATUS[0]}
set -e
(( train_rc == 0 )) || { status failed; exit "$train_rc"; }
[[ -s "$OUTPUT_DIR/pytorch_model.bin" && -s "$OUTPUT_DIR/ema/model.safetensors" ]]
status complete
