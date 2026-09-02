#!/usr/bin/env bash
set -Eeuo pipefail
RUN_NAME=${RDT_RUN_NAME:-formal}; ROOT=/workspace/runs/rdt_mars/$RUN_NAME; REPO=/workspace/repos/rdt-1b; PY=/workspace/venvs/rdt/bin/python; OUT=$ROOT/checkpoints; TARGET=${RDT_MAX_TRAIN_STEPS:-300000}
mkdir -p "$ROOT" "$OUT"; export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_TOKEN="$(</workspace/.secrets/hf_token)" TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled DS_SKIP_CUDA_CHECK=1 RDT_MARS_DATASET=/workspace/datasets/mars_control RDT_OUTPUT_DIR="$OUT" RDT_GC_PARENT_PID=$$
status(){ "$PY" - "$ROOT/status.json" "$1" "$TARGET" <<'PY'
import json,os,sys,tempfile
p,s,n=sys.argv[1:]; os.makedirs(os.path.dirname(p),exist_ok=True); fd,t=tempfile.mkstemp(dir=os.path.dirname(p))
with os.fdopen(fd,'w') as f: json.dump({'schema':'mars-control.rdt.train.v1','status':s,'target_steps':int(n),'gpus':[0,1,2,3],'optimizer_scope':'all_rdt_parameters','episodes':600,'local_streams':1650,'protocol':'decentralized_local_rgb_qpos_absolute_action'},f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
}
cd "$REPO"; status training; resume=(); latest=""
for c in "$OUT"/checkpoint-*; do [[ -d "$c" && "${c##*-}" =~ ^[0-9]+$ && -s "$c/pytorch_model.bin" && -s "$c/ema/model.safetensors" ]] || continue; [[ -z "$latest" || "${c##*-}" -gt "${latest##*-}" ]] && latest=$c; done
[[ -z "$latest" ]] || resume=(--resume_from_checkpoint="$latest")
"$PY" /workspace/repos/before-we-act/deployment/rdt_mars/rdt_checkpoint_gc.py >>"$ROOT/checkpoint_gc.log" 2>&1 & gc=$!; trap 'kill "$gc" 2>/dev/null || true; wait "$gc" 2>/dev/null || true' EXIT TERM INT
set +e
"$PY" -m deepspeed.launcher.runner --num_gpus=4 main.py --deepspeed=./configs/zero2.json --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 --output_dir="$OUT" --train_batch_size=${RDT_MICRO_BATCH:-4} --sample_batch_size=1 --max_train_steps="$TARGET" --checkpointing_period=${RDT_CHECKPOINT_PERIOD:-2500} --checkpoints_total_limit=2 --lr_scheduler=constant --learning_rate=1e-4 --mixed_precision=bf16 --dataloader_num_workers=${RDT_NUM_WORKERS:-4} --dataset_type=finetune --state_noise_snr=40 --load_from_hdf5 --precomp_lang_embed --set_grads_to_none --allow_tf32 --report_to=tensorboard "${resume[@]}" 2>&1 | tee -a "$ROOT/train.log"
rc=${PIPESTATUS[0]}; set -e; ((rc==0)) || { status failed; exit "$rc"; }; [[ -s "$OUT/pytorch_model.bin" && -s "$OUT/ema/model.safetensors" ]]; status complete
