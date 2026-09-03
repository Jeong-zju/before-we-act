#!/usr/bin/env bash
set -Eeuo pipefail
RUN_NAME=${RDT_DUO_RUN_NAME:-formal}; RUN=/workspace/runs/rdt_duo/$RUN_NAME; REPO=/workspace/repos/rdt-1b; PY=${RDT_DUO_PYTHON:-/venv/main/bin/python}; OUT=$RUN/checkpoints; TARGET=${RDT_DUO_MAX_TRAIN_STEPS:-300000}
mkdir -p "$RUN" "$OUT"; export PYTHONUNBUFFERED=1 HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)" TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled DS_SKIP_CUDA_CHECK=1 RDT_DUOBENCH_DATA=/workspace/runs/rdt_duo/data
status(){ "$PY" - "$RUN/status.json" "$1" "$TARGET" <<'PY'
import json,os,sys,tempfile
p,s,n=sys.argv[1:]; os.makedirs(os.path.dirname(p),exist_ok=True); fd,t=tempfile.mkstemp(dir=os.path.dirname(p))
with os.fdopen(fd,'w') as f: json.dump({'schema':'duobench.rdt.train.v1','status':s,'target_steps':int(n),'gpus':[0,1,2,3],'optimizer_scope':'all_rdt_parameters','episodes':550,'causal_samples':285438,'local_streams':1100,'protocol':'decentralized_local_rgb_qpos_to_local_absolute_action8'},f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(t,p)
PY
}
cd "$REPO"; status training; resume=(); latest=""
for c in "$OUT"/checkpoint-*; do [[ -d "$c" && "${c##*-}" =~ ^[0-9]+$ && -s "$c/pytorch_model.bin" && -s "$c/ema/model.safetensors" ]] || continue; [[ -z "$latest" || "${c##*-}" -gt "${latest##*-}" ]] && latest=$c; done
[[ -z "$latest" ]] || resume=(--resume_from_checkpoint="$latest")
set +e
"$PY" -m accelerate.commands.launch --num_processes 4 --multi_gpu main.py --deepspeed=./configs/zero2.json --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 --output_dir="$OUT" --train_batch_size=${RDT_DUO_MICRO_BATCH:-4} --sample_batch_size=1 --max_train_steps="$TARGET" --checkpointing_period=${RDT_DUO_CHECKPOINT_PERIOD:-2500} --checkpoints_total_limit=2 --lr_scheduler=constant --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=bf16 --dataloader_num_workers=${RDT_DUO_NUM_WORKERS:-8} --dataset_type=finetune --state_noise_snr=40 --load_from_hdf5 --precomp_lang_embed --set_grads_to_none --allow_tf32 --report_to=tensorboard "${resume[@]}" 2>&1 | tee -a "$RUN/train.log"
rc=${PIPESTATUS[0]}; set -e; ((rc==0)) || { status failed; exit "$rc"; }; [[ -s "$OUT/pytorch_model.bin" && -s "$OUT/ema/model.safetensors" ]]; ln -sfn "$OUT" "$RUN/final"; status complete
