#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${MARS_OPENVLA_RUN_ROOT:-/workspace/bwa_mars_openvla_runs}; STAGE=${MARS_OPENVLA_STAGE:-formal}
PY=${OPENVLA_PYTHON:-/workspace/venvs/openvla/bin/python}; DATA=${MARS_OPENVLA_DATA_ROOT:-/workspace/datasets/mars_control}
OUT="$ROOT/$STAGE/openvla_oft"; RUN_ID=${MARS_OPENVLA_RUN_ID:-openvla7b_mars_control_lora_r32_$STAGE}; mkdir -p "$OUT"
CONFIG=${MARS_OPENVLA_CONFIG:-/workspace/repos/before-we-act/configs/openvla_oft_mars_control_lora_r32_formal_v1.json}
[[ -f "$CONFIG" ]] || { echo "missing frozen OpenVLA/MARS config: $CONFIG" >&2; exit 2; }
TOKEN_FILE=${HF_TOKEN_FILE:-/workspace/.secrets/hf_token}
[[ -s "$TOKEN_FILE" ]] || { echo "missing Hugging Face token file: $TOKEN_FILE" >&2; exit 2; }
export HF_HOME=${HF_HOME:-/workspace/.hf_home} HUGGINGFACE_HUB_TOKEN="$(< "$TOKEN_FILE")" OPENVLA_MARS_CONTROL_ROOT="$DATA" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
cd /workspace/repos/openvla-oft
read_config() { "$PY" - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(str(value) if not isinstance(value, bool) else str(value))
PY
}
if [[ "$STAGE" == "formal" ]]; then
  # The paper-facing contract is authoritative for formal reruns.
  steps=$(read_config optimization.max_steps)
  batch=$(read_config optimization.per_device_batch_size)
  accum=$(read_config optimization.gradient_accumulation_steps)
  nproc=$(read_config optimization.world_size)
  save_freq=$(read_config checkpoint.save_freq)
  RUN_ID=$(read_config checkpoint.run_id)
else
  steps=${MARS_OPENVLA_MAX_STEPS:-150000}; batch=${MARS_OPENVLA_BATCH_SIZE:-8}; accum=${MARS_OPENVLA_GRAD_ACCUM:-1}; nproc=${MARS_OPENVLA_NPROC:-4}
  save_freq=${MARS_OPENVLA_SAVE_FREQ:-10000}
fi
vla_path=$(read_config source.base_model); dataset_name=$(read_config data.dataset_name)
learning_rate=$(read_config optimization.learning_rate); warmup=$(read_config optimization.lr_warmup_steps); decay_step=$(read_config optimization.num_steps_before_decay)
shuffle_buffer=$(read_config upstream_config_defaults_retained.shuffle_buffer_size)
use_lora=$(read_config lora.enabled); lora_rank=$(read_config lora.rank); lora_dropout=$(read_config lora.dropout); merge_lora=$(read_config lora.merge_lora_during_training)
use_l1=True; use_diffusion=$(read_config model.diffusion.enabled); diffusion_steps=$(read_config model.diffusion.num_diffusion_steps_train); diffusion_freq=$(read_config model.diffusion.sample_frequency)
use_film=$(read_config model.film.enabled); num_images=$(read_config model.num_images_in_input); use_proprio=$(read_config model.use_proprio)
image_aug=False; use_val=False; val_freq=$(read_config upstream_config_defaults_retained.val_freq); val_limit=$(read_config upstream_config_defaults_retained.val_time_limit)
wandb_enabled=False; wandb_log_freq=$(read_config upstream_config_defaults_retained.wandb_log_freq)
printf '%s\n' '{"baseline":"openvla_oft","benchmark":"MARS-Control","status":"training","policy_contract":"shared_weights_decentralized_local_rgb_qpos9_to_local_action8"}' > "$OUT/status.json"
args=(vla-scripts/finetune.py --vla_path="$vla_path" --data_root_dir="$DATA" --dataset_name="$dataset_name" --run_root_dir="$OUT" --run_id_override="$RUN_ID" --shuffle_buffer_size="$shuffle_buffer" --batch_size="$batch" --grad_accumulation_steps="$accum" --learning_rate="$learning_rate" --lr_warmup_steps="$warmup" --num_steps_before_decay="$decay_step" --max_steps="$steps" --save_freq="$save_freq" --save_latest_checkpoint_only=True --use_lora="$use_lora" --lora_rank="$lora_rank" --lora_dropout="$lora_dropout" --use_l1_regression="$use_l1" --use_diffusion="$use_diffusion" --num_diffusion_steps_train="$diffusion_steps" --diffusion_sample_freq="$diffusion_freq" --use_film="$use_film" --num_images_in_input="$num_images" --use_proprio="$use_proprio" --merge_lora_during_training="$merge_lora" --image_aug="$image_aug" --use_val_set="$use_val" --val_freq="$val_freq" --val_time_limit="$val_limit" --wandb_enabled="$wandb_enabled" --wandb_log_freq="$wandb_log_freq")
RUN_DIR="$OUT/$RUN_ID"
if [[ -f "$RUN_DIR/latest_step.json" && -f "$RUN_DIR/action_head--latest_checkpoint.pt" ]]; then
  resume_step=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["step"]))' "$RUN_DIR/latest_step.json")
  if (( resume_step < steps )); then args+=(--resume=True --resume_step="$resume_step" --resume_checkpoint="$RUN_DIR"); fi
fi
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc" "${args[@]}" 2>&1 | tee -a "$OUT/train.log"
test -f "$RUN_DIR/action_head--latest_checkpoint.pt"; ln -sfn "$RUN_DIR" "$OUT/final"
printf '{"baseline":"openvla_oft","benchmark":"MARS-Control","status":"complete","protocol":"decentralized_local_rgb_qpos9_action8","target_steps":%s}\n' "$steps" > "$OUT/status.json"
