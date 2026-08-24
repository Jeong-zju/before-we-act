#!/usr/bin/env bash
set -Eeuo pipefail

STAGE=${OPENVLA_STAGE:-formal}
ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}/${STAGE}/openvla_oft
REPO=${OPENVLA_REPO:-/workspace/repos/openvla-oft}
PYTHON=${OPENVLA_PYTHON:-/workspace/venvs/openvla/bin/python}
BASE=${OPENVLA_BASE:-/workspace/models/openvla-7b}
DATA=${OPENVLA_ROBOFACTORY_ROOT:-/workspace/datasets/robofactory_multitask}
RUN_ID=${OPENVLA_RUN_ID:-openvla7b_robofactory_lora_r32_${STAGE}}
RUN_DIR="$ROOT/$RUN_ID"
TARGET_STEPS=${OPENVLA_MAX_STEPS:-150000}
NPROC=${OPENVLA_NPROC:-4}
GRAD_ACCUM=${OPENVLA_GRAD_ACCUM:-1}
mkdir -p "$ROOT"

export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
export OPENVLA_ROBOFACTORY_ROOT="$DATA"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO"
[[ -x "$PYTHON" ]] || { echo "OpenVLA environment missing: $PYTHON" >&2; exit 78; }
[[ -d "$BASE" ]] || { echo "OpenVLA base snapshot missing: $BASE" >&2; exit 78; }

printf '%s\n' '{"baseline":"openvla_oft","status":"training","protocol":"decentralized_local_rgb_qpos_action"}' > "$ROOT/status.json"

TRAIN_BASE="$BASE"
RESUME_ARGS=()
STEP=0
if [[ -f "$RUN_DIR/latest_step.json" && -f "$RUN_DIR/action_head--latest_checkpoint.pt" ]]; then
  STEP=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$RUN_DIR/latest_step.json")
  if (( STEP < TARGET_STEPS )); then
    RESUME_ARGS=(--resume=True --resume_step="$STEP" --resume_checkpoint="$RUN_DIR")
  fi
fi

ARGS=(
  vla-scripts/finetune.py
  --vla_path="$TRAIN_BASE"
  --data_root_dir="$DATA"
  --dataset_name=robofactory
  --run_root_dir="$ROOT"
  --run_id_override="$RUN_ID"
  --batch_size=${OPENVLA_BATCH_SIZE:-8}
  --grad_accumulation_steps="$GRAD_ACCUM"
  --learning_rate=${OPENVLA_LR:-5e-4}
  --num_steps_before_decay=${OPENVLA_DECAY_STEP:-100000}
  --max_steps="$TARGET_STEPS"
  --save_freq=${OPENVLA_SAVE_FREQ:-10000}
  --save_latest_checkpoint_only=True
  --use_lora=True
  --lora_rank=32
  --use_l1_regression=True
  --use_diffusion=False
  --use_film=False
  --num_images_in_input=1
  --use_proprio=True
  --merge_lora_during_training=${OPENVLA_MERGE_LORA:-True}
  --image_aug=False
  --use_val_set=False
  --wandb_enabled=False
  "${RESUME_ARGS[@]}"
)

if (( STEP < TARGET_STEPS )); then
  "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    "${ARGS[@]}" 2>&1 | tee -a "$ROOT/train.log"
else
  printf 'Checkpoint already reached target: step=%s target=%s\n' "$STEP" "$TARGET_STEPS" | tee -a "$ROOT/train.log"
fi

FINAL="$RUN_DIR"
[[ -f "$FINAL/action_head--latest_checkpoint.pt" ]] || { echo "OpenVLA checkpoint missing" >&2; exit 1; }
rm -f "$ROOT/final"
ln -s "$FINAL" "$ROOT/final"
printf '%s\n' '{"baseline":"openvla_oft","status":"complete","protocol":"decentralized_local_rgb_qpos_action"}' > "$ROOT/status.json"
