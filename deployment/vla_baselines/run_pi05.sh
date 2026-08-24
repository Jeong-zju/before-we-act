#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}/formal/pi05
REPO=${OPENPI_REPO:-/workspace/repos/openpi}
PYTHON=${OPENPI_PYTHON:-$REPO/.venv/bin/python}
DATA=${OPENPI_ROBOFACTORY_ROOT:-/workspace/datasets/robofactory_multitask}
EXP=${OPENPI_EXP_NAME:-decentralized_all150}
CHECKPOINT_DIR="$ROOT/checkpoints/pi05_robofactory_lora/$EXP"
TARGET_STEPS=${OPENPI_MAX_STEPS:-120000}
mkdir -p "$ROOT"

export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export HUGGINGFACE_HUB_TOKEN="$(< /workspace/.secrets/hf_token)"
export OPENPI_ROBOFACTORY_ROOT="$DATA"
export JAX_PLATFORMS=cuda
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

cd "$REPO"
[[ -x "$PYTHON" ]] || { echo "openpi environment missing: $PYTHON" >&2; exit 78; }

printf '%s\n' '{"baseline":"pi05","status":"training","protocol":"decentralized_local_rgb_qpos_action"}' > "$ROOT/status.json"
RESUME_ARGS=()
LATEST=$(find "$CHECKPOINT_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | awk '/^[0-9]+$/' | sort -n | tail -1 || true)
if [[ -n "$LATEST" && -d "$CHECKPOINT_DIR/$LATEST/params" && "$LATEST" -lt "$TARGET_STEPS" ]]; then
  RESUME_ARGS=(--resume)
fi

if [[ -z "$LATEST" || "$LATEST" -lt "$TARGET_STEPS" ]]; then
  "$PYTHON" scripts/train.py pi05_robofactory_lora \
    --checkpoint-base-dir="$ROOT/checkpoints" \
    --exp-name="$EXP" \
    --batch-size=${OPENPI_BATCH_SIZE:-32} \
    --num-workers=${OPENPI_NUM_WORKERS:-0} \
    --num-train-steps="$TARGET_STEPS" \
    --save-interval=${OPENPI_SAVE_INTERVAL:-1000} \
    --keep-period=${OPENPI_KEEP_PERIOD:-10000} \
    --fsdp-devices=4 \
    --no-wandb-enabled \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$ROOT/train.log"
else
  printf 'Checkpoint already reached target: step=%s target=%s\n' "$LATEST" "$TARGET_STEPS" | tee -a "$ROOT/train.log"
fi

LATEST=$(find "$CHECKPOINT_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -n | tail -1 || true)
[[ -n "$LATEST" && -d "$CHECKPOINT_DIR/$LATEST/params" ]] || { echo "π0.5 checkpoint missing" >&2; exit 1; }
rm -f "$ROOT/final"
ln -s "$CHECKPOINT_DIR/$LATEST" "$ROOT/final"
printf '%s\n' '{"baseline":"pi05","status":"complete","protocol":"decentralized_local_rgb_qpos_action"}' > "$ROOT/status.json"
