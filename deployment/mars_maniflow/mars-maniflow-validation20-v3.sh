#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate

repo=/workspace/repos/before-we-act
robofactory=/workspace/repos/RoboFactory
checkpoint=/workspace/runs/mars_maniflow_fixed/formal/last.pt
output=/workspace/runs/mars_maniflow_fixed/formal/validation20_temporal_ensemble_v3
expected_sha256=69a53b432b90aeebd286d7ec445a025264676ab4b81f86a02d5379008d14781e

actual_sha256=$(sha256sum "$checkpoint" | cut -d ' ' -f 1)
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "checkpoint SHA256 mismatch: $actual_sha256" >&2
  exit 1
fi

export PYTHONPATH="$repo:$robofactory:/workspace/repos/ManiFlow_Policy/ManiFlow"
export CUDA_VISIBLE_DEVICES=0
export VK_DRIVER_FILES=/workspace/nvidia-580.159.03/root/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export XDG_RUNTIME_DIR=/tmp/bwa-xdg-runtime
export LD_LIBRARY_PATH="/workspace/nvidia-580.159.03/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

cd "$repo"
exec python -u -m deployment.mars_maniflow.validate \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --robofactory-root "$robofactory" \
  --episodes 20 \
  --device cuda:0 \
  --replan-interval 8 \
  --temporal-ensemble-decay 0.01
