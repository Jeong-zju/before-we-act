#!/usr/bin/env bash
set -Eeuo pipefail

python_env=/venv/robofactory-act
repo=/workspace/RoboFactory
commit=5868242322414a91454e22f1dd9641f613ba1bcf

if [[ ! -x "$python_env/bin/python" ]]; then
  uv python install 3.10
  uv venv --python 3.10 "$python_env"
fi

if [[ ! -d "$repo/.git" ]]; then
  git clone https://github.com/MARS-EAI/RoboFactory.git "$repo"
fi

if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
  printf 'RoboFactory has tracked changes; refusing to change revisions.\n' >&2
  exit 3
fi
git -C "$repo" fetch origin "$commit"
git -C "$repo" checkout --detach "$commit"

uv pip install --python "$python_env/bin/python" \
  --prerelease=allow \
  -r "$repo/robofactory/requirements.txt"
uv pip install --python "$python_env/bin/python" --no-deps --editable "$repo"
uv pip install --python "$python_env/bin/python" \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  --index https://download.pytorch.org/whl/cu128
uv pip install --python "$python_env/bin/python" \
  'transformers==5.14.1' \
  'huggingface_hub==1.25.1' \
  'hf_xet==1.5.2' \
  'h5py==3.16.0' \
  'numpy==1.26.4' \
  'scipy==1.15.3' \
  'pillow==12.2.0' \
  'tqdm==4.70.0'

cd "$repo/robofactory"
asset_attempt=1
asset_delay=30
while (( asset_attempt <= 8 )); do
  if HF_HOME=/workspace/.hf_home \
      HF_HUB_DISABLE_XET=1 \
      HF_XET_HIGH_PERFORMANCE=0 \
      "$python_env/bin/python" script/download_assets.py; then
    break
  fi
  if (( asset_attempt == 8 )); then
    printf 'RoboFactory asset download failed after 8 attempts.\n' >&2
    exit 1
  fi
  printf 'RoboFactory asset download failed; retrying in %d seconds.\n' "$asset_delay" >&2
  sleep "$asset_delay"
  asset_attempt=$((asset_attempt + 1))
  asset_delay=$((asset_delay * 2))
  (( asset_delay > 300 )) && asset_delay=300
done

"$python_env/bin/python" - <<'PY'
import h5py
import numpy
import torch
import torchvision
import transformers
import robofactory

assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.cuda.is_available()
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "h5py": h5py.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY
