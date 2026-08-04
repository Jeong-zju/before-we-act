#!/usr/bin/env bash
set -Eeuo pipefail

destination=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
mkdir -p "$destination"
HF_HOME=/workspace/.hf_home \
HF_HUB_DISABLE_XET=1 \
HF_XET_HIGH_PERFORMANCE=0 \
  /venv/main/bin/hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
    --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
    --local-dir "$destination" \
    --max-workers 1
touch "$destination/.download-complete"
