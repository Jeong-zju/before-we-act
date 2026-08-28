#!/bin/bash
set -euo pipefail

repo=/workspace/repos/care-official
python_bin=/workspace/venvs/mars/bin/python
raw=/workspace/datasets/mars_control/raw
run=/workspace/runs/care_dino_mars
dino=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m
cache=${run}/dino_cache
norm=${run}/mars_norm.json

cd "${repo}"
while true; do
  if "${python_bin}" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")=="PASSED" and x.get("episodes")==600 else 1)' "${cache}/cache_receipt.json" 2>/dev/null; then
    break
  fi
  sleep 15
done

"${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m before_we_act.train_mars_temporal_policy \
  --stage smoke --raw-root "${raw}" --normalization "${norm}" \
  --visual-cache "${cache}" --dino-model "${dino}" \
  --output "${run}/reference_smoke" --updates 2 --workers 2 \
  --save-every 2 --log-every 1

test -f "${run}/reference_smoke/checkpoint_latest.pt"

exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m before_we_act.train_mars_temporal_policy \
  --stage formal --raw-root "${raw}" --normalization "${norm}" \
  --visual-cache "${cache}" --dino-model "${dino}" \
  --output "${run}/reference_formal" --updates 120000 --workers 8 \
  --save-every 5000 --log-every 20
