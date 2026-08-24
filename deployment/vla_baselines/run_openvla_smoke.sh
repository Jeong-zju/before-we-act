#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=${BWA_RUN_ROOT:-/workspace/bwa_vla_runs}
SMOKE_ROOT="$RUN_ROOT/smoke/openvla_oft"
PROFILE="$SMOKE_ROOT/resource_profile.json"
TRAIN=/workspace/repos/before-we-act/deployment/vla_baselines/run_openvla_oft.sh
PYTHON=${OPENVLA_PYTHON:-/workspace/venvs/openvla/bin/python}
mkdir -p "$SMOKE_ROOT"

if [[ -f "$PROFILE" ]]; then
  "$PYTHON" - "$PROFILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "complete"
assert p["world_size"] == 4
assert p["per_device_batch"] >= 1
PY
  exit 0
fi

min_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n1 | tr -d ' ')
if (( min_mib >= 75000 )); then
  candidates=(8 6 4 2 1)
elif (( min_mib >= 39000 )); then
  candidates=(3 2 1)
else
  candidates=(1)
fi

for batch in "${candidates[@]}"; do
  attempt_log="$SMOKE_ROOT/batch-${batch}.attempt.log"
  set +e
  OPENVLA_STAGE=smoke \
  OPENVLA_RUN_ID="openvla7b_robofactory_lora_r32_smoke_b${batch}" \
  OPENVLA_MAX_STEPS=2 \
  OPENVLA_SAVE_FREQ=2 \
  OPENVLA_BATCH_SIZE="$batch" \
  OPENVLA_GRAD_ACCUM=1 \
  OPENVLA_NPROC=4 \
  OPENVLA_MERGE_LORA=True \
    "$TRAIN" 2>&1 | tee "$attempt_log"
  code=${PIPESTATUS[0]}
  set -e
  if (( code == 0 )); then
    "$PYTHON" - "$PROFILE" "$batch" "$min_mib" "$SMOKE_ROOT/final" <<'PY'
import json, os, pathlib, sys, tempfile
path, batch, memory, checkpoint = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
payload = {
    "schema": "bwa.openvla.resource_profile.v1",
    "status": "complete",
    "world_size": 4,
    "per_device_batch": batch,
    "formal_gradient_accumulation": max(1, (8 + batch - 1) // batch),
    "min_gpu_memory_mib": memory,
    "smoke_checkpoint": checkpoint,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
    exit 0
  fi
  if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$attempt_log"; then
    echo "OpenVLA smoke failed for a non-OOM reason; refusing to hide the error" >&2
    exit "$code"
  fi
  failed="$SMOKE_ROOT/failed-batch-${batch}-$(date -u +%Y%m%dT%H%M%SZ)"
  run_dir="$SMOKE_ROOT/openvla7b_robofactory_lora_r32_smoke_b${batch}"
  [[ ! -e "$run_dir" ]] || mv "$run_dir" "$failed"
done

echo "OpenVLA smoke exhausted all safe per-device batch sizes" >&2
exit 1
