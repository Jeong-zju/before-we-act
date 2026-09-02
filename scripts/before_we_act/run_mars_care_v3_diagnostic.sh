#!/usr/bin/env bash
set -Eeuo pipefail

# Single-shot four-GPU scorer-v3 diagnostic.  This is an isolated development
# run: it never reads Validation20, never mutates the archived formal run, and
# cannot restart itself after either success or failure.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
prepared=${MARS_CARE_PREPARED:-/workspace/runs/care_official_mars_v1/care_prepared.pt}
root=${MARS_CARE_V3_DIAGNOSTIC_ROOT:-/workspace/runs/care_mars_optimization_v2/scorer_v3_family_disjoint}
updates=${MARS_CARE_V3_DIAGNOSTIC_UPDATES:-1000}
seeds=${MARS_CARE_V3_DIAGNOSTIC_SEEDS:-20260904,20260905,20260906}
status=${root}/status.json
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${logs}"
cd "${repo}"

conditions=(
  "v2_control:0:0:0"
  "slot_only:1:0:0"
  "task_only:0:1:0"
  "slot_task_horizon:1:1:1"
)
names=(v2_control slot_only task_only slot_task_horizon)
pids=()

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${updates}" "${seeds}" "${prepared}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
stage, detail = sys.argv[2], sys.argv[3]
updates, seeds, prepared = int(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
now = datetime.now(timezone.utc).isoformat()
old = json.loads(path.read_text()) if path.exists() else {}
history = list(old.get("history", []))
history.append({"time_utc": now, "stage": stage, "detail": detail})
digest = hashlib.sha256(prepared.read_bytes()).hexdigest() if prepared.is_file() else None
value = {
    "format_version": "before-we-act.care-mars-v3-diagnostic-status/1",
    "stage": stage,
    "detail": detail,
    "updated_at_utc": now,
    "updates_per_condition_seed": updates,
    "seeds": [int(value) for value in seeds.split(",")],
    "prepared_data": str(prepared),
    "prepared_data_sha256": digest,
    "gpu_assignment": {
        "0": "v2_control",
        "1": "slot_only",
        "2": "task_only",
        "3": "slot_task_horizon",
    },
    "care_theory_contract_unchanged": True,
    "legacy_formal_run_unchanged": True,
    "validation20_used_for_tuning": False,
    "automatic_retry": False,
    "history": history,
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
failed() {
  local code=$?
  cleanup
  write_status FAILED "exit=${code} line=${BASH_LINENO[0]}; no automatic retry" || true
  exit "${code}"
}
trap cleanup EXIT
trap failed ERR

if [[ ! -x "${python_bin}" || ! -f "${prepared}" ]]; then
  write_status FAILED "missing Python environment or prepared data; no automatic retry"
  exit 2
fi
if [[ $("${python_bin}" -c 'import torch; print(torch.cuda.device_count())') -ne 4 ]]; then
  write_status FAILED "expected exactly four visible GPUs; no automatic retry"
  exit 2
fi
if [[ -f "${status}" ]] && "${python_bin}" - "${status}" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage") == "COMPLETE" else 1)
PY
then
  echo "CARE scorer-v3 diagnostic already complete; preserving ${root}"
  exit 0
fi
if compgen -G "${root}/*.json" >/dev/null; then
  write_status FAILED "refusing to overwrite an incomplete diagnostic root"
  exit 2
fi

write_status RUNNING "four orthogonal conditions x three preregistered seeds on four RTX 5090 GPUs"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m scripts.before_we_act.analyze_mars_care_scorer_v3 \
    --prepared-data "${prepared}" \
    --output "${root}/${names[gpu]}.json" \
    --conditions "${conditions[gpu]}" \
    --seeds "${seeds}" \
    --updates "${updates}" \
    --batch-size 48 \
    --eval-every 200 \
    --device cuda:0 \
    >"${logs}/${names[gpu]}.log" 2>&1 &
  pids+=("$!")
done

failed_job=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed_job=1
done
pids=()
if [[ "${failed_job}" -ne 0 ]]; then
  write_status FAILED "one or more condition workers failed; no automatic retry"
  exit 1
fi

write_status COMPLETE "v3 diagnostic complete; awaiting offline scorer admission audit"
