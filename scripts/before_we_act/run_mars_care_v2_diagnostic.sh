#!/usr/bin/env bash
set -Eeuo pipefail

# Single-shot, four-GPU CARE scorer-v2 development diagnostic.  This service
# never touches the completed formal run and is intentionally not restartable.
repo=${MARS_CARE_REPO:-/workspace/repos/care-official}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
prepared=${MARS_CARE_PREPARED:-/workspace/runs/care_official_mars_v1/care_prepared.pt}
root=${MARS_CARE_V2_DIAGNOSTIC_ROOT:-/workspace/runs/care_mars_optimization_v2/scorer_family_disjoint}
updates=${MARS_CARE_V2_DIAGNOSTIC_UPDATES:-1000}
seeds=${MARS_CARE_V2_DIAGNOSTIC_SEEDS:-20260901,20260902,20260903}
status=${root}/status.json
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
mkdir -p "${logs}"
cd "${repo}"

conditions=(
  "legacy:100:0:0"
  "full_robust:100:1:0"
  "prefix1_robust_no_ref:1:1:0"
  "prefix1_robust_ref:1:1:1"
)
names=(legacy full_robust prefix1_robust_no_ref prefix1_robust_ref)
pids=()

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${updates}" "${seeds}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
stage, detail, updates, seeds = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
now = datetime.now(timezone.utc).isoformat()
old = {}
if path.exists():
    old = json.loads(path.read_text())
history = list(old.get("history", []))
history.append({"time_utc": now, "stage": stage, "detail": detail})
value = {
    "format_version": "before-we-act.care-mars-v2-diagnostic-status/1",
    "stage": stage,
    "detail": detail,
    "updated_at_utc": now,
    "updates_per_condition_seed": updates,
    "seeds": [int(value) for value in seeds.split(",")],
    "gpu_assignment": {
        "0": "legacy",
        "1": "full_robust",
        "2": "prefix1_robust_no_ref",
        "3": "prefix1_robust_ref",
    },
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

if [[ -f "${status}" ]] && "${python_bin}" - "${status}" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage") == "COMPLETE" else 1)
PY
then
  echo "CARE scorer-v2 diagnostic already complete; preserving ${root}"
  exit 0
fi

if compgen -G "${root}/*.json" >/dev/null; then
  write_status FAILED "refusing to overwrite an incomplete diagnostic root"
  exit 2
fi

write_status RUNNING "four orthogonal conditions x three seeds on four RTX 5090 GPUs"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m scripts.before_we_act.analyze_mars_care_scorer_v2 \
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

write_status COMPLETE "four conditions x three seeds completed; awaiting offline admission audit"

