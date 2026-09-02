#!/usr/bin/env bash
set -Eeuo pipefail

# Single-shot paired closed-loop recorder.  It observes the completed formal
# policy but never mutates that run, never reads Validation20 for tuning, and
# cannot restart itself after either success or failure.
repo=${MARS_CARE_REPO:-/workspace/repos/care-mars-v2}
python_bin=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
robofactory=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
formal=${MARS_CARE_FORMAL_ROOT:-/workspace/runs/care_official_mars_v1}
reference=${MARS_CARE_REFERENCE_CHECKPOINT:-${formal}/belief_selected/deployment_checkpoint.pt}
care=${MARS_CARE_CHECKPOINT:-${formal}/care_offline/care_deployment_checkpoint.pt}
root=${MARS_CARE_ROLLOUT_DIAGNOSTIC_ROOT:-/workspace/runs/care_mars_optimization_v2/rollout_diagnostic_v1}
status=${root}/status.json
records=${root}/records
results=${root}/results
logs=${root}/logs

export PYTHONPATH=${repo}/stereo_core:${repo}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${logs}" "${results}"
cd "${repo}"

# Fixed before execution and disjoint from Validation20 [20260827,20260846].
tasks=(place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube)
seeds=(20260910 20260911 20260912 20260913)
max_steps=(500 500 1200 800)
modes=(selector_off care)

write_status() {
  local stage=$1 detail=${2:-}
  "${python_bin}" - "${status}" "${stage}" "${detail}" "${reference}" "${care}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
stage, detail = sys.argv[2], sys.argv[3]
reference, care = Path(sys.argv[4]), Path(sys.argv[5])
old = json.loads(path.read_text()) if path.exists() else {}
now = datetime.now(timezone.utc).isoformat()
history = list(old.get("history", []))
history.append({"time_utc": now, "stage": stage, "detail": detail})
def digest(value):
    return hashlib.sha256(value.read_bytes()).hexdigest() if value.is_file() else None
value = {
    "format_version": "before-we-act.care-mars-rollout-diagnostic-status/1",
    "stage": stage,
    "detail": detail,
    "updated_at_utc": now,
    "reference_checkpoint": str(reference),
    "reference_checkpoint_sha256": digest(reference),
    "care_checkpoint": str(care),
    "care_checkpoint_sha256": digest(care),
    "tasks": ["place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube"],
    "seeds": [20260910, 20260911, 20260912, 20260913],
    "validation20_seed_range": [20260827, 20260846],
    "selector_modes": ["selector_off", "care"],
    "observer_only": True,
    "privileged_state_returned_to_policy": False,
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

failed() {
  local code=$?
  write_status FAILED "exit=${code} line=${BASH_LINENO[0]}; recorded prefix retained; no automatic retry" || true
  exit "${code}"
}
trap failed ERR

if [[ ! -x "${python_bin}" || ! -f "${reference}" || ! -f "${care}" ]]; then
  write_status FAILED "missing Python/reference/CARE checkpoint; no automatic retry"
  exit 2
fi
if [[ $("${python_bin}" -c 'import torch; print(torch.cuda.device_count())') -ne 4 ]]; then
  write_status FAILED "expected four visible RTX 5090 GPUs; no automatic retry"
  exit 2
fi
if [[ -f "${status}" ]] && "${python_bin}" - "${status}" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("stage") == "COMPLETE" else 1)
PY
then
  echo "MARS CARE rollout diagnostic already complete; preserving ${root}"
  exit 0
fi
if compgen -G "${results}/*.json" >/dev/null || compgen -G "${records}/*" >/dev/null; then
  write_status FAILED "refusing to overwrite incomplete rollout evidence"
  exit 2
fi

write_status RUNNING "paired selector-off/CARE recorder; globally serial Vulkan on GPU0"
for mode in "${modes[@]}"; do
  for index in 0 1 2 3; do
    task=${tasks[index]}
    seed=${seeds[index]}
    limit=${max_steps[index]}
    output=${results}/${mode}_${task}.json
    log=${logs}/${mode}_${task}.log
    write_status RUNNING "mode=${mode} task=${task} seed=${seed} max_steps=${limit}"
    # SAPIEN/Vulkan is process-global on this instance, so physical rollouts
    # are deliberately serial even though all four GPUs are available.
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m before_we_act.evaluate_mars_care_closed_loop \
      --reference-checkpoint "${reference}" \
      --care-checkpoint "${care}" \
      --task "${task}" \
      --robofactory-root "${robofactory}" \
      --output "${output}" \
      --episodes 1 \
      --seed-start "${seed}" \
      --max-steps "${limit}" \
      --mode "${mode}" \
      --device cuda:0 \
      --render-device cuda:0 \
      --record-root "${records}" \
      --record-fps 20 \
      >"${log}" 2>&1
  done
done

"${python_bin}" - "${results}" "${root}/summary.json" <<'PY'
import json
import os
from pathlib import Path
import sys

source, output = Path(sys.argv[1]), Path(sys.argv[2])
tasks = ["place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube"]
rows = {}
for task in tasks:
    off = json.loads((source / f"selector_off_{task}.json").read_text())["rows"][0]
    care = json.loads((source / f"care_{task}.json").read_text())["rows"][0]
    rows[task] = {
        "selector_off": off,
        "care": care,
        "paired_action_trace_equal": off["action_trace_sha256"] == care["action_trace_sha256"],
    }
summary = {
    "format_version": "before-we-act.care-mars-rollout-diagnostic-summary/1",
    "status": "complete",
    "observer_only": True,
    "privileged_state_returned_to_policy": False,
    "tasks": rows,
    "care_overrides": sum(row["care"]["overrides"] for row in rows.values()),
    "all_paired_action_traces_equal": all(row["paired_action_trace_equal"] for row in rows.values()),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY

write_status COMPLETE "paired four-task MP4/telemetry diagnostic complete; awaiting failure audit"

