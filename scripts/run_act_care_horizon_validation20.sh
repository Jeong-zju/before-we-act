#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ACT_REPO_ROOT:-/workspace/bwa-baselines}"
PYTHON_BIN="${ACT_PYTHON:-/venv/robofactory-act/bin/python}"
CHECKPOINT="${ACT_CHECKPOINT:-/workspace/bwa-baselines-runs/formal/act/last.pt}"
DATA_ROOT="${ACT_DATA_ROOT:-/workspace/datasets/robofactory_multitask}"
ROBOFACTORY_ROOT="${ACT_ROBOFACTORY_ROOT:-/workspace/RoboFactory}"
RUN_ROOT="${ACT_RUN_ROOT:-/workspace/bwa-baselines-runs/formal/act_care_horizon_validation20_v1}"
OUTPUT_ROOT="${RUN_ROOT}/closed_loop"
LOG_ROOT="${RUN_ROOT}/logs"
SHARD_ROOT="${RUN_ROOT}/shards"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${SHARD_ROOT}"
export PYTHONPATH="${ROOT}:${ROBOFACTORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

run_task() {
  local task="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/evaluate_act_closed_loop.py \
    --checkpoint "${CHECKPOINT}" \
    --stats-root "${DATA_ROOT}" \
    --config-root "${ROBOFACTORY_ROOT}/robofactory/configs/table" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${task}" \
    --episodes 20 \
    --max-steps-profile care \
    --seed 20260820 \
    --device cuda:0 \
    --cpu-threads 10 \
    --sim-backend cpu \
    --temporal-ensemble-decay 0.01 \
    --formal-six-task \
    >"${LOG_ROOT}/${task}.log" 2>&1
}

run_wave() {
  local item task gpu pid code=0
  local -a pids=()
  for item in "$@"; do
    task="${item%%:*}"
    gpu="${item##*:}"
    run_task "${task}" "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || code=$?
  done
  return "${code}"
}

run_shard() {
  local task="$1" gpu="$2" start="$3" end="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/evaluate_act_closed_loop.py \
    --checkpoint "${CHECKPOINT}" \
    --stats-root "${DATA_ROOT}" \
    --config-root "${ROBOFACTORY_ROOT}/robofactory/configs/table" \
    --output-root "${SHARD_ROOT}/${task}_${start}_${end}" \
    --task "${task}" \
    --episodes 20 \
    --episode-start "${start}" \
    --episode-end "${end}" \
    --max-steps-profile care \
    --seed 20260820 \
    --device cuda:0 \
    --cpu-threads 10 \
    --sim-backend cpu \
    --temporal-ensemble-decay 0.01 \
    --formal-six-task \
    >"${LOG_ROOT}/${task}_${start}_${end}.log" 2>&1
}

shard_pids=()
for item in \
  camera_alignment:1:0:5 camera_alignment:1:5:10 camera_alignment:1:10:15 camera_alignment:1:15:20 \
  long_pipeline_delivery:2:0:5 long_pipeline_delivery:2:5:10 long_pipeline_delivery:2:10:15 long_pipeline_delivery:2:15:20 \
  take_photo:3:0:5 take_photo:3:5:10 take_photo:3:10:15 take_photo:3:15:20; do
  IFS=: read -r task gpu start end <<<"${item}"
  run_shard "${task}" "${gpu}" "${start}" "${end}" &
  shard_pids+=("$!")
done
for pid in "${shard_pids[@]}"; do wait "${pid}"; done

run_wave lift_barrier:1 pass_shoe:2 place_food:3

"${PYTHON_BIN}" - "${SHARD_ROOT}" "${OUTPUT_ROOT}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

shard_root, output_root = map(Path, sys.argv[1:])
for task in ("camera_alignment", "long_pipeline_delivery", "take_photo"):
    payloads = [
        json.loads((shard_root / f"{task}_{start}_{end}" / f"{task}.json").read_text())
        for start, end in ((0, 5), (5, 10), (10, 15), (15, 20))
    ]
    rows = sorted(
        [row for payload in payloads for row in payload["episodes_detail"]],
        key=lambda row: int(row["episode"]),
    )
    if [int(row["episode"]) for row in rows] != list(range(20)):
        raise RuntimeError(f"{task}: shards do not cover episodes 0..19 exactly once")
    if any(row.get("error") for row in rows):
        raise RuntimeError(f"{task}: shard contains failed episodes")
    merged = dict(payloads[0])
    merged.update(
        episodes=20,
        successes=sum(bool(row["success"]) for row in rows),
        success_rate=sum(bool(row["success"]) for row in rows) / 20,
        episode_range=[0, 20],
        episodes_detail=rows,
    )
    destination = output_root / f"{task}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
PY

# Every per-task worker writes its own JSON. This final resumable pass skips
# completed episodes and atomically produces the six-task aggregate summary.
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u scripts/evaluate_act_closed_loop.py \
  --checkpoint "${CHECKPOINT}" \
  --stats-root "${DATA_ROOT}" \
  --config-root "${ROBOFACTORY_ROOT}/robofactory/configs/table" \
  --output-root "${OUTPUT_ROOT}" \
  --episodes 20 \
  --max-steps-profile care \
  --seed 20260820 \
  --device cuda:0 \
  --cpu-threads 10 \
  --sim-backend cpu \
  --temporal-ensemble-decay 0.01 \
  --formal-six-task \
  >"${LOG_ROOT}/aggregate.log" 2>&1

sha256sum "${OUTPUT_ROOT}/summary.json" >"${OUTPUT_ROOT}/summary.json.sha256"
"${PYTHON_BIN}" - "${RUN_ROOT}" "${CHECKPOINT}" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

run_root, checkpoint = map(Path, sys.argv[1:])
payload = {
    "baseline": "act_care_horizon_validation20_v1",
    "checkpoint": str(checkpoint),
    "eval_exit_code": 0,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "gpu": "1,2,3",
    "state": "completed",
    "summary": str(run_root / "closed_loop" / "summary.json"),
}
(run_root / "pipeline_status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
printf 'ACT_CARE_HORIZON_VALIDATION20_COMPLETED summary=%s\n' "${OUTPUT_ROOT}/summary.json"
