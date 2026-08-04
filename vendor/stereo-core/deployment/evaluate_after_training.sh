#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/workspace/no_wrist_stereo_core
run_root=/workspace/runs/no_wrist_stereo_core_120k
checkpoint="$run_root/checkpoint_120000.pt"
eval_root="$run_root/frozen100"
mkdir -p "$eval_root"

while [[ ! -f "$checkpoint" ]]; do
  if ! pgrep -f '/workspace/no_wrist_stereo_core/stereo_core/train_no_wrist_pair.py' >/dev/null; then
    printf '[%s] formal trainer is absent; resuming from latest checkpoint.\n' "$(date -u +%FT%TZ)"
    OUTPUT="$run_root" WORKERS=8 "$repo_root/deployment/train_no_wrist_core.sh" \
      2>&1 | tee -a /workspace/logs/no_wrist_train_formal.log
  fi
  printf '[%s] waiting for checkpoint_120000.pt\n' "$(date -u +%FT%TZ)"
  sleep 60
done

export PYTHONPATH="$repo_root/stereo_core${PYTHONPATH:+:$PYTHONPATH}"
tasks=(lift_barrier camera_alignment three_robots_stack_cube long_pipeline_delivery take_photo)
for task in "${tasks[@]}"; do
  output="$eval_root/$task.json"
  log="$eval_root/$task.log"
  if [[ -f "$output" ]] && /venv/robofactory-act/bin/python - "$output" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("episodes") == 100 and len(payload.get("rows", [])) == 100 else 1)
PY
  then
    printf '[%s] preserving completed evaluation: %s\n' "$(date -u +%FT%TZ)" "$task"
    continue
  fi
  attempt=1
  while (( attempt <= 3 )); do
    printf '[%s] evaluating %s attempt %d/3\n' "$(date -u +%FT%TZ)" "$task" "$attempt"
    if /venv/robofactory-act/bin/python -u \
        "$repo_root/stereo_core/evaluate_no_wrist_pair.py" \
        --checkpoint "$checkpoint" \
        --task "$task" \
        --seed-file "$repo_root/protocol/frozen100/$task.json" \
        --episodes 100 \
        --max-steps 1500 \
        --device cuda:0 \
        --resume-log "$log" \
        --output "$output" \
        2>&1 | tee -a "$log"; then
      break
    fi
    if (( attempt == 3 )); then
      printf 'evaluation failed after 3 attempts: %s\n' "$task" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    sleep 30
  done
done

/venv/robofactory-act/bin/python - "$eval_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = ("lift_barrier", "camera_alignment", "three_robots_stack_cube", "long_pipeline_delivery", "take_photo")
results = {task: json.loads((root / f"{task}.json").read_text()) for task in tasks}
summary = {
    "task_order": list(tasks),
    "successes": {task: results[task]["successes"] for task in tasks},
    "success_rates": {task: results[task]["success_rate"] for task in tasks},
    "macro_success_rate": sum(results[task]["success_rate"] for task in tasks) / len(tasks),
    "peer_stereo_core_successes": {
        "lift_barrier": 99,
        "camera_alignment": 100,
        "three_robots_stack_cube": 99,
        "long_pipeline_delivery": 94,
        "take_photo": 29,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
PY

touch "$eval_root/.complete"
printf '[%s] frozen100 evaluation complete\n' "$(date -u +%FT%TZ)"
