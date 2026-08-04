import json
import sys
import time
from pathlib import Path

core_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
trial_paths = [Path(item) for item in sys.argv[3:]]
while not all(path.is_file() and path.stat().st_size > 0 for path in trial_paths):
    time.sleep(15)
core = json.loads(core_path.read_text())
trials = [json.loads(path.read_text()) for path in trial_paths]
trial_maps = [{int(row["seed"]): bool(row["success"]) for row in trial["rows"]} for trial in trials]
if not all(set(item) == set(trial_maps[0]) for item in trial_maps[1:]):
    raise ValueError("trial seed sets differ")
recovered = sorted(seed for seed in trial_maps[0] if any(item[seed] for item in trial_maps))
per_seed = {
    str(seed): [int(item[seed]) for item in trial_maps]
    for seed in sorted(trial_maps[0])
}
result = {
    "metric": "Targeted Recovery@3 (not a replacement for frozen SR@1)",
    "targeted_seeds": len(trial_maps[0]),
    "recovered_seeds": len(recovered),
    "recovered_seed_ids": recovered,
    "trial_successes": [trial["successes"] for trial in trials],
    "per_seed_trials": per_seed,
    "frozen_sr1_successes": int(core["successes"]),
    "supplementary_any_success_total": int(core["successes"]) + len(recovered),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
