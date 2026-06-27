#!/usr/bin/env bash
set -e

echo "[1/6] Environment smoke test"
PYTHONPATH=. python envs/two_robot_carry_env.py

echo "[2/6] Gym wrapper smoke test"
PYTHONPATH=. python envs/wrappers.py

echo "[3/6] Collect 20 scripted episodes"
rm -rf examples/stage1/check_scripted
PYTHONPATH=. python data/collect_stage1_scripted.py --num_episodes 20 --out_dir examples/stage1/check_scripted --noise_std 0.0

echo "[4/6] Replay first episode"
PYTHONPATH=. python data/replay_stage1.py --demo examples/stage1/check_scripted/episode_000000.hdf5 --plot outputs/stage1/check_replay.png

echo "[5/6] Dataset stats"
PYTHONPATH=. python data/stats_stage1.py --data_dir examples/stage1/check_scripted --out outputs/stage1/check_stats.png

echo "[6/6] Validate success rate"
python - <<'PY'
from pathlib import Path
import h5py
import numpy as np

paths = sorted(Path("examples/stage1/check_scripted").glob("episode_*.hdf5"))
success = []
for p in paths:
    with h5py.File(p, "r") as f:
        success.append(float(f["final_success"][0]))
rate = float(np.mean(success))
print("success_rate:", rate)
if rate < 0.70:
    raise RuntimeError(f"Stage 1 scripted success rate too low: {rate:.3f}")
print("Stage 1 check passed.")
PY
