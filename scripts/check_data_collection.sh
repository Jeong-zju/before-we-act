#!/usr/bin/env bash
set -e

echo "[1/9] Environment smoke test"
PYTHONPATH=. python envs/two_robot_carry_env.py

echo "[2/9] Debug noisy policy"
PYTHONPATH=. python scripts/debug_noisy_policy.py

echo "[3/9] Collect scripted check data"
rm -rf datasets/stage2/check_scripted
PYTHONPATH=. python data/collect.py --mode scripted --num_episodes 20 --out_dir datasets/stage2/check_scripted --seed_start 70000 --noise_std 0.0 --randomize 1

echo "[4/9] Validate scripted check data"
PYTHONPATH=. python data/validate_dataset.py --data_dir datasets/stage2/check_scripted

echo "[5/9] Collect noisy check data"
rm -rf datasets/stage2/check_noisy
PYTHONPATH=. python data/collect.py --mode noisy --num_episodes 30 --out_dir datasets/stage2/check_noisy --seed_start 71000 --noise_std 10.0 --randomize 1

echo "[6/9] Validate noisy check data"
PYTHONPATH=. python data/validate_dataset.py --data_dir datasets/stage2/check_noisy

echo "[7/9] Assert noisy is not all-success"
python - <<'PY'
from pathlib import Path
import h5py
import numpy as np

paths = sorted(Path("datasets/stage2/check_noisy").glob("episode_*.hdf5"))
success = []
reasons = {}
for p in paths:
    with h5py.File(p, "r") as f:
        s = int(bool(f.attrs.get("success", False)))
        r = str(f.attrs.get("failure_reason", "none"))
        success.append(s)
        reasons[r] = reasons.get(r, 0) + 1
rate = float(np.mean(success))
print("noisy_success_rate:", rate)
print("reasons:", reasons)
if rate >= 0.99:
    raise RuntimeError("Noisy dataset is still almost all-success. Noise/failure mechanism is broken.")
PY

echo "[8/9] Split check data"
rm -rf datasets/stage2/check_split
PYTHONPATH=. python data/split_dataset.py --sources datasets/stage2/check_scripted datasets/stage2/check_noisy --out_root datasets/stage2/check_split --train_ratio 0.8 --val_ratio 0.1 --seed 0 --copy 1

echo "[9/9] Dataset window smoke test"
PYTHONPATH=. python data/dataset.py --data_dir datasets/stage2/check_split/train --window 16

echo "Stage 2 check passed."
