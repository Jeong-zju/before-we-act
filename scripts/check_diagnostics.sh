#!/usr/bin/env bash
set -e

echo "[1/7] Check train/val/test exist"
test -d datasets/stage2/train
test -d datasets/stage2/val
test -d datasets/stage2/test

echo "[2/7] Validate train"
PYTHONPATH=. python data/validate_dataset.py --data_dir datasets/stage2/train

echo "[3/7] Dataset window smoke test"
PYTHONPATH=. python data/dataset.py --data_dir datasets/stage2/train --window 32

echo "[4/7] Generate train diagnostics"
PYTHONPATH=. python data/diagnostics.py --data_dir datasets/stage2/train --out_dir outputs/diagnostics/check_train --num_examples 2

echo "[5/7] Replay one episode"
EP=$(find datasets/stage2/train -name "episode_*.hdf5" | head -n 1)
PYTHONPATH=. python data/replay.py --episode "$EP" --out_dir outputs/replay/check --video 0

echo "[6/7] Filter valid train episodes"
rm -rf datasets/filtered/check_train_valid
PYTHONPATH=. python data/filter_dataset.py --src datasets/stage2/train --dst datasets/filtered/check_train_valid --min_len 16 --copy 0

echo "[7/7] Validate filtered episodes"
PYTHONPATH=. python data/validate_dataset.py --data_dir datasets/filtered/check_train_valid

echo "Diagnostics check passed."
