#!/usr/bin/env bash
set -e

echo "[1/6] Python / Torch / MuJoCo check"
python - <<'PY'
import torch, mujoco, h5py
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mujoco:", mujoco.__version__)
print("h5py ok")
PY

echo "[2/6] Env smoke test"
PYTHONPATH=. python envs/mujoco_carry_env.py

echo "[3/6] Collect dummy demo"
PYTHONPATH=. python data/collect_dummy.py

echo "[4/6] Replay dummy demo"
PYTHONPATH=. python data/replay.py --demo examples/demo_000.hdf5

echo "[5/6] Dummy train"
PYTHONPATH=. python train/train_dummy.py

echo "[6/6] Pytest"
PYTHONPATH=. pytest -q

echo "Stage 0 check passed."
