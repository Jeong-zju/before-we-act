from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_scalar(path: Path):
    with h5py.File(path, "r") as f:
        final_success = float(f["final_success"][0])
        steps = len(f["actions"])
        max_force = float(np.max(f["force_proxy"][:]))
        mean_force = float(np.mean(f["force_proxy"][:]))
        total_reward = float(np.sum(f["rewards"][:]))
    return final_success, steps, max_force, mean_force, total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="examples/stage1/scripted")
    parser.add_argument("--out", type=str, default="outputs/stage1/stats.png")
    args = parser.parse_args()

    paths = sorted(Path(args.data_dir).glob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {args.data_dir}")

    rows = [read_scalar(p) for p in paths]
    arr = np.asarray(rows, dtype=np.float64)

    success = arr[:, 0]
    steps = arr[:, 1]
    max_force = arr[:, 2]
    mean_force = arr[:, 3]
    total_reward = arr[:, 4]

    print("num episodes:", len(paths))
    print("success rate:", float(success.mean()))
    print("mean steps:", float(steps.mean()))
    print("max force mean:", float(max_force.mean()))
    print("max force max:", float(max_force.max()))
    print("mean reward:", float(total_reward.mean()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(steps, bins=20)
    plt.xlabel("episode length")
    plt.ylabel("count")
    plt.title(f"Stage 1 episode length, success={success.mean():.2f}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print("saved:", out)


if __name__ == "__main__":
    main()
