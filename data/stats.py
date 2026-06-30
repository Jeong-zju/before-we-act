from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_episode_stats(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        actions = f["actions/joint"][:]
        obj = f["global/object_pose"][:]
        force = f["global/force_proxy"][:].reshape(-1)
        contacts = f["global/contacts"][:].reshape(-1)
        robot_distance = f["global/robot_distance"][:].reshape(-1)
        phase = f["labels/phase"][:].reshape(-1)
        rewards = f["rewards/reward"][:].reshape(-1)

        return {
            "file": str(path),
            "success": int(bool(f.attrs.get("success", False))),
            "failure": int(bool(f.attrs.get("failure", False))),
            "failure_reason": str(f.attrs.get("failure_reason", "none")),
            "T": int(actions.shape[0]),
            "total_reward": float(rewards.sum()),
            "max_force": float(force.max()) if len(force) else 0.0,
            "mean_force": float(force.mean()) if len(force) else 0.0,
            "max_contacts": int(contacts.max()) if len(contacts) else 0,
            "min_robot_distance": float(robot_distance.min()) if len(robot_distance) else 0.0,
            "max_robot_distance": float(robot_distance.max()) if len(robot_distance) else 0.0,
            "final_object_x": float(obj[-1, 0]),
            "final_object_y": float(obj[-1, 1]),
            "mean_abs_action": float(np.abs(actions).mean()),
            "num_unique_phases": int(len(np.unique(phase))),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {data_dir}")

    rows = [read_episode_stats(p) for p in tqdm(paths)]
    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{data_dir.name}_stats.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "data_dir": str(data_dir),
        "num_episodes": int(len(df)),
        "success_rate": float(df["success"].mean()),
        "mean_T": float(df["T"].mean()),
        "mean_total_reward": float(df["total_reward"].mean()),
        "mean_max_force": float(df["max_force"].mean()),
        "max_force": float(df["max_force"].max()),
        "mean_min_robot_distance": float(df["min_robot_distance"].mean()),
        "mean_max_robot_distance": float(df["max_robot_distance"].mean()),
        "failure_reasons": df["failure_reason"].value_counts().to_dict(),
    }

    json_path = out_dir / f"{data_dir.name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("saved csv:", csv_path)
    print("saved summary:", json_path)

    plt.figure(figsize=(8, 5))
    plt.hist(df["T"], bins=30)
    plt.xlabel("episode length")
    plt.ylabel("count")
    plt.title(f"{data_dir.name}: episode length")
    plt.tight_layout()
    fig_path = out_dir / f"{data_dir.name}_length_hist.png"
    plt.savefig(fig_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["max_force"], bins=30)
    plt.xlabel("max force proxy")
    plt.ylabel("count")
    plt.title(f"{data_dir.name}: max force proxy")
    plt.tight_layout()
    fig_path2 = out_dir / f"{data_dir.name}_force_hist.png"
    plt.savefig(fig_path2)
    plt.close()

    print("saved figures:", fig_path, fig_path2)


if __name__ == "__main__":
    main()
