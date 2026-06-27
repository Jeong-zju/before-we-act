from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_episode(path: Path):
    with h5py.File(path, "r") as f:
        ep = {
            "robot_0": f["obs/robot_0"][:],
            "robot_1": f["obs/robot_1"][:],
            "object": f["obs/object"][:],
            "global_state": f["global_state"][:],
            "actions": f["actions"][:],
            "rewards": f["rewards"][:],
            "force_proxy": f["force_proxy"][:],
            "contacts": f["contacts"][:],
            "success": f["success"][:],
            "failure": f["failure"][:],
            "final_success": f["final_success"][:],
        }
    return ep


def plot_episode(ep, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = ep["object"]
    global_state = ep["global_state"]
    a_xy = global_state[:, 0:2]
    b_xy = global_state[:, 3:5]

    plt.figure(figsize=(7, 8))
    plt.plot(obj[:, 0], obj[:, 1], label="object")
    plt.plot(a_xy[:, 0], a_xy[:, 1], label="robot_0")
    plt.plot(b_xy[:, 0], b_xy[:, 1], label="robot_1")
    plt.axhspan(2.77, 3.33, alpha=0.2, label="goal")
    plt.axvline(-0.90, linestyle="--", linewidth=1)
    plt.axvline(0.90, linestyle="--", linewidth=1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.6, 3.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Stage 1 trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, required=True)
    parser.add_argument("--plot", type=str, default="outputs/stage1/replay_plot.png")
    args = parser.parse_args()

    ep = load_episode(Path(args.demo))
    print("demo:", args.demo)
    for k, v in ep.items():
        print(k, v.shape, v.dtype)

    print("final_success:", bool(ep["final_success"][0]))
    print("mean_reward:", float(ep["rewards"].mean()))
    print("max_force_proxy:", float(ep["force_proxy"].max()))
    print("num_steps:", len(ep["actions"]))

    plot_episode(ep, Path(args.plot))
    print("saved plot:", args.plot)


if __name__ == "__main__":
    main()
