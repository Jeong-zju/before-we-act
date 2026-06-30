from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.diagnostics import read_episode, plot_episode


def print_episode_summary(path: str | Path):
    path = Path(path)
    with h5py.File(path, "r") as f:
        print("file:", path)
        print("schema_version:", f.attrs.get("schema_version", "unknown"))
        print("success:", f.attrs.get("success", "unknown"))
        print("failure:", f.attrs.get("failure", "unknown"))
        print("failure_reason:", f.attrs.get("failure_reason", "unknown"))
        print("num_steps:", f["actions/joint"].shape[0])
        print("obs/robot_0/proprio:", f["obs/robot_0/proprio"].shape)
        print("obs/robot_1/proprio:", f["obs/robot_1/proprio"].shape)
        print("actions/joint:", f["actions/joint"].shape)
        print("global/object_pose:", f["global/object_pose"].shape)
        print("global/global_state:", f["global/global_state"].shape)


def export_topdown_video(ep: dict, out_path: str | Path, fps: int = 20):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = ep["object_pose"]
    gs = ep["global_state"]
    r0 = gs[:, 0:2]
    r1 = gs[:, 3:5]

    frames = []
    T = len(obj)
    step = max(1, T // 250)

    for t in range(0, T, step):
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.plot(obj[: t + 1, 0], obj[: t + 1, 1], label="object")
        ax.plot(r0[: t + 1, 0], r0[: t + 1, 1], label="robot_0")
        ax.plot(r1[: t + 1, 0], r1[: t + 1, 1], label="robot_1")
        ax.scatter(obj[t, 0], obj[t, 1], s=40)
        ax.axvline(-0.90, linestyle="--", linewidth=1)
        ax.axvline(0.90, linestyle="--", linewidth=1)
        ax.axhspan(2.77, 3.33, alpha=0.2)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 3.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Replay t={t}/{T}")
        ax.legend(loc="upper left")
        fig.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
        frames.append(rgb)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    print("saved video:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/replay")
    parser.add_argument("--video", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    episode_path = Path(args.episode)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_episode_summary(episode_path)
    ep = read_episode(episode_path)

    prefix = episode_path.stem
    plot_episode(ep, out_dir, prefix=prefix)

    if args.video:
        export_topdown_video(ep, out_dir / f"{prefix}_topdown.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
