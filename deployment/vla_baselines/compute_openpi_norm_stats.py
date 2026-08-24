#!/usr/bin/env python3
"""Compute exact local-only π0.5 normalization statistics without decoding RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/workspace/datasets/robofactory_multitask")
    parser.add_argument(
        "--output-dir",
        default="/workspace/bwa_vla_runs/openpi/assets/pi05_robofactory_lora/robofactory",
    )
    parser.add_argument("--action-horizon", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    episodes = streams = 0
    for task_dir in sorted(root.iterdir()):
        manifest_path = task_dir / "training_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for episode in manifest.get("episodes", []):
            path = task_dir / episode["hdf5_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as h:
                n = int(h["data/observation/agents/panda_0/qpos"].shape[0])
                done = np.asarray(h["data/done"][:n], bool)
                first_done = np.flatnonzero(done)
                if len(first_done):
                    n = int(first_done[0] + 1)
                agents = sorted(int(k.rsplit("_", 1)[1]) for k in h["data/observation/agents"])
                for agent in agents:
                    qpos = np.asarray(h[f"data/observation/agents/panda_{agent}/qpos"][:n], np.float32)
                    action = np.asarray(h[f"data/action/agents/panda_{agent}/commanded"][:n], np.float32)
                    # For every state t, form the future absolute target chunk and
                    # convert its seven arm joints to deltas from qpos[t].
                    indices = np.minimum(
                        np.arange(n)[:, None] + np.arange(args.action_horizon)[None, :], n - 1
                    )
                    chunks = action[indices]
                    chunks[..., :7] -= qpos[:, None, :7]
                    state_rows.append(qpos)
                    action_rows.append(chunks.reshape(-1, 8))
                    streams += 1
            episodes += 1

    if episodes != 900:
        raise RuntimeError(f"Expected all 900 episodes, found {episodes}")
    states = np.concatenate(state_rows, axis=0)
    actions = np.concatenate(action_rows, axis=0)

    def stats(values: np.ndarray) -> dict:
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
        return {
            "mean": values.mean(0).astype(float).tolist(),
            "std": values.std(0).astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }

    output = Path(args.output_dir) / "norm_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"norm_stats": {"state": stats(states), "actions": stats(actions)}}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "episodes": episodes, "streams": streams,
                      "state_rows": len(states), "action_rows": len(actions), "output": str(output)}))


if __name__ == "__main__":
    main()
