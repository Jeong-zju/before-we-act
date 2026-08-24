#!/usr/bin/env python3
"""Fail-fast audit for the six-task, all-episode decentralized contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

TASKS = ("lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo", "pass_shoe", "place_food")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    root = Path(os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
    rows = {}
    total_episodes = total_streams = total_timesteps = 0
    for task in TASKS:
        task_root = root / task
        receipt = json.loads((task_root / "download_receipt.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("episodes_total") != 150:
            raise RuntimeError(f"invalid download receipt: {task}")
        manifest = json.loads((task_root / "training_manifest.json").read_text())
        episodes = manifest.get("episodes", [])
        if len(episodes) != 150:
            raise RuntimeError(f"{task}: expected 150 episodes, got {len(episodes)}")
        task_streams = task_timesteps = 0
        for episode in episodes:
            path = task_root / episode["hdf5_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as handle:
                agents = sorted(handle["data/observation/agents"])
                n = int(handle[f"data/observation/agents/{agents[0]}/qpos"].shape[0])
                done = np.asarray(handle["data/done"][:n], bool)
                first_done = np.flatnonzero(done)
                if len(first_done):
                    n = int(first_done[0] + 1)
                for key in agents:
                    agent = key.rsplit("_", 1)[1]
                    qpos = handle[f"data/observation/agents/{key}/qpos"]
                    image = handle[f"data/observation/images/agent_{agent}"]
                    action = handle[f"data/action/agents/panda_{agent}/commanded"]
                    if qpos.shape[-1] != 9 or action.shape[-1] != 8 or len(qpos) != len(image) or len(qpos) != len(action):
                        raise RuntimeError(f"local contract mismatch: {path} agent {agent}")
                task_streams += len(agents)
                task_timesteps += n * len(agents)
        rows[task] = {"episodes": 150, "local_agent_streams": task_streams, "local_timesteps": task_timesteps}
        total_episodes += len(episodes)
        total_streams += task_streams
        total_timesteps += task_timesteps
    payload = {
        "schema": "bwa.vla.training_contract.v1",
        "status": "complete",
        "tasks": rows,
        "episodes": total_episodes,
        "local_agent_streams": total_streams,
        "local_timesteps": total_timesteps,
        "training_split": "all_episodes_ignore_manifest_split",
        "input_fields": ["data/observation/images/agent_i", "data/observation/agents/panda_i/qpos"],
        "output_fields": ["data/action/agents/panda_i/commanded"],
        "forbidden_fields": ["global", "peer", "joint_concatenation"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if total_episodes != 900:
        raise RuntimeError(f"expected 900 episodes, got {total_episodes}")
    atomic_json(Path("/workspace/bwa_vla_runs/audit/training_contract.json"), payload)


if __name__ == "__main__":
    main()
