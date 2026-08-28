"""Strict decentralized, all-episode RoboFactory adapter for RDT-1B."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """Expose one independent stream for every task/episode/local arm."""

    def __init__(self) -> None:
        root = Path(os.environ.get("RDT_ROBOFACTORY_DATASET", "/workspace/datasets/robofactory_multitask"))
        self.DATASET_NAME = "robofactory"
        with open("configs/base.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.CHUNK_SIZE = int(config["common"]["action_chunk_size"])
        self.IMG_HISTORY_SIZE = int(config["common"]["img_history_size"])
        self.STATE_DIM = int(config["common"]["state_dim"])
        self.file_paths: list[tuple[str, int, str]] = []
        self._episode_lengths: dict[str, int] = {}
        expected = found = 0
        for task_dir in sorted(root.iterdir()):
            manifest_path = task_dir / "training_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            for episode in manifest.get("episodes", []):
                expected += 1
                path = task_dir / episode["hdf5_path"]
                if not path.is_file():
                    continue
                with h5py.File(path, "r") as handle:
                    agents = sorted(int(key.rsplit("_", 1)[1]) for key in handle["data/observation/agents"])
                    n = int(handle[f"data/observation/agents/panda_{agents[0]}/qpos"].shape[0])
                    done = np.asarray(handle["data/done"][:n], bool)
                    terminal = np.flatnonzero(done)
                    if len(terminal):
                        n = int(terminal[0] + 1)
                if n < 1:
                    raise RuntimeError(f"empty episode: {path}")
                found += 1
                self._episode_lengths[str(path)] = n
                self.file_paths.extend((str(path), agent, task_dir.name) for agent in agents)
        if os.environ.get("RDT_ALLOW_INCOMPLETE_DATASET") != "1" and (expected != 900 or found != 900):
            raise RuntimeError(f"formal RDT training requires 6x150 episodes; found {found}/{expected}")
        if not self.file_paths:
            raise RuntimeError(f"no RoboFactory HDF5 streams under {root}")

        self._episodes = sorted({path for path, _, _ in self.file_paths})
        self._tasks = sorted({task for _, _, task in self.file_paths})
        self._task_episodes = {
            task: sorted({path for path, _, row_task in self.file_paths if row_task == task})
            for task in self._tasks
        }
        self._episode_to_items = {
            path: [index for index, row in enumerate(self.file_paths) if row[0] == path]
            for path in self._episodes
        }
        self._task_weights = {}
        for task, episodes in self._task_episodes.items():
            weights = np.asarray([self._episode_lengths[path] for path in episodes], np.float64)
            self._task_weights[task] = weights / weights.sum()

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    @staticmethod
    def _instruction(handle: h5py.File) -> str:
        value = handle["data/task/text"][0]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def _indices(self) -> list[int]:
        return [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)] + [
            STATE_VEC_IDX_MAPPING["right_gripper_open"]
        ]

    def _unified(self, values: np.ndarray, *, action: bool) -> np.ndarray:
        output = np.zeros(values.shape[:-1] + (self.STATE_DIM,), np.float32)
        indices = self._indices()
        output[..., indices[:7]] = values[..., :7]
        # RoboFactory commanded action uses [-1, 1]; qpos stores two finger
        # widths in metres.  RDT's sole normalization exception is gripper [0, 1].
        output[..., indices[7]] = (values[..., 7] + 1.0) / 2.0 if action else values[..., 7:9].mean(-1) / 0.04
        return output

    def _trajectory(self, path: str, agent: int) -> dict:
        n = self._episode_lengths[path]
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle[f"data/observation/agents/panda_{agent}/qpos"][:n], np.float32)
            action = np.asarray(handle[f"data/action/agents/panda_{agent}/commanded"][:n], np.float32)
            instruction = self._instruction(handle)
        return {"state": self._unified(qpos, action=False), "action": self._unified(action, action=True),
                "instruction": instruction}

    def get_item(self, index: int | None = None, state_only: bool = False):
        if index is None:
            # Balance tasks first, then sample episodes proportional to usable
            # timesteps, then sample one local arm.  No split label is consulted.
            task = str(np.random.choice(self._tasks))
            episode = str(np.random.choice(self._task_episodes[task], p=self._task_weights[task]))
            index = int(np.random.choice(self._episode_to_items[episode]))
        path, agent, task = self.file_paths[int(index)]
        trajectory = self._trajectory(path, agent)
        if state_only:
            return {"state": trajectory["state"], "action": trajectory["action"]}
        return self._sample(path, agent, task, trajectory)

    def _sample(self, path: str, agent: int, task: str, trajectory: dict) -> dict:
        state = trajectory["state"]
        n = len(state)
        timestep = int(np.random.randint(0, n))
        actions = trajectory["action"][timestep:timestep + self.CHUNK_SIZE]
        if len(actions) < self.CHUNK_SIZE:
            actions = np.concatenate([actions, np.repeat(actions[-1:], self.CHUNK_SIZE - len(actions), 0)], 0)
        with h5py.File(path, "r") as handle:
            frames = np.asarray(handle[f"data/observation/images/agent_{agent}"][
                max(0, timestep - self.IMG_HISTORY_SIZE + 1):timestep + 1
            ], np.uint8)
        valid = len(frames)
        if valid < self.IMG_HISTORY_SIZE:
            frames = np.concatenate([np.repeat(frames[:1], self.IMG_HISTORY_SIZE - valid, 0), frames], 0)
        local_mask = np.zeros(self.IMG_HISTORY_SIZE, bool)
        local_mask[-valid:] = True
        empty = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0), np.uint8)
        indicator = np.zeros(self.STATE_DIM, np.float32)
        indicator[self._indices()] = 1.0
        embed = Path(path).parents[1] / "lang_embed.pt"
        instruction = str(embed) if embed.is_file() else trajectory["instruction"]
        return {
            "meta": {"dataset_name": self.DATASET_NAME, "#steps": n, "step_id": timestep,
                     "instruction": instruction, "task": task, "agent": agent},
            "state": state[timestep:timestep + 1], "state_std": state.std(0).clip(1e-4),
            "state_mean": state.mean(0), "state_norm": np.sqrt(np.mean(state ** 2, axis=0)),
            "actions": actions, "state_indicator": indicator,
            "cam_high": frames, "cam_high_mask": local_mask,
            "cam_right_wrist": empty, "cam_right_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, bool),
            "cam_left_wrist": empty, "cam_left_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, bool),
        }


if __name__ == "__main__":
    dataset = HDF5VLADataset()
    sample = dataset.get_item(0)
    print(json.dumps({"streams": len(dataset), "episodes": len(dataset._episodes), "tasks": dataset._tasks,
                      "state": sample["state"].shape, "actions": sample["actions"].shape,
                      "image": sample["cam_high"].shape}, default=list))
