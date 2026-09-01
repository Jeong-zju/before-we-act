from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import ACTION_LAG_ROWS, EXECUTION_STEPS, HORIZON, IMAGE_SIZE, OBS_STEPS, TASKS


def episode_bounds(episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
    return starts, np.r_[starts[1:], len(episodes)]


def compute_corpus_stats(root: str | Path) -> dict:
    """Compute DP min/max statistics over every causal pair and both arms."""

    root = Path(root)
    q_min = q_max = a_min = a_max = None
    decisions = episodes_total = 0
    for task in TASKS:
        state = np.load(root / task / "state.npy", mmap_mode="r").reshape(-1, 2, 8)
        action = np.load(root / task / "action.npy", mmap_mode="r").reshape(-1, 2, 8)
        episode_ids = np.load(root / task / "episodes.npy", mmap_mode="r")
        starts, ends = episode_bounds(episode_ids)
        episodes_total += len(starts)
        for start, end in zip(starts, ends, strict=True):
            if end - start <= ACTION_LAG_ROWS:
                raise ValueError(f"{task}: episode at row {start} has no causal pair")
            q = np.asarray(state[start : end - ACTION_LAG_ROWS], np.float32).reshape(-1, 8)
            a = np.asarray(action[start + ACTION_LAG_ROWS : end], np.float32).reshape(-1, 8)
            decisions += len(q)
            for name, value in (
                ("q_min", q.min(0)),
                ("q_max", q.max(0)),
                ("a_min", a.min(0)),
                ("a_max", a.max(0)),
            ):
                old = locals()[name]
                updated = value if old is None else (
                    np.minimum(old, value) if name.endswith("min") else np.maximum(old, value)
                )
                if name == "q_min":
                    q_min = updated
                elif name == "q_max":
                    q_max = updated
                elif name == "a_min":
                    a_min = updated
                else:
                    a_max = updated
    result = {
        "q_min": q_min.tolist(),
        "q_max": q_max.tolist(),
        "a_min": a_min.tolist(),
        "a_max": a_max.tolist(),
        "episodes": episodes_total,
        "causal_decisions": decisions,
        # ``decisions`` is already counted after flattening both local arms.
        "indexed_local_samples": decisions,
        "all_550_episodes_no_split": episodes_total == 550,
        "population": "all_causal_pairs_all_11_tasks_all_50_demos_both_local_arms",
        "action_lag_rows": ACTION_LAG_ROWS,
    }
    return result


class DuoDPDataset(Dataset):
    """Three local observations to an eight-row causal absolute-action window."""

    def __init__(
        self,
        root: str | Path,
        *,
        obs_steps: int = OBS_STEPS,
        horizon: int = HORIZON,
        image_size: int = IMAGE_SIZE,
    ):
        self.root = Path(root)
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        if (self.obs_steps, self.horizon, self.image_size) != (OBS_STEPS, HORIZON, IMAGE_SIZE):
            raise ValueError("formal DuoBench DP uses obs=3, horizon=8, image_size=224")
        manifest = json.loads((self.root / "manifest.json").read_text())
        if int(manifest["recording_alignment"]["action_lag_rows"]) != ACTION_LAG_ROWS:
            raise ValueError("prepared dataset does not carry the causal lag-1 contract")
        self.data: list[dict] = []
        self.task_streams: list[list[tuple[int, int, int, int]]] = [[] for _ in TASKS]
        self.task_transition_indices: list[list[tuple[int, int, int, int, int]]] = [
            [] for _ in TASKS
        ]
        self.indexed_local_samples = 0
        for task_id, task in enumerate(TASKS):
            arrays = {
                key: np.load(self.root / task / f"{key}.npy", mmap_mode="r")
                for key in ("state", "action", "head", "left", "right", "episodes")
            }
            starts, ends = episode_bounds(arrays["episodes"])
            if len(starts) != 50:
                raise ValueError(f"{task}: expected all 50 demonstrations")
            self.data.append(arrays)
            for start, end in zip(starts, ends, strict=True):
                for arm in (0, 1):
                    stream = (task_id, arm, int(start), int(end))
                    self.task_streams[task_id].append(stream)
                    self.indexed_local_samples += int(end - start - ACTION_LAG_ROWS)
                    action = arrays["action"].reshape(-1, 2, 8)
                    gripper = np.asarray(action[start:end, arm, 7], np.float32)
                    transitions = start + np.flatnonzero(gripper[1:] != gripper[:-1]) + 1
                    anchors: set[int] = set()
                    for transition in transitions:
                        # DP emits current+1 ... current+6.  Oversample anchors
                        # whose executable chunk contains an open/close event.
                        low = max(int(start), int(transition) - EXECUTION_STEPS)
                        high = min(int(end) - ACTION_LAG_ROWS, int(transition))
                        anchors.update(range(low, high))
                    self.task_transition_indices[task_id].extend(
                        (*stream, current) for current in sorted(anchors)
                    )

    def __len__(self) -> int:
        return self.indexed_local_samples

    @staticmethod
    def _gather(array: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(array[int(position)]) for position in positions])

    def __getitem__(self, index):
        if not (isinstance(index, tuple) and len(index) == 5):
            raise TypeError("DuoDPDataset indices are emitted by TaskEpisodeBatchSampler")
        task_id, arm, episode_start, episode_end, current = index
        data = self.data[task_id]
        observation_positions = np.clip(
            np.arange(current - self.obs_steps + 1, current + 1),
            episode_start,
            episode_end - 1,
        )
        # Row i is a post-action observation.  The action executable after the
        # newest observation is therefore row i+1.  DP exposes horizon index
        # obs_steps-1, so the full target begins two rows earlier plus lag 1.
        action_positions = np.clip(
            np.arange(current - self.obs_steps + 1, current - self.obs_steps + 1 + self.horizon)
            + ACTION_LAG_ROWS,
            episode_start + ACTION_LAG_ROWS,
            episode_end - 1,
        )
        state = data["state"].reshape(-1, 2, 8)
        action = data["action"].reshape(-1, 2, 8)
        head = self._gather(data["head"], observation_positions)
        wrist = self._gather(data["left" if arm == 0 else "right"], observation_positions)
        if head.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3) or wrist.shape != head.shape:
            raise ValueError(f"unexpected prepared RGB shape {head.shape}/{wrist.shape}")
        image = np.concatenate((head, wrist), axis=2)
        qpos = np.asarray(state[observation_positions, arm], np.float32)
        targets = np.asarray(action[action_positions, arm], np.float32)
        task_onehot = np.zeros((self.obs_steps, len(TASKS)), dtype=np.float32)
        task_onehot[:, task_id] = 1.0
        return {
            "head_wrist": torch.from_numpy(image.copy()).permute(0, 3, 1, 2).contiguous(),
            "agent_pos": torch.from_numpy(qpos.copy()),
            "action": torch.from_numpy(targets.copy()),
            "task_id": torch.from_numpy(task_onehot),
        }


class TaskEpisodeBatchSampler(Sampler[list[tuple[int, int, int, int, int]]]):
    """Uniform task, then uniform demonstration/arm stream and causal anchor."""

    def __init__(
        self,
        dataset: DuoDPDataset,
        batch_size: int,
        updates: int,
        seed: int,
        transition_fraction: float = 0.0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)
        self.transition_fraction = float(transition_fraction)
        if not 0.0 <= self.transition_fraction <= 1.0:
            raise ValueError("transition_fraction must be in [0, 1]")
        if self.transition_fraction and any(
            not rows for rows in self.dataset.task_transition_indices
        ):
            missing = [
                TASKS[i]
                for i, rows in enumerate(self.dataset.task_transition_indices)
                if not rows
            ]
            raise ValueError(f"tasks have no gripper-transition anchors: {missing}")
        self.epoch = 0

    def __len__(self) -> int:
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.updates):
            batch = []
            for _sample in range(self.batch_size):
                task_id = rng.randrange(len(TASKS))
                if self.transition_fraction and rng.random() < self.transition_fraction:
                    batch.append(rng.choice(self.dataset.task_transition_indices[task_id]))
                    continue
                stream = rng.choice(self.dataset.task_streams[task_id])
                _task_id, _arm, start, end = stream
                current = rng.randrange(start, end - ACTION_LAG_ROWS)
                batch.append((*stream, current))
            yield batch
