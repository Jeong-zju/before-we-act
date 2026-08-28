from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import TASKS


def h5_files(raw_root: Path, task) -> list[Path]:
    root = raw_root / task.env_id / "motionplanning"
    merged = root / f"{task.name}.h5"
    if merged.is_file(): return [merged]
    return sorted(root.glob(f"{task.name}.shard*.h5"))


def trajectory_names(handle: h5py.File) -> list[str]:
    return sorted((key for key in handle if key.startswith("traj_")), key=lambda x: int(x.split("_")[-1]))


def audit_raw(raw_root: Path, expected: int = 150) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "complete", "expected_per_task": expected, "tasks": {}}
    for task in TASKS:
        paths = h5_files(raw_root, task)
        if not paths: raise FileNotFoundError(raw_root / task.env_id / "motionplanning")
        names_all, successes, steps = [], 0, 0
        for path in paths:
          with h5py.File(path, "r") as handle:
            names = trajectory_names(handle); names_all.extend(f"{path.name}:{name}" for name in names)
            for name in names:
                group = handle[name]
                success = np.asarray(group["success"])
                successes += int(bool(success[-1]))
                steps += int(group[f"actions/panda-0"].shape[0])
        if len(names_all) != expected or successes != expected:
            raise RuntimeError(f"{task.name}: expected {expected} successful trajectories, got {len(names_all)}/{successes}")
        report["tasks"][task.name] = {
                "paths": [str(x) for x in paths], "trajectories": len(names_all), "successful": successes,
                "joint_steps": steps, "arms": task.arms, "local_trajectory_equivalents": expected * task.arms,
            }
    report["trajectories_total"] = expected * len(TASKS)
    report["local_trajectory_equivalents"] = expected * sum(task.arms for task in TASKS)
    return report


def compute_normalization(raw_root: Path, output: Path) -> dict[str, Any]:
    action_parts, qpos_parts = [], []
    episodes = steps = 0
    for task in TASKS:
      for path in h5_files(raw_root, task):
        with h5py.File(path, "r") as handle:
            for trajectory in trajectory_names(handle):
                episodes += 1
                group = handle[trajectory]
                for arm in range(task.arms):
                    actions = np.asarray(group[f"actions/panda-{arm}"], dtype=np.float32)
                    qpos = np.asarray(group[f"obs/agent/panda-{arm}/qpos"], dtype=np.float32)
                    actions = actions.reshape(len(actions), -1)
                    qpos = qpos.reshape(len(qpos), -1)
                    n = min(len(actions), len(qpos))
                    # Absolute joint-position targets make "copy the current
                    # qpos" an extremely strong but useless shortcut.  Encode
                    # the seven arm joints as the commanded residual and keep
                    # the gripper command absolute.
                    encoded = actions[:n].copy()
                    encoded[:, :7] -= qpos[:n, :7]
                    action_parts.append(encoded)
                    qpos_parts.append(qpos[:n])
                    steps += n
    actions = np.concatenate(action_parts)
    qpos = np.concatenate(qpos_parts)
    value = {
        "format": "mars-care-normalization-v2-residual", "episodes": episodes, "local_steps": steps,
        "action_encoding": "joint_residual_gripper_absolute",
        "action_mean": actions.mean(0).tolist(), "action_std": np.maximum(actions.std(0), 1e-4).tolist(),
        "qpos_mean": qpos.mean(0).tolist(), "qpos_std": np.maximum(qpos.std(0), 1e-4).tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output)
    return value


class MarsLocalDataset(Dataset):
    """Every successful joint trajectory expanded into arm-local samples; no held-out split."""

    def __init__(self, raw_root: Path, normalization: Path, horizon: int = 16, image_size: int = 224, history: int = 16):
        self.raw_root, self.horizon, self.image_size, self.history = raw_root, horizon, image_size, history
        self.norm = json.loads(normalization.read_text())
        self.action_encoding = self.norm.get("action_encoding")
        if self.action_encoding != "joint_residual_gripper_absolute":
            raise RuntimeError("CARE repair requires residual-action normalization v2")
        self.rows: list[tuple[int, int, str, int, int]] = []
        self.task_rows: list[list[int]] = [[] for _ in TASKS]
        self.task_active_rows: list[list[int]] = [[] for _ in TASKS]
        self.paths: list[Path] = []
        for task_id, task in enumerate(TASKS):
          for path in h5_files(raw_root, task):
            path_id = len(self.paths); self.paths.append(path)
            with h5py.File(path, "r") as handle:
                for trajectory in trajectory_names(handle):
                    length = int(handle[trajectory]["actions/panda-0"].shape[0])
                    for arm in range(task.arms):
                        action_array = np.asarray(handle[trajectory][f"actions/panda-{arm}"], dtype=np.float32)
                        qpos_array = np.asarray(handle[trajectory][f"obs/agent/panda-{arm}/qpos"][:length], dtype=np.float32)
                        moving = np.linalg.norm(action_array[:, :7] - qpos_array[:, :7], axis=1) > 0.01
                        grip_transition = np.abs(action_array[:, 7]) < 0.95
                        grip_transition[1:] |= np.abs(np.diff(action_array[:, 7])) > 0.05
                        active = moving | grip_transition
                        for step in range(length):
                            self.task_rows[task_id].append(len(self.rows))
                            if active[step]: self.task_active_rows[task_id].append(len(self.rows))
                            self.rows.append((task_id, path_id, trajectory, arm, step))
        self._handles: dict[int, h5py.File] = {}
        self.a_mean = np.asarray(self.norm["action_mean"], np.float32)
        self.a_std = np.asarray(self.norm["action_std"], np.float32)
        self.q_mean = np.asarray(self.norm["qpos_mean"], np.float32)
        self.q_std = np.asarray(self.norm["qpos_std"], np.float32)

    def __getstate__(self):
        state = dict(self.__dict__); state["_handles"] = {}; return state

    def _handle(self, path_id: int) -> h5py.File:
        if path_id not in self._handles: self._handles[path_id] = h5py.File(self.paths[path_id], "r", swmr=True)
        return self._handles[path_id]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task_id, path_id, trajectory, arm, step = self.rows[index]
        group = self._handle(path_id)[trajectory]
        actions_ds = group[f"actions/panda-{arm}"]
        qpos_ds = group[f"obs/agent/panda-{arm}/qpos"]
        image = np.asarray(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"][step], dtype=np.uint8)
        if image.ndim == 4: image = image[0]
        length = int(actions_ds.shape[0])
        end = min(length, step + self.horizon)
        actions = np.asarray(actions_ds[step:end], dtype=np.float32).reshape(end - step, -1)
        action_qpos = np.asarray(qpos_ds[step:end], dtype=np.float32).reshape(end - step, -1)
        if self.action_encoding == "joint_residual_gripper_absolute":
            actions[:, :7] -= action_qpos[:, :7]
        if len(actions) < self.horizon:
            actions = np.concatenate((actions, np.repeat(actions[-1:], self.horizon - len(actions), 0)))
        qpos = np.asarray(qpos_ds[min(step, len(qpos_ds) - 1)], dtype=np.float32).reshape(-1)
        future = np.asarray(qpos_ds[min(step + self.horizon, len(qpos_ds) - 1)], dtype=np.float32).reshape(-1)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255)
        image = torch.nn.functional.interpolate(image[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]

        # Legal arm-local history.  Each state is paired with the action that
        # led into it; the first state has no previous action.  Left padding is
        # explicitly masked instead of pretending padded values are history.
        history = np.zeros((self.history, 17), dtype=np.float32)
        history_mask = np.zeros(self.history, dtype=np.float32)
        begin = max(0, step - self.history + 1)
        states = np.asarray(qpos_ds[begin : step + 1], dtype=np.float32).reshape(step + 1 - begin, -1)
        history[-len(states) :, :9] = (states - self.q_mean) / self.q_std
        history_mask[-len(states) :] = 1
        for offset, state_step in enumerate(range(begin, step + 1)):
            if state_step == 0:
                continue
            previous = np.asarray(actions_ds[state_step - 1], dtype=np.float32).reshape(-1).copy()
            if self.action_encoding == "joint_residual_gripper_absolute":
                previous[:7] -= np.asarray(qpos_ds[state_step - 1], dtype=np.float32).reshape(-1)[:7]
            history[-len(states) + offset, 9:] = (previous - self.a_mean) / self.a_std
        return {
            "image": image,
            "qpos": torch.from_numpy((qpos - self.q_mean) / self.q_std),
            "actions": torch.from_numpy((actions - self.a_mean) / self.a_std),
            "future_delta": torch.from_numpy((future - qpos) / self.q_std),
            "history": torch.from_numpy(history),
            "history_mask": torch.from_numpy(history_mask),
            "task_id": torch.tensor(task_id),
        }


class TaskBalancedDistributedSampler(Sampler[int]):
    """Uniform task sampling with deterministic independent DDP streams."""

    def __init__(self, dataset: MarsLocalDataset, rank: int = 0, replicas: int = 1, seed: int = 20260825):
        self.task_rows = dataset.task_rows
        self.task_active_rows = dataset.task_active_rows
        self.rank, self.replicas, self.seed, self.epoch = rank, replicas, seed, 0
        self.num_samples = max(1, len(dataset) // replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * self.replicas + self.rank)
        tasks = torch.randint(len(self.task_rows), (self.num_samples,), generator=generator)
        active = torch.rand(self.num_samples, generator=generator) < 0.5
        values: list[int] = []
        for position_index, task_id in enumerate(tasks.tolist()):
            rows = self.task_active_rows[task_id] if active[position_index] else self.task_rows[task_id]
            position = int(torch.randint(len(rows), (1,), generator=generator))
            values.append(rows[position])
        return iter(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    args = parser.parse_args()
    report = audit_raw(args.raw_root)
    report["normalization"] = compute_normalization(args.raw_root, args.normalization)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
