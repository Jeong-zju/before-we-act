from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PlanSampleSpec:
    file_idx: int
    start: int
    agent_id: int


class PlanWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        horizon: int = 16,
        stride: int = 1,
        include_failures: bool = True,
        max_episodes: int = -1,
    ):
        self.data_dir = Path(data_dir)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.include_failures = bool(include_failures)

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if max_episodes > 0:
            self.paths = self.paths[:max_episodes]

        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index: List[PlanSampleSpec] = []

        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                T = int(f["actions/joint"].shape[0])
                success = bool(f.attrs.get("success", False))
                failure = bool(f.attrs.get("failure", False))

            if failure and not self.include_failures:
                continue

            # Need a full future segment [start, start + horizon).
            max_start = T - self.horizon
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, self.stride):
                self.index.append(PlanSampleSpec(file_idx=file_idx, start=start, agent_id=0))
                self.index.append(PlanSampleSpec(file_idx=file_idx, start=start, agent_id=1))

        if not self.index:
            raise RuntimeError(
                f"No valid future windows in {self.data_dir}. "
                f"Try reducing horizon={self.horizon}."
            )

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _extract_agent_trajectory(global_state: np.ndarray, object_pose: np.ndarray, agent_id: int) -> np.ndarray:
        if agent_id == 0:
            robot_pose = global_state[:, 0:3]
        elif agent_id == 1:
            robot_pose = global_state[:, 3:6]
        else:
            raise ValueError(f"agent_id must be 0 or 1, got {agent_id}")

        obj_xy = object_pose[:, 0:2]
        traj = np.concatenate([robot_pose, obj_xy], axis=-1)
        return traj.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.index[idx]
        path = self.paths[spec.file_idx]
        start = spec.start
        end = start + self.horizon
        agent_id = spec.agent_id

        with h5py.File(path, "r") as f:
            if agent_id == 0:
                actions = f["actions/robot_0"][start:end]
            else:
                actions = f["actions/robot_1"][start:end]

            global_state = f["global/global_state"][start:end]
            object_pose = f["global/object_pose"][start:end]
            phase = f["labels/phase"][start:end].reshape(-1)
            force = f["global/force_proxy"][start:end].reshape(-1)
            contacts = f["global/contacts"][start:end].reshape(-1)
            success = float(bool(f.attrs.get("success", False)))
            failure = float(bool(f.attrs.get("failure", False)))

        trajectory = self._extract_agent_trajectory(global_state, object_pose, agent_id)

        return {
            "actions": torch.tensor(actions, dtype=torch.float32),
            "trajectory": torch.tensor(trajectory, dtype=torch.float32),
            "phase": torch.tensor(phase, dtype=torch.long),
            "force_proxy": torch.tensor(force, dtype=torch.float32),
            "contacts": torch.tensor(contacts, dtype=torch.float32),
            "agent_id": torch.tensor(agent_id, dtype=torch.long),
            "success": torch.tensor(success, dtype=torch.float32),
            "failure": torch.tensor(failure, dtype=torch.float32),
        }


def compute_plan_normalization(data_dir: str, horizon: int = 16, max_samples: int = 20000) -> Dict[str, torch.Tensor]:
    ds = PlanWindowDataset(data_dir=data_dir, horizon=horizon, stride=max(1, horizon // 2))
    n = min(len(ds), max_samples)

    actions = []
    trajectories = []

    for i in range(n):
        sample = ds[i]
        actions.append(sample["actions"])
        trajectories.append(sample["trajectory"])

    a = torch.stack(actions, dim=0).reshape(-1, 4)
    x = torch.stack(trajectories, dim=0).reshape(-1, 5)

    stats = {
        "action_mean": a.mean(dim=0),
        "action_std": a.std(dim=0).clamp_min(1e-6),
        "traj_mean": x.mean(dim=0),
        "traj_std": x.std(dim=0).clamp_min(1e-6),
    }
    return stats


def save_normalization(stats: Dict[str, torch.Tensor], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    print("saved normalization:", path)


def load_normalization(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_norm", type=str, default="")
    args = parser.parse_args()

    ds = PlanWindowDataset(args.data_dir, horizon=args.horizon)
    print("data_dir:", args.data_dir)
    print("num episodes:", len(ds.paths))
    print("num plan windows:", len(ds))

    sample = ds[0]
    for k, v in sample.items():
        print(k, tuple(v.shape), v.dtype)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(dl))
    print("batch actions:", batch["actions"].shape)
    print("batch trajectory:", batch["trajectory"].shape)
    print("batch phase:", batch["phase"].shape)

    if args.save_norm:
        stats = compute_plan_normalization(args.data_dir, horizon=args.horizon)
        save_normalization(stats, args.save_norm)


if __name__ == "__main__":
    main()
