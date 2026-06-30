from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import torch
from torch.utils.data import Dataset


class Stage2WindowDataset(Dataset):
    def __init__(self, data_dir: str, window: int = 32, stride: int = 1):
        self.data_dir = Path(data_dir)
        self.window = int(window)
        self.stride = int(stride)

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index = []
        for file_idx, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                T = f["actions/joint"].shape[0]
            for start in range(0, max(0, T - self.window + 1), self.stride):
                self.index.append((file_idx, start))

        if not self.index:
            raise RuntimeError("No valid windows. Reduce window size.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start = self.index[idx]
        path = self.paths[file_idx]
        end = start + self.window

        with h5py.File(path, "r") as f:
            obs0 = f["obs/robot_0/proprio"][start:end]
            obs1 = f["obs/robot_1/proprio"][start:end]
            actions = f["actions/joint"][start:end]
            global_state = f["global/global_state"][start:end]
            object_pose = f["global/object_pose"][start:end]
            force = f["global/force_proxy"][start:end]
            contacts = f["global/contacts"][start:end]
            robot_distance = f["global/robot_distance"][start:end]
            phase = f["labels/phase"][start:end]
            success = f["labels/success"][start:end]
            failure = f["labels/failure"][start:end]
            communication_dummy = f["labels/communication_dummy"][start:end]

        return {
            "obs_robot_0": torch.tensor(obs0, dtype=torch.float32),
            "obs_robot_1": torch.tensor(obs1, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "global_state": torch.tensor(global_state, dtype=torch.float32),
            "object_pose": torch.tensor(object_pose, dtype=torch.float32),
            "force_proxy": torch.tensor(force, dtype=torch.float32),
            "contacts": torch.tensor(contacts, dtype=torch.float32),
            "robot_distance": torch.tensor(robot_distance, dtype=torch.float32),
            "phase": torch.tensor(phase, dtype=torch.long).squeeze(-1),
            "success": torch.tensor(success, dtype=torch.float32),
            "failure": torch.tensor(failure, dtype=torch.float32),
            "communication_dummy": torch.tensor(communication_dummy, dtype=torch.float32),
        }


def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    ds = Stage2WindowDataset(args.data_dir, window=args.window)
    print("num files:", len(ds.paths))
    print("num windows:", len(ds))

    sample = ds[0]
    for k, v in sample.items():
        print(k, tuple(v.shape), v.dtype)

    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print("batch actions:", batch["actions"].shape)


if __name__ == "__main__":
    main()
