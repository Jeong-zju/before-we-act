from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.slot_dataset import build_future_trajectory


@dataclass
class IntentionSampleSpec:
    file_idx: int
    t: int
    ego_id: int


class IntentionWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        history: int = 8,
        horizon: int = 16,
        stride: int = 1,
        max_episodes: int = -1,
        include_failures: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.include_failures = bool(include_failures)

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if max_episodes > 0:
            self.paths = self.paths[:max_episodes]

        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index: List[IntentionSampleSpec] = []
        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                T = int(f["actions/joint"].shape[0])
                failure = bool(f.attrs.get("failure", False))

            if failure and not self.include_failures:
                continue

            min_t = self.history - 1
            max_t = T - self.horizon - 1
            if max_t < min_t:
                continue

            for t in range(min_t, max_t + 1, self.stride):
                self.index.append(IntentionSampleSpec(file_idx=file_idx, t=t, ego_id=0))
                self.index.append(IntentionSampleSpec(file_idx=file_idx, t=t, ego_id=1))

        if not self.index:
            raise RuntimeError(f"No valid intention windows in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _make_local_history(proprio, actions, force, contacts):
        return np.concatenate(
            [
                proprio.astype(np.float32),
                actions.astype(np.float32),
                force.astype(np.float32),
                contacts.astype(np.float32),
            ],
            axis=-1,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.index[idx]
        path = self.paths[spec.file_idx]
        t = spec.t
        ego_id = spec.ego_id
        target_id = 1 - ego_id
        L = self.history
        H = self.horizon

        hs = t - L + 1
        he = t + 1
        fs = t
        fe = t + H

        with h5py.File(path, "r") as f:
            phase_all = f["labels/phase"][:].reshape(-1).astype(np.int64)
            force_all = f["global/force_proxy"][:].reshape(-1).astype(np.float32)
            contact_all = f["global/contacts"][:].reshape(-1).astype(np.float32)

            if ego_id == 0:
                ego_prop = f["obs/robot_0/proprio"][hs:he]
                ego_prev_action = f["actions/robot_0"][hs:he]
                ego_future_action = f["actions/robot_0"][fs:fe]
                target_future_action = f["actions/robot_1"][fs:fe]
            else:
                ego_prop = f["obs/robot_1/proprio"][hs:he]
                ego_prev_action = f["actions/robot_1"][hs:he]
                ego_future_action = f["actions/robot_1"][fs:fe]
                target_future_action = f["actions/robot_0"][fs:fe]

            global_state_future = f["global/global_state"][fs:fe]
            object_pose_future = f["global/object_pose"][fs:fe]
            global_state_now = f["global/global_state"][t]
            object_pose_now = f["global/object_pose"][t]

        force_hist = force_all[hs:he].reshape(-1, 1)
        contact_hist = contact_all[hs:he].reshape(-1, 1)
        local_history = self._make_local_history(ego_prop, ego_prev_action, force_hist, contact_hist)
        phase_history = phase_all[hs:he]

        ego_traj = build_future_trajectory(global_state_future, object_pose_future, ego_id)
        target_traj = build_future_trajectory(global_state_future, object_pose_future, target_id)

        # lightweight context labels for diagnostics / later conditioning
        ego_pose_now = global_state_now[0:3] if ego_id == 0 else global_state_now[3:6]
        target_pose_now = global_state_now[3:6] if ego_id == 0 else global_state_now[0:3]
        rel_target_pose = target_pose_now.astype(np.float32) - ego_pose_now.astype(np.float32)
        rel_target_pose[2] = ((rel_target_pose[2] + np.pi) % (2 * np.pi)) - np.pi

        object_rel = object_pose_now.astype(np.float32) - ego_pose_now.astype(np.float32)
        object_rel[2] = ((object_rel[2] + np.pi) % (2 * np.pi)) - np.pi

        return {
            "local_history": torch.tensor(local_history, dtype=torch.float32),
            "phase_history": torch.tensor(phase_history, dtype=torch.long),
            "ego_id": torch.tensor(ego_id, dtype=torch.long),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "ego_plan_actions": torch.tensor(ego_future_action, dtype=torch.float32),
            "ego_plan_trajectory": torch.tensor(ego_traj, dtype=torch.float32),
            "target_plan_actions": torch.tensor(target_future_action, dtype=torch.float32),
            "target_plan_trajectory": torch.tensor(target_traj, dtype=torch.float32),
            "target_contact": torch.tensor(float(contact_all[t] > 0), dtype=torch.float32),
            "target_force": torch.tensor(float(force_all[t]), dtype=torch.float32),
            "phase": torch.tensor(int(phase_all[t]), dtype=torch.long),
            "rel_target_pose": torch.tensor(rel_target_pose, dtype=torch.float32),
            "object_rel_pose": torch.tensor(object_rel, dtype=torch.float32),
        }


def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    ds = IntentionWindowDataset(args.data_dir, history=args.history, horizon=args.horizon)
    print("data_dir:", args.data_dir)
    print("num episodes:", len(ds.paths))
    print("num intention windows:", len(ds))

    sample = ds[0]
    for k, v in sample.items():
        print(k, tuple(v.shape), v.dtype)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(dl))
    print("batch local_history:", batch["local_history"].shape)
    print("batch target_plan_actions:", batch["target_plan_actions"].shape)
    print("batch target_plan_trajectory:", batch["target_plan_trajectory"].shape)


if __name__ == "__main__":
    main()
