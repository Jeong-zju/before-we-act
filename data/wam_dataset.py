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
class WAMSampleSpec:
    file_idx: int
    t: int


class WAMWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        history: int = 8,
        horizon: int = 16,
        stride: int = 1,
        max_episodes: int = -1,
        include_failures: bool = True,
        goal_y: float = 3.05,
    ):
        self.data_dir = Path(data_dir)
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.include_failures = bool(include_failures)
        self.goal_y = float(goal_y)

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if max_episodes > 0:
            self.paths = self.paths[:max_episodes]

        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index: List[WAMSampleSpec] = []
        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                T = int(f["actions/joint"].shape[0])
                failure = bool(f.attrs.get("failure", False))

            if failure and not self.include_failures:
                continue

            # Need history ending at t+horizon and actions [t, t+horizon).
            min_t = self.history - 1
            max_t = T - self.horizon - 1
            if max_t < min_t:
                continue

            for t in range(min_t, max_t + 1, self.stride):
                self.index.append(WAMSampleSpec(file_idx=file_idx, t=t))

        if not self.index:
            raise RuntimeError(f"No valid WAM windows in {self.data_dir}")

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
        H = self.horizon
        L = self.history

        local_seq = []
        phase_seq = []

        with h5py.File(path, "r") as f:
            phase_all = f["labels/phase"][:].reshape(-1)
            force_all = f["global/force_proxy"][:].reshape(-1)
            contacts_all = f["global/contacts"][:].reshape(-1)

            actions_joint = f["actions/joint"][t:t + H]
            actions_r0 = f["actions/robot_0"][t:t + H]
            actions_r1 = f["actions/robot_1"][t:t + H]
            global_state_future = f["global/global_state"][t:t + H]
            object_pose_future = f["global/object_pose"][t:t + H]

            for tau in range(t, t + H + 1):
                hs = tau - L + 1
                he = tau + 1

                p0 = f["obs/robot_0/proprio"][hs:he]
                p1 = f["obs/robot_1/proprio"][hs:he]
                a0 = f["actions/robot_0"][hs:he]
                a1 = f["actions/robot_1"][hs:he]
                ff = force_all[hs:he].reshape(-1, 1)
                cc = contacts_all[hs:he].reshape(-1, 1)

                local0 = self._make_local_history(p0, a0, ff, cc)
                local1 = self._make_local_history(p1, a1, ff, cc)
                local_seq.append(np.stack([local0, local1], axis=0))

                ph = phase_all[hs:he].astype(np.int64)
                phase_seq.append(np.stack([ph, ph], axis=0))

            target_contact = (contacts_all[t:t + H] > 0).astype(np.float32)
            target_force = force_all[t:t + H].astype(np.float32)
            target_phase = phase_all[t:t + H].astype(np.int64)

        # Plan tokenizer input for each agent at current t.
        traj0 = build_future_trajectory(global_state_future, object_pose_future, agent_id=0)
        traj1 = build_future_trajectory(global_state_future, object_pose_future, agent_id=1)

        object_y = object_pose_future[:, 1].astype(np.float32)
        goal_distance = np.maximum(self.goal_y - object_y, 0.0).astype(np.float32)
        progress = object_y.astype(np.float32)

        return {
            "local_history_seq": torch.tensor(np.stack(local_seq, axis=0), dtype=torch.float32),
            "phase_history_seq": torch.tensor(np.stack(phase_seq, axis=0), dtype=torch.long),
            "agent_ids": torch.tensor([0, 1], dtype=torch.long),
            "future_actions": torch.tensor(actions_joint, dtype=torch.float32),
            "agent_actions": torch.tensor(np.stack([actions_r0, actions_r1], axis=1), dtype=torch.float32),
            "plan_actions": torch.tensor(np.stack([actions_r0, actions_r1], axis=0), dtype=torch.float32),
            "plan_trajectory": torch.tensor(np.stack([traj0, traj1], axis=0), dtype=torch.float32),
            "target_contact": torch.tensor(target_contact, dtype=torch.float32),
            "target_force": torch.tensor(target_force, dtype=torch.float32),
            "target_phase": torch.tensor(target_phase, dtype=torch.long),
            "target_progress": torch.tensor(progress, dtype=torch.float32),
            "target_goal_distance": torch.tensor(goal_distance, dtype=torch.float32),
        }


def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ds = WAMWindowDataset(args.data_dir, history=args.history, horizon=args.horizon)
    print("data_dir:", args.data_dir)
    print("num episodes:", len(ds.paths))
    print("num windows:", len(ds))

    sample = ds[0]
    for k, v in sample.items():
        print(k, tuple(v.shape), v.dtype)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(dl))
    print("batch local_history_seq:", batch["local_history_seq"].shape)
    print("batch future_actions:", batch["future_actions"].shape)
    print("batch plan_actions:", batch["plan_actions"].shape)
    print("batch plan_trajectory:", batch["plan_trajectory"].shape)


if __name__ == "__main__":
    main()
