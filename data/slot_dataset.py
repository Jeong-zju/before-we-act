from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig


@dataclass
class SlotSampleSpec:
    file_idx: int
    t: int
    agent_id: int


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_angle(a - b)


def extract_agent_pose(global_state: np.ndarray, agent_id: int) -> np.ndarray:
    if agent_id == 0:
        return global_state[..., 0:3]
    if agent_id == 1:
        return global_state[..., 3:6]
    raise ValueError(f"agent_id must be 0 or 1, got {agent_id}")


def extract_other_pose(global_state: np.ndarray, agent_id: int) -> np.ndarray:
    return extract_agent_pose(global_state, 1 - agent_id)


def relative_pose(target_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    out = target_pose.copy()
    out[..., 0] = target_pose[..., 0] - reference_pose[..., 0]
    out[..., 1] = target_pose[..., 1] - reference_pose[..., 1]
    out[..., 2] = angle_diff(target_pose[..., 2], reference_pose[..., 2])
    return out.astype(np.float32)


def as_scalar_int(x) -> int:
    arr = np.asarray(x).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape={np.asarray(x).shape}")
    return int(arr[0])


def as_scalar_float(x) -> float:
    arr = np.asarray(x).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape={np.asarray(x).shape}")
    return float(arr[0])


def build_future_trajectory(global_state: np.ndarray, object_pose: np.ndarray, agent_id: int) -> np.ndarray:
    robot_pose = extract_agent_pose(global_state, agent_id)
    obj_xy = object_pose[..., 0:2]
    return np.concatenate([robot_pose, obj_xy], axis=-1).astype(np.float32)


class SlotWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        history: int = 8,
        horizon: int = 16,
        stride: int = 1,
        tokenizer_ckpt: str = "",
        max_episodes: int = -1,
        include_failures: bool = True,
        preload_tokenizer_labels: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.include_failures = bool(include_failures)
        self.tokenizer_ckpt = tokenizer_ckpt
        self.preload_tokenizer_labels = bool(preload_tokenizer_labels)

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if max_episodes > 0:
            self.paths = self.paths[:max_episodes]

        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index: List[SlotSampleSpec] = []

        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                T = int(f["actions/joint"].shape[0])
                failure = bool(f.attrs.get("failure", False))

            if failure and not self.include_failures:
                continue

            # Need history ending at t and future [t, t+horizon).
            min_t = self.history - 1
            max_t = T - self.horizon
            if max_t < min_t:
                continue

            for t in range(min_t, max_t + 1, self.stride):
                self.index.append(SlotSampleSpec(file_idx=file_idx, t=t, agent_id=0))
                self.index.append(SlotSampleSpec(file_idx=file_idx, t=t, agent_id=1))

        if not self.index:
            raise RuntimeError(f"No valid slot windows in {self.data_dir}")

        self._tokenizer = None
        self._tokenizer_norm = None
        if tokenizer_ckpt:
            self._load_tokenizer(tokenizer_ckpt)

    def _load_tokenizer(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = PlanTokenizerConfig(**ckpt["config"])
        model = PlanTokenizer(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._tokenizer = model
        self._tokenizer_norm = ckpt["normalization"]

    def __len__(self) -> int:
        return len(self.index)

    def _encode_plan_token(self, actions: np.ndarray, trajectory: np.ndarray) -> int:
        if self._tokenizer is None:
            return -1

        a = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)
        x = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0)

        norm = self._tokenizer_norm
        a = (a - norm["action_mean"].view(1, 1, -1)) / norm["action_std"].view(1, 1, -1)
        x = (x - norm["traj_mean"].view(1, 1, -1)) / norm["traj_std"].view(1, 1, -1)

        with torch.no_grad():
            enc = self._tokenizer.encode_future_segment(a, x)
        return int(enc["code_indices"][0].item())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.index[idx]
        path = self.paths[spec.file_idx]
        t = spec.t
        agent_id = spec.agent_id
        hist_start = t - self.history + 1
        hist_end = t + 1
        fut_start = t
        fut_end = t + self.horizon

        with h5py.File(path, "r") as f:
            if agent_id == 0:
                proprio = f["obs/robot_0/proprio"][hist_start:hist_end]
                prev_action = f["actions/robot_0"][hist_start:hist_end]
                future_action = f["actions/robot_0"][fut_start:fut_end]
            else:
                proprio = f["obs/robot_1/proprio"][hist_start:hist_end]
                prev_action = f["actions/robot_1"][hist_start:hist_end]
                future_action = f["actions/robot_1"][fut_start:fut_end]

            force = f["global/force_proxy"][hist_start:hist_end].reshape(-1, 1)
            contacts = f["global/contacts"][hist_start:hist_end].reshape(-1, 1)
            global_state_hist = f["global/global_state"][hist_start:hist_end]
            object_pose_hist = f["global/object_pose"][hist_start:hist_end]
            phase_all = f["labels/phase"][:].reshape(-1)
            phase_hist = phase_all[hist_start:hist_end]

            global_state_now = f["global/global_state"][t]
            object_pose_now = f["global/object_pose"][t]
            phase_now = as_scalar_int(phase_all[t])

            contacts_all = f["global/contacts"][:].reshape(-1)
            force_all = f["global/force_proxy"][:].reshape(-1)
            contact_now = float(as_scalar_float(contacts_all[t]) > 0)
            force_now = as_scalar_float(force_all[t])

            global_state_future = f["global/global_state"][fut_start:fut_end]
            object_pose_future = f["global/object_pose"][fut_start:fut_end]

        self_pose = extract_agent_pose(global_state_now, agent_id).astype(np.float32)
        other_pose = extract_other_pose(global_state_now, agent_id).astype(np.float32)
        object_pose = object_pose_now.astype(np.float32)

        other_rel = relative_pose(other_pose, self_pose)
        object_rel = relative_pose(object_pose, self_pose)

        future_trajectory = build_future_trajectory(global_state_future, object_pose_future, agent_id)
        plan_token = self._encode_plan_token(future_action, future_trajectory)

        # local feature is intentionally semi-structured MVP input.
        local_history = np.concatenate(
            [
                proprio.astype(np.float32),
                prev_action.astype(np.float32),
                force.astype(np.float32),
                contacts.astype(np.float32),
            ],
            axis=-1,
        )

        return {
            "local_history": torch.tensor(local_history, dtype=torch.float32),
            "proprio_history": torch.tensor(proprio, dtype=torch.float32),
            "prev_action_history": torch.tensor(prev_action, dtype=torch.float32),
            "force_history": torch.tensor(force, dtype=torch.float32),
            "contact_history": torch.tensor(contacts, dtype=torch.float32),
            "phase_history": torch.tensor(phase_hist, dtype=torch.long),
            "agent_id": torch.tensor(agent_id, dtype=torch.long),
            "self_pose": torch.tensor(self_pose, dtype=torch.float32),
            "other_rel_pose": torch.tensor(other_rel, dtype=torch.float32),
            "object_rel_pose": torch.tensor(object_rel, dtype=torch.float32),
            "contact": torch.tensor(contact_now, dtype=torch.float32),
            "force_proxy": torch.tensor(force_now, dtype=torch.float32),
            "phase": torch.tensor(phase_now, dtype=torch.long),
            "plan_token": torch.tensor(plan_token, dtype=torch.long),
        }


def compute_slot_normalization(data_dir: str, history: int = 8, horizon: int = 16, max_samples: int = 20000) -> Dict[str, torch.Tensor]:
    ds = SlotWindowDataset(data_dir=data_dir, history=history, horizon=horizon, stride=max(1, history), tokenizer_ckpt="")
    n = min(len(ds), max_samples)

    local = []
    self_pose = []
    other_rel = []
    object_rel = []

    for i in range(n):
        s = ds[i]
        local.append(s["local_history"])
        self_pose.append(s["self_pose"])
        other_rel.append(s["other_rel_pose"])
        object_rel.append(s["object_rel_pose"])

    local = torch.stack(local, dim=0).reshape(-1, 17)
    self_pose = torch.stack(self_pose, dim=0)
    other_rel = torch.stack(other_rel, dim=0)
    object_rel = torch.stack(object_rel, dim=0)

    stats = {
        "local_mean": local.mean(dim=0),
        "local_std": local.std(dim=0).clamp_min(1e-6),
        "self_pose_mean": self_pose.mean(dim=0),
        "self_pose_std": self_pose.std(dim=0).clamp_min(1e-6),
        "other_rel_mean": other_rel.mean(dim=0),
        "other_rel_std": other_rel.std(dim=0).clamp_min(1e-6),
        "object_rel_mean": object_rel.mean(dim=0),
        "object_rel_std": object_rel.std(dim=0).clamp_min(1e-6),
    }
    return stats


def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokenizer_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--save_norm", type=str, default="")
    args = parser.parse_args()

    ds = SlotWindowDataset(args.data_dir, history=args.history, horizon=args.horizon, tokenizer_ckpt=args.tokenizer_ckpt)
    print("data_dir:", args.data_dir)
    print("num episodes:", len(ds.paths))
    print("num slot windows:", len(ds))

    s = ds[0]
    for k, v in s.items():
        print(k, tuple(v.shape) if hasattr(v, "shape") else v, v.dtype if hasattr(v, "dtype") else type(v))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    b = next(iter(dl))
    print("batch local_history:", b["local_history"].shape)
    print("batch self_pose:", b["self_pose"].shape)
    print("batch object_rel_pose:", b["object_rel_pose"].shape)
    print("batch plan_token:", b["plan_token"].shape, b["plan_token"][:8])

    if args.save_norm:
        stats = compute_slot_normalization(args.data_dir, history=args.history, horizon=args.horizon)
        Path(args.save_norm).parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, args.save_norm)
        print("saved normalization:", args.save_norm)


if __name__ == "__main__":
    main()
