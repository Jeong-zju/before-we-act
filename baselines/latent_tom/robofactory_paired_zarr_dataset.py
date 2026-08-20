from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class RoboFactoryPairedZarrDataset(BaseImageDataset):
    """Two-agent view of the joint RoboFactory train split for LatentToM.

    Agent 0 and agent 1 are present in all six tasks. Their synchronized head
    cameras form the private observations; their pixel mean is the shared view.
    """

    def __init__(self, zarr_path_0: str, zarr_path_1: str, horizon: int = 40,
                 pad_before: int = 1, pad_after: int = 19, n_obs_steps: int = 2,
                 seed: int = 42, val_ratio: float = 0.02,
                 max_train_episodes=None, **kwargs):
        rb0 = ReplayBuffer.create_from_path(zarr_path_0, mode="r")
        rb1 = ReplayBuffer.create_from_path(zarr_path_1, mode="r")
        ends0 = np.asarray(rb0.episode_ends[:])
        ends1 = np.asarray(rb1.episode_ends[:])
        if not np.array_equal(ends0, ends1):
            raise ValueError("agent zarr episode boundaries do not match")

        val_mask = get_val_mask(rb0.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        first_k = {"head_camera": n_obs_steps, "state": n_obs_steps}
        sampler_args = dict(sequence_length=horizon, pad_before=pad_before,
                            pad_after=pad_after, episode_mask=train_mask,
                            key_first_k=first_k)
        self.sampler0 = SequenceSampler(replay_buffer=rb0, **sampler_args)
        self.sampler1 = SequenceSampler(replay_buffer=rb1, **sampler_args)
        self.rb0, self.rb1 = rb0, rb1
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.zarr_path = zarr_path_0
        self.train_episodes_num = int(train_mask.sum())
        self.val_episodes_num = int(val_mask.sum())

    def get_validation_dataset(self):
        out = copy.copy(self)
        first_k = {"head_camera": self.n_obs_steps, "state": self.n_obs_steps}
        args = dict(sequence_length=self.horizon, pad_before=self.pad_before,
                    pad_after=self.pad_after, episode_mask=self.val_mask,
                    key_first_k=first_k)
        out.sampler0 = SequenceSampler(replay_buffer=self.rb0, **args)
        out.sampler1 = SequenceSampler(replay_buffer=self.rb1, **args)
        return out

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        normalizer["arm1_action"] = SingleFieldLinearNormalizer.create_fit(self.rb0["action"])
        normalizer["arm2_action"] = SingleFieldLinearNormalizer.create_fit(self.rb1["action"])
        for key in ("arm1_robot_eef_pos", "arm1_eef_quat"):
            sl = slice(0, 3) if key.endswith("pos") else slice(3, 7)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.rb0["state"][:, sl])
        for key in ("arm2_robot_eef_pos", "arm2_eef_quat"):
            sl = slice(0, 3) if key.endswith("pos") else slice(3, 7)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.rb1["state"][:, sl])
        for key in ("camera_1", "camera_3", "camera_4"):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler0)

    @staticmethod
    def _chw_float(images: np.ndarray) -> np.ndarray:
        return np.moveaxis(images, -1, 1).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a = self.sampler0.sample_sequence(idx)
        b = self.sampler1.sample_sequence(idx)
        im0 = self._chw_float(a["head_camera"][:self.n_obs_steps])
        im1 = self._chw_float(b["head_camera"][:self.n_obs_steps])
        shared = (im0 + im1) * 0.5
        s0 = a["state"][:self.n_obs_steps].astype(np.float32)
        s1 = b["state"][:self.n_obs_steps].astype(np.float32)
        obs = {
            "camera_1": torch.from_numpy(im0),
            "camera_3": torch.from_numpy(shared),
            "camera_4": torch.from_numpy(im1),
            "arm1_robot_eef_pos": torch.from_numpy(s0[:, :3]),
            "arm1_eef_quat": torch.from_numpy(s0[:, 3:7]),
            "arm2_robot_eef_pos": torch.from_numpy(s1[:, :3]),
            "arm2_eef_quat": torch.from_numpy(s1[:, 3:7]),
        }
        return {
            "obs": obs,
            "arm1_action": torch.from_numpy(a["action"].astype(np.float32)),
            "arm2_action": torch.from_numpy(b["action"].astype(np.float32)),
        }
