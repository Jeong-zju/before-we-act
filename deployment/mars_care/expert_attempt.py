"""Run exactly one official MARS motion-planning attempt."""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from planner.run import MP_SOLUTIONS
from utils.wrappers.record import RecordEpisodeMA


class _AmortizedArray:
    """A first-axis array builder with geometric growth.

    ManiSkill's recorder calls ``np.concatenate`` for every simulator step.
    For RGB trajectories that repeatedly copies the entire history, making
    recording quadratic in the episode length. This object preserves the
    recorder's indexing interface while growing only O(log N) times.
    """

    def __init__(self, first: np.ndarray, second: np.ndarray):
        second = self._compatible(first, second)
        required = len(first) + len(second)
        capacity = 16
        while capacity < required:
            capacity *= 2
        self._data = np.empty((capacity, *first.shape[1:]), dtype=first.dtype)
        self._data[: len(first)] = first
        self._data[len(first) : required] = second
        self._length = required

    @staticmethod
    def _compatible(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
        if reference.ndim > value.ndim:
            if reference.shape[1] == 1:
                value = value[:, None, ...]
            elif reference.shape[0] == 1:
                value = value[None, ...]
        return value

    def append(self, value: np.ndarray) -> "_AmortizedArray":
        value = self._compatible(self._data[: self._length], value)
        required = self._length + len(value)
        if required > len(self._data):
            capacity = len(self._data)
            while capacity < required:
                capacity *= 2
            grown = np.empty((capacity, *self._data.shape[1:]), dtype=self._data.dtype)
            grown[: self._length] = self._data[: self._length]
            self._data = grown
        self._data[self._length : required] = value
        self._length = required
        return self

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return (self._length, *self._data.shape[1:])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._data[: self._length][index]

    def __array__(self, dtype=None):
        value = self._data[: self._length]
        return value.astype(dtype, copy=False) if dtype is not None else value


def enable_amortized_recording() -> None:
    """Patch only this short-lived attempt process; official files stay untouched."""
    from mani_skill.utils import common

    original_append = common.append_dict_array
    original_index = common.index_dict_array

    def append_dict_array(left, right):
        if isinstance(left, _AmortizedArray):
            return left.append(right)
        if isinstance(left, np.ndarray):
            return _AmortizedArray(left, right)
        if isinstance(left, list):
            return left + right
        if isinstance(left, dict):
            for key in left:
                if key not in right:
                    raise KeyError(f"missing recorder field: {key}")
                left[key] = append_dict_array(left[key], right[key])
            return left
        return original_append(left, right)

    def index_dict_array(value, index, inplace=True):
        if isinstance(value, _AmortizedArray):
            return value[index]
        return original_index(value, index, inplace=inplace)

    common.append_dict_array = append_dict_array
    common.index_dict_array = index_dict_array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--render-device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--legacy-record-buffer", action="store_true")
    args = parser.parse_args()
    if not args.legacy_record_buffer:
        enable_amortized_recording()
    with open(args.config) as stream:
        env_id = yaml.safe_load(stream)["task_name"] + "-rf"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        env_id,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
        render_backend=args.render_device,
    )
    env = RecordEpisodeMA(
        env,
        output_dir=str(args.output_dir),
        trajectory_name=args.name,
        save_video=False,
        source_type="motionplanning",
        source_desc="official MARS motion planning solution",
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,
        record_observation=True,
    )
    try:
        result = MP_SOLUTIONS[env_id](env, seed=args.seed, debug=False, vis=False)
        success = result != -1 and bool(np.asarray(result[-1]["success"].detach().cpu()).all())
        env.flush_trajectory(save=success)
        print({"seed": args.seed, "success": success, "render_device": args.render_device}, flush=True)
        return 0 if success else 3
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
