from __future__ import annotations

import json
import os
import bisect
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

TASKS = ("lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo", "pass_shoe", "place_food")


def _episode_rows(root: Path):
    rows = []
    for task_id, task in enumerate(TASKS):
        manifest = json.loads((root / task / "training_manifest.json").read_text())
        for ep in manifest["episodes"]:
            rows.append((task_id, task, root / task / ep["hdf5_path"]))
    return rows


class LocalWindowDataset(Dataset):
    """Finite index of every timestep in every episode and local agent stream.

    The dataset never opens a peer image/state path.  A worker chooses one
    ``agent_i`` and returns only that agent's RGB, qpos, task id, and commanded
    action chunk.  Both physical arms therefore train the same parameter set.
    """

    def __init__(self, root: str | Path, *, obs_steps: int = 2, horizon: int = 40,
                 seed: int = 20260822):
        super().__init__()
        self.root = Path(root)
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        self.seed = int(seed)
        candidate_cache = Path(os.environ.get(
            "BWA_RESIZED_CACHE", "/workspace/datasets/robofactory_multitask_320x240"))
        self.cache_root = candidate_cache if (candidate_cache / "cache_manifest.json").is_file() else None
        rows = _episode_rows(self.root)
        if len(rows) != 900:
            raise RuntimeError(f"expected 900 episodes, found {len(rows)}")
        self.streams = []
        self.ends = []
        total = 0
        for task_id, task, path in rows:
            with h5py.File(path, "r") as handle:
                agents = sorted(int(k.rsplit("_", 1)[1]) for k in handle["data/observation/agents"].keys())
                for agent in agents:
                    n = min(
                        len(handle[f"data/observation/agents/panda_{agent}/qpos"]),
                        len(handle[f"data/observation/images/agent_{agent}"]),
                        len(handle[f"data/action/agents/panda_{agent}/commanded"]),
                    )
                    self.streams.append((task_id, task, path, agent, n))
                    total += n
                    self.ends.append(total)
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._image_arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def _images(self, path: Path, agent: int, handle: h5py.File):
        if self.cache_root is None:
            return handle[f"data/observation/images/agent_{agent}"]
        cache_path = self.cache_root / path.relative_to(self.root).with_suffix("") / f"agent_{agent}.npy"
        key = str(cache_path)
        images = self._image_arrays.get(key)
        if images is None:
            images = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            self._image_arrays[key] = images
            if len(self._image_arrays) > 32:
                self._image_arrays.popitem(last=False)
        else:
            self._image_arrays.move_to_end(key)
        return images

    def __len__(self):
        return self.ends[-1]

    def _read(self, task_id: int, task: str, path: Path, agent: int, start: int):
        key = str(path)
        handle = self._handles.get(key)
        if handle is None:
            handle = h5py.File(path, "r", libver="latest", swmr=True)
            self._handles[key] = handle
            if len(self._handles) > 32:
                _, old = self._handles.popitem(last=False)
                old.close()
        else:
            self._handles.move_to_end(key)
        qpos = handle[f"data/observation/agents/panda_{agent}/qpos"]
        image = self._images(path, agent, handle)
        action = handle[f"data/action/agents/panda_{agent}/commanded"]
        n = min(len(qpos), len(image), len(action))
        end_obs = min(start + self.obs_steps, n)
        end_act = min(start + self.horizon, n)
        obs_img = np.asarray(image[start:end_obs], dtype=np.uint8)
        obs_qpos = np.asarray(qpos[start:end_obs], dtype=np.float32)
        act = np.asarray(action[start:end_act], dtype=np.float32)
        if len(obs_img) == 0:
            raise RuntimeError(f"empty episode window: {path}")
        if len(obs_img) < self.obs_steps:
            obs_img = np.concatenate([obs_img, np.repeat(obs_img[-1:], self.obs_steps - len(obs_img), axis=0)])
            obs_qpos = np.concatenate([obs_qpos, np.repeat(obs_qpos[-1:], self.obs_steps - len(obs_qpos), axis=0)])
        if len(act) < self.horizon:
            act = np.concatenate([act, np.repeat(act[-1:], self.horizon - len(act), axis=0)])
        # Bounded action targets are stored in normalized [-1, 1] form by the
        # RoboFactory conversion; retain them exactly and normalize in policy.
        task_onehot = np.zeros((len(TASKS),), dtype=np.float32)
        task_onehot[task_id] = 1.0
        return {
            # Keep uint8 through workers and host-to-device transfer; conversion
            # happens once on GPU and cuts pinned-memory traffic by 4x.
            "image": torch.from_numpy(np.moveaxis(obs_img, -1, 1).copy()),
            "qpos": torch.from_numpy(obs_qpos),
            "task": torch.from_numpy(task_onehot),
            "action": torch.from_numpy(act),
        }

    def __getitems__(self, indices):
        """Read a locality-preserving batch with one HDF5 slice per stream.

        PyTorch's map-dataset fetcher calls ``__getitems__`` once for an entire
        batch.  Consecutive training indices have heavily overlapping two-frame
        observations and 40-step action targets, so reading every item
        separately multiplies HDF5 traffic by roughly the action horizon.  The
        returned samples are identical to repeated ``__getitem__`` calls; only
        the physical reads are coalesced.
        """
        normalized = []
        for position, raw_index in enumerate(indices):
            index = int(raw_index)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            stream_index = bisect.bisect_right(self.ends, index)
            previous = self.ends[stream_index - 1] if stream_index else 0
            normalized.append((position, stream_index, index - previous))

        grouped = {}
        for position, stream_index, start in normalized:
            grouped.setdefault(stream_index, []).append((position, start))
        result = [None] * len(normalized)

        for stream_index, requested in grouped.items():
            task_id, _task, path, agent, n = self.streams[stream_index]
            key = str(path)
            handle = self._handles.get(key)
            if handle is None:
                handle = h5py.File(path, "r", libver="latest", swmr=True)
                self._handles[key] = handle
                if len(self._handles) > 32:
                    _, old = self._handles.popitem(last=False)
                    old.close()
            else:
                self._handles.move_to_end(key)

            qpos_ds = handle[f"data/observation/agents/panda_{agent}/qpos"]
            image_ds = self._images(path, agent, handle)
            action_ds = handle[f"data/action/agents/panda_{agent}/commanded"]
            lo = min(start for _, start in requested)
            hi_obs = min(max(start for _, start in requested) + self.obs_steps, n)
            hi_act = min(max(start for _, start in requested) + self.horizon, n)
            images = np.asarray(image_ds[lo:hi_obs], dtype=np.uint8)
            qposes = np.asarray(qpos_ds[lo:hi_obs], dtype=np.float32)
            actions = np.asarray(action_ds[lo:hi_act], dtype=np.float32)

            task_onehot = np.zeros((len(TASKS),), dtype=np.float32)
            task_onehot[task_id] = 1.0
            for position, start in requested:
                offset = start - lo
                obs_img = images[offset:offset + self.obs_steps]
                obs_qpos = qposes[offset:offset + self.obs_steps]
                act = actions[offset:offset + self.horizon]
                if len(obs_img) < self.obs_steps:
                    obs_img = np.concatenate([
                        obs_img, np.repeat(obs_img[-1:], self.obs_steps - len(obs_img), axis=0)
                    ])
                    obs_qpos = np.concatenate([
                        obs_qpos, np.repeat(obs_qpos[-1:], self.obs_steps - len(obs_qpos), axis=0)
                    ])
                if len(act) < self.horizon:
                    act = np.concatenate([act, np.repeat(act[-1:], self.horizon - len(act), axis=0)])
                result[position] = {
                    "image": torch.from_numpy(np.moveaxis(obs_img, -1, 1).copy()),
                    "qpos": torch.from_numpy(np.ascontiguousarray(obs_qpos)),
                    "task": torch.from_numpy(task_onehot.copy()),
                    "action": torch.from_numpy(np.ascontiguousarray(act)),
                }
        return result

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        stream_index = bisect.bisect_right(self.ends, index)
        previous = self.ends[stream_index - 1] if stream_index else 0
        task_id, task, path, agent, _ = self.streams[stream_index]
        return self._read(task_id, task, path, agent, index - previous)
