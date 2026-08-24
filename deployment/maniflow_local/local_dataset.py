from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

TASKS = ("lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo", "pass_shoe", "place_food")


def episode_rows(root: Path):
    rows = []
    for task in TASKS:
        manifest = json.loads((root / task / "training_manifest.json").read_text())
        for episode in manifest["episodes"]:
            rows.append((task, root / task / episode["hdf5_path"]))
    return rows


def _stream_index(root: Path, obs_steps: int, horizon: int):
    streams, ends = [], []
    total = 0
    codec_by_task = {}
    for task in TASKS:
        manifest = json.loads((root / task / "training_manifest.json").read_text())
        config = manifest["action"]["codec"]["config"]
        codec_by_task[task] = (np.asarray(config["low"], np.float32), np.asarray(config["high"], np.float32))
    for task, path in episode_rows(root):
        with h5py.File(path, "r") as handle:
            agents = sorted(int(k.rsplit("_", 1)[1]) for k in handle["data/observation/agents"])
            for agent in agents:
                n = min(
                    len(handle[f"data/observation/images/agent_{agent}"]),
                    len(handle[f"data/observation/agents/panda_{agent}/qpos"]),
                    len(handle[f"data/action/agents/panda_{agent}/commanded"]),
                )
                # SequenceSampler semantics: start in [-pad_before, n-horizon+pad_after].
                count = max(0, n - horizon + (obs_steps - 1) + (horizon - 1) + 1)
                low, high = codec_by_task[task]
                streams.append((task, path, agent, n, count, low[agent * 8:(agent + 1) * 8], high[agent * 8:(agent + 1) * 8]))
                total += count
                ends.append(total)
    return streams, ends


class LocalManiFlowDataset(Dataset):
    """HDF5-backed windows with one robot's RGB/qpos/action only.

    Each physical arm is expanded into an independent local stream. The same
    dataset instance is used by a shared policy; no peer/global field is read.
    """

    def __init__(self, root: str | Path, *, obs_steps: int = 2, horizon: int = 16):
        self.root = Path(root)
        self.obs_steps, self.horizon = int(obs_steps), int(horizon)
        self.streams, self.ends = _stream_index(self.root, self.obs_steps, self.horizon)
        if len(episode_rows(self.root)) != 900 or not self.streams:
            raise RuntimeError("expected all 900 six-task episodes and non-empty local streams")
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def _locate(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        stream_idx = bisect.bisect_right(self.ends, idx)
        prev = self.ends[stream_idx - 1] if stream_idx else 0
        return self.streams[stream_idx], idx - prev

    def _handle(self, path: Path):
        key = str(path)
        handle = self._handles.get(key)
        if handle is None:
            handle = h5py.File(path, "r", libver="latest", swmr=True)
            self._handles[key] = handle
            if len(self._handles) > 32:
                _, old = self._handles.popitem(last=False); old.close()
        else:
            self._handles.move_to_end(key)
        return handle

    @staticmethod
    def _pad(array, before: int, after: int):
        if before:
            array = np.concatenate([np.repeat(array[:1], before, axis=0), array], axis=0)
        if after:
            array = np.concatenate([array, np.repeat(array[-1:], after, axis=0)], axis=0)
        return array

    def __getitem__(self, idx: int):
        (task, path, agent, n, _count, low, high), start = self._locate(idx)
        start -= self.obs_steps - 1
        end = start + self.horizon
        handle = self._handle(path)
        image_ds = handle[f"data/observation/images/agent_{agent}"]
        qpos_ds = handle[f"data/observation/agents/panda_{agent}/qpos"]
        action_ds = handle[f"data/action/agents/panda_{agent}/commanded"]
        lo, hi = max(0, start), min(n, end)
        obs_end = start + self.obs_steps
        obs_lo, obs_hi = max(0, start), min(n, obs_end)
        image = np.asarray(image_ds[obs_lo:obs_hi], dtype=np.uint8)
        qpos = np.asarray(qpos_ds[obs_lo:obs_hi], dtype=np.float32)
        action = np.asarray(action_ds[lo:hi, :8], dtype=np.float32)
        obs_before, obs_after = obs_lo - start, obs_end - obs_hi
        image = self._pad(image, obs_before, obs_after)
        qpos = self._pad(qpos, obs_before, obs_after)
        action = self._pad(action, lo - start, end - hi)
        action = np.clip(2.0 * (action - low) / (high - low) - 1.0, -1.0, 1.0)
        return {
            "obs": {
                "head_cam": torch.from_numpy(np.ascontiguousarray(image)),
                "agent_pos": torch.from_numpy(np.ascontiguousarray(qpos[:, :9])),
            },
            "action": torch.from_numpy(np.ascontiguousarray(action)),
            "task": task,
        }

    def __getitems__(self, indices):
        """Coalesce overlapping windows from each local actor stream."""
        located = []
        for position, idx in enumerate(indices):
            stream, local = self._locate(int(idx))
            stream_idx = bisect.bisect_right(self.ends, int(idx))
            located.append((position, stream_idx, local - self.obs_steps + 1))
        groups = {}
        for position, stream_idx, start in located:
            groups.setdefault(stream_idx, []).append((position, start))
        result = [None] * len(located)
        for stream_idx, requests in groups.items():
            task, path, agent, n, _count, low, high = self.streams[stream_idx]
            handle = self._handle(path)
            image_ds = handle[f"data/observation/images/agent_{agent}"]
            qpos_ds = handle[f"data/observation/agents/panda_{agent}/qpos"]
            action_ds = handle[f"data/action/agents/panda_{agent}/commanded"]
            obs_lo = max(0, min(start for _, start in requests))
            obs_hi = min(n, max(start for _, start in requests) + self.obs_steps)
            act_lo = max(0, min(start for _, start in requests))
            act_hi = min(n, max(start for _, start in requests) + self.horizon)
            images = np.asarray(image_ds[obs_lo:obs_hi], np.uint8)
            qposes = np.asarray(qpos_ds[obs_lo:obs_hi], np.float32)
            actions = np.asarray(action_ds[act_lo:act_hi, :8], np.float32)
            for position, start in requests:
                oi, oj = max(0, start) - obs_lo, min(n, start + self.obs_steps) - obs_lo
                ai, aj = max(0, start) - act_lo, min(n, start + self.horizon) - act_lo
                image = self._pad(images[oi:oj], max(0, -start), max(0, start + self.obs_steps - n))
                qpos = self._pad(qposes[oi:oj], max(0, -start), max(0, start + self.obs_steps - n))
                action = self._pad(actions[ai:aj], max(0, -start), max(0, start + self.horizon - n))
                action = np.clip(2.0 * (action - low) / (high - low) - 1.0, -1.0, 1.0)
                result[position] = {"obs": {
                    "head_cam": torch.from_numpy(np.ascontiguousarray(image)),
                    "agent_pos": torch.from_numpy(np.ascontiguousarray(qpos[:, :9])),
                }, "action": torch.from_numpy(np.ascontiguousarray(action)), "task": task}
        return result


def action_stats(root: str | Path):
    """Streaming all-episode action limits for ManiFlow's [-1,1] codec."""
    mins = np.full(8, np.inf, dtype=np.float64)
    maxs = np.full(8, -np.inf, dtype=np.float64)
    sums = np.zeros(8, dtype=np.float64); sqs = np.zeros(8, dtype=np.float64); count = 0
    for _task, path in episode_rows(Path(root)):
        with h5py.File(path, "r") as handle:
            for name in handle["data/action/agents"]:
                arr = np.asarray(handle[f"data/action/agents/{name}/commanded"][:, :8], np.float64)
                mins = np.minimum(mins, arr.min(0)); maxs = np.maximum(maxs, arr.max(0))
                sums += arr.sum(0); sqs += np.square(arr).sum(0); count += len(arr)
    mean = sums / count; std = np.sqrt(np.maximum(sqs / count - np.square(mean), 0.0))
    return mins.astype(np.float32), maxs.astype(np.float32), mean.astype(np.float32), std.astype(np.float32), count
