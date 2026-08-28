from __future__ import annotations

import bisect, glob, json, random
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import Dataset, Sampler
from .common import TASKS, ARMS, ACTION_LOW, ACTION_HIGH

def index_corpus(root: str | Path, stats_path: str | Path | None = None):
    by_task, stats = [[] for _ in TASKS], {"q_min": None, "q_max": None, "a_min": None, "a_max": None}
    episodes = local_streams = timesteps = 0
    for tid, task in enumerate(TASKS):
        task_episodes = 0
        for path in sorted(glob.glob(str(Path(root) / task / "motionplanning" / "*.shard*.h5"))):
            with h5py.File(path, "r") as handle:
                for trajectory in sorted(handle, key=lambda x: int(x.rsplit("_", 1)[-1])):
                    group = handle[trajectory]; task_episodes += 1; episodes += 1
                    if not bool(np.asarray(group["success"])[-1]): raise ValueError(f"non-success trajectory: {path}:{trajectory}")
                    n = min(len(group[f"actions/panda-{arm}"]) for arm in range(ARMS[task]))
                    for arm in range(ARMS[task]):
                        by_task[tid].append((path, trajectory, arm, n)); local_streams += 1; timesteps += n
                        q = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][:n], np.float32)
                        a = np.clip(np.asarray(group[f"actions/panda-{arm}"][:n], np.float32), ACTION_LOW, ACTION_HIGH)
                        for key, value in (("q_min", q.min(0)), ("a_min", a.min(0))): stats[key] = value if stats[key] is None else np.minimum(stats[key], value)
                        for key, value in (("q_max", q.max(0)), ("a_max", a.max(0))): stats[key] = value if stats[key] is None else np.maximum(stats[key], value)
        if task_episodes != 150: raise ValueError(f"{task}: expected 150 episodes, got {task_episodes}")
    payload = {k: v.tolist() for k, v in stats.items()} | {"episodes": episodes, "local_streams": local_streams, "indexed_local_timesteps": timesteps, "all_data": True}
    if stats_path: Path(stats_path).parent.mkdir(parents=True, exist_ok=True); Path(stats_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return by_task, payload

class MarsDPDataset(Dataset):
    def __init__(self, root: str | Path, stats_path: str | Path, obs_steps=3, horizon=8):
        self.obs_steps, self.horizon, self.handles = int(obs_steps), int(horizon), {}
        self.streams, self.stats = index_corpus(root, stats_path)
        self.entries, self.task_indices = [], [[] for _ in TASKS]
        for tid, streams in enumerate(self.streams):
            for stream in streams:
                for t in range(stream[3]):
                    idx = len(self.entries); self.entries.append((*stream, t, tid)); self.task_indices[tid].append(idx)
    def __getstate__(self): state = dict(self.__dict__); state["handles"] = {}; return state
    def __len__(self): return len(self.entries)
    def __getitem__(self, index):
        path, trajectory, arm, n, current, _tid = self.entries[index]
        if path not in self.handles: self.handles[path] = h5py.File(path, "r", swmr=True)
        group = self.handles[path][trajectory]; start = current - (self.obs_steps - 1)
        positions = np.clip(np.arange(start, start + self.horizon), 0, n - 1)
        obs_pos = positions[:self.obs_steps]
        def gather(dataset, indices):
            # h5py requires strictly increasing fancy indices; repeated/clipped
            # prefix frames are legal in our padded windows, so gather rows
            # explicitly to preserve the exact temporal contract.
            return np.stack([np.asarray(dataset[int(i)]) for i in indices])
        images = gather(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"], obs_pos).astype(np.uint8, copy=False)
        qpos = gather(group[f"obs/agent/panda-{arm}/qpos"], obs_pos).astype(np.float32, copy=False)
        actions = np.clip(gather(group[f"actions/panda-{arm}"], positions).astype(np.float32, copy=False), ACTION_LOW, ACTION_HIGH)
        return {"head_cam": torch.from_numpy(images).permute(0, 3, 1, 2), "agent_pos": torch.from_numpy(qpos), "action": torch.from_numpy(actions)}

class TaskBalancedBatchSampler(Sampler):
    def __init__(self, task_indices, batch_size, updates, seed):
        if batch_size % len(task_indices): raise ValueError("batch size must divide evenly across tasks")
        self.rows, self.batch_size, self.updates, self.seed, self.epoch = task_indices, batch_size, updates, seed, 0
    def __len__(self): return self.updates
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1; each = self.batch_size // len(self.rows)
        for _ in range(self.updates):
            batch = []
            for rows in self.rows: batch.extend(rng.choices(rows, k=each))
            rng.shuffle(batch); yield batch
