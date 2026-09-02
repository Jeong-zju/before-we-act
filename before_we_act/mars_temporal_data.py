"""MARS-Control adapter for the official temporal CARE reference policy.

This module deliberately lives beside, rather than inside, the frozen six-task
data contract.  Every arm is an independent sample and only its own RGB, qpos,
and executed action history are exposed to the deployed policy.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.temporal_history_data import (
    ACTION_HORIZON,
    HISTORY_STEPS,
    TeamTemporalRequest,
    task_text_tensor,
)
from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    PD_ACTION_HIGH,
    PD_ACTION_LOW,
    action_contract_hash,
    audit_action_array,
    canonicalize_action,
    contract_metadata,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
)


MARS_TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
TASK_TEXT = {
    "place_cube_in_cup": "Place the cube in the cup",
    "strike_cube_hard": "Strike the cube hard",
    "three_robots_place_shoes": "Place the shoes with three robots",
    "four_robots_stack_cube": "Stack the cubes with four robots",
}
ENV_DIR = {
    "place_cube_in_cup": "PlaceCubeInCup-rf",
    "strike_cube_hard": "StrikeCubeHard-rf",
    "three_robots_place_shoes": "ThreeRobotsPlaceShoes-rf",
    "four_robots_stack_cube": "FourRobotsStackCube-rf",
}
ARMS = {
    "place_cube_in_cup": 2,
    "strike_cube_hard": 2,
    "three_robots_place_shoes": 3,
    "four_robots_stack_cube": 4,
}
BASE_XY = {
    "place_cube_in_cup": ((0.0,-0.85),(0.0,0.65)),
    "strike_cube_hard": ((0.0,-0.65),(0.0,0.65)),
    "three_robots_place_shoes": ((-0.65,-0.55),(-0.65,0.60),(1.0,0.0)),
    "four_robots_stack_cube": ((0.8,1.0),(-0.8,1.0),(0.65,0.0),(-0.65,0.0)),
}
EFFECTIVE_BATCH = 48
SAMPLES_PER_TASK = EFFECTIVE_BATCH // len(MARS_TASKS)
def clip_pd_action(value: np.ndarray) -> np.ndarray:
    """Compatibility wrapper around the shared action contract."""

    return canonicalize_action(value)


def local_task_text(task: str, arm: int) -> str:
    x,y=BASE_XY[task][arm]
    return f"{TASK_TEXT[task]}|own_base={x:+.2f},{y:+.2f}"


@dataclass(frozen=True)
class MarsTemporalEpisode:
    path: str
    trajectory: str
    task: str
    task_text: str
    arms: tuple[int, ...]
    length: int
    episode_index: int
    cache_key: str


def _h5_files(root: Path, task: str) -> list[Path]:
    source = root / ENV_DIR[task] / "motionplanning"
    merged = source / f"{task}.h5"
    return [merged] if merged.is_file() else sorted(source.glob(f"{task}.shard*.h5"))


def load_mars_episodes(root: str | Path) -> list[MarsTemporalEpisode]:
    root = Path(root).resolve(strict=True)
    records: list[MarsTemporalEpisode] = []
    counts: Counter[str] = Counter()
    for task in MARS_TASKS:
        files = _h5_files(root, task)
        if not files:
            raise FileNotFoundError(root / ENV_DIR[task] / "motionplanning")
        episode_index = 0
        for path in files:
            with h5py.File(path, "r") as handle:
                names = sorted(
                    (key for key in handle if key.startswith("traj_")),
                    key=lambda value: int(value.rsplit("_", 1)[-1]),
                )
                for trajectory in names:
                    group = handle[trajectory]
                    length = int(group["actions/panda-0"].shape[0])
                    success = np.asarray(group["success"])
                    if length < 1 or not bool(success[-1]):
                        raise ValueError(f"non-successful MARS episode: {path}:{trajectory}")
                    identity = f"{path.resolve()}:{trajectory}:{length}".encode()
                    records.append(
                        MarsTemporalEpisode(
                            path=str(path.resolve()),
                            trajectory=trajectory,
                            task=task,
                            task_text=TASK_TEXT[task],
                            arms=tuple(range(ARMS[task])),
                            length=length,
                            episode_index=episode_index,
                            cache_key=hashlib.sha256(identity).hexdigest(),
                        )
                    )
                    episode_index += 1
                    counts[task] += 1
    expected = Counter({task: 150 for task in MARS_TASKS})
    if counts != expected or len(records) != 600:
        raise ValueError(f"MARS formal corpus must be 150 successes/task: {counts}")
    return records


def compute_normalization(episodes: Sequence[MarsTemporalEpisode], output: Path) -> dict:
    q_sum = np.zeros(9, np.float64); q_sq = np.zeros(9, np.float64); q_n = 0
    a_sum = np.zeros(8, np.float64); a_sq = np.zeros(8, np.float64); a_n = 0
    raw_action_values = 0; out_of_bounds_values = 0; max_abs_change = 0.0
    out_of_bounds_by_dimension = np.zeros(8, dtype=np.int64)
    for episode in episodes:
        with h5py.File(episode.path, "r") as handle:
            group = handle[episode.trajectory]
            for arm in episode.arms:
                q = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][:episode.length], np.float64)
                raw_action = np.asarray(
                    group[f"actions/panda-{arm}"][:episode.length], np.float32
                )
                canonical, audit = audit_action_array(raw_action)
                a = canonical.astype(np.float64)
                raw_action_values += int(audit["raw_values"])
                out_of_bounds_values += int(audit["out_of_bounds_values"])
                max_abs_change = max(max_abs_change, float(audit["max_abs_change"]))
                out_of_bounds_by_dimension += np.asarray(
                    audit["out_of_bounds_by_dimension"], dtype=np.int64
                )
                q_sum += q.sum(0); q_sq += np.square(q).sum(0); q_n += len(q)
                a_sum += a.sum(0); a_sq += np.square(a).sum(0); a_n += len(a)
    q_mean = q_sum / q_n; a_mean = a_sum / a_n
    q_std = np.sqrt(np.maximum(q_sq / q_n - q_mean ** 2, 1e-8))
    a_std = np.sqrt(np.maximum(a_sq / a_n - a_mean ** 2, 1e-8))
    value = {
        "format_version": "before-we-act.mars.normalization-absolute/4-action-contract",
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "action_contract": contract_metadata(),
        "episodes": len(episodes), "local_q_steps": q_n, "local_action_steps": a_n,
        "q_mean": q_mean.tolist(), "q_std": q_std.tolist(),
        "a_mean": a_mean.tolist(), "a_std": a_std.tolist(),
        "action_encoding": "absolute_pd_joint_pos",
        "action_canonicalization": {
            "source": "immutable_raw_hdf5_read_time_projection",
            "raw_values": raw_action_values,
            "out_of_bounds_values": out_of_bounds_values,
            "out_of_bounds_fraction": (
                float(out_of_bounds_values / raw_action_values)
                if raw_action_values else 0.0
            ),
            "max_abs_change": max_abs_change,
            "out_of_bounds_by_dimension": out_of_bounds_by_dimension.tolist(),
        },
    }
    value["normalization_sha256"] = normalization_stats_hash(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n")
    return value


def validate_mars_normalization(stats: Mapping) -> None:
    """Reject stale or hand-edited MARS normalization artifacts."""

    if stats.get("format_version") != (
        "before-we-act.mars.normalization-absolute/4-action-contract"
    ):
        raise ValueError("MARS normalization predates the shared action contract")
    if stats.get("action_encoding") != "absolute_pd_joint_pos":
        raise ValueError("MARS action encoding differs from the shared contract")
    validate_checkpoint_action_contract(
        {"action_contract": stats.get("action_contract")}
    )
    if stats.get("action_contract_version") != ACTION_CONTRACT_VERSION:
        raise ValueError("MARS normalization action contract version differs")
    if stats["action_contract"].get("sha256") != action_contract_hash():
        raise ValueError("MARS normalization action contract hash differs")
    expected = normalization_stats_hash(stats)
    if stats.get("normalization_sha256") != expected:
        raise ValueError("MARS normalization statistics hash differs")


class MarsVisualCache:
    def __init__(self, root: str | Path, limit: int = 16):
        self.root = Path(root); self.limit = limit
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
    def path_for(self, episode: MarsTemporalEpisode) -> Path:
        return self.root / episode.task / f"{episode.cache_key}.npz"
    def load(self, episode: MarsTemporalEpisode) -> dict[str, np.ndarray]:
        if episode.cache_key in self.values:
            self.values.move_to_end(episode.cache_key)
            return self.values[episode.cache_key]
        path = self.path_for(episode)
        if not path.is_file(): raise FileNotFoundError(f"missing MARS DINO cache: {path}")
        with np.load(path, allow_pickle=False) as source:
            result = {f"agent_{arm}": np.asarray(source[f"agent_{arm}"]) for arm in episode.arms}
        for name, value in result.items():
            if value.shape != (episode.length, 768) or value.dtype != np.float16:
                raise ValueError(f"invalid cache {path}/{name}: {value.shape}/{value.dtype}")
        self.values[episode.cache_key] = result
        while len(self.values) > self.limit: self.values.popitem(last=False)
        return result


class MarsTemporalDataset(Dataset):
    MODEL_INPUT_FIELDS = frozenset({
        "global_rgb", "local_rgb", "history_visual_raw", "history_qpos",
        "history_action", "history_mask", "action_history_mask", "task_bytes",
        "task_text_mask", "episode_reset",
    })
    def __init__(self, episodes: Sequence[MarsTemporalEpisode], stats: Mapping,
                 visual_cache_root: str | Path):
        validate_mars_normalization(stats)
        self.episodes = list(episodes); self.cache = MarsVisualCache(visual_cache_root)
        self.q_mean = torch.tensor(stats["q_mean"], dtype=torch.float32)
        self.q_std = torch.tensor(stats["q_std"], dtype=torch.float32)
        self.a_mean = torch.tensor(stats["a_mean"], dtype=torch.float32)
        self.a_std = torch.tensor(stats["a_std"], dtype=torch.float32)
    def __len__(self): return sum(x.length * len(x.arms) for x in self.episodes)
    def __getitem__(self, request: TeamTemporalRequest | tuple):
        if not isinstance(request, TeamTemporalRequest): request = TeamTemporalRequest(*request)
        episode = self.episodes[request.episode_list_index]
        t = request.time_index; arm = request.arm
        first = max(0, t - HISTORY_STEPS + 1); obs_idx = list(range(first, t + 1))
        action_first = max(0, t - HISTORY_STEPS); action_idx = list(range(action_first, t))
        obs_offset = HISTORY_STEPS - len(obs_idx); action_offset = HISTORY_STEPS - len(action_idx)
        visual = torch.zeros(HISTORY_STEPS, 2, 768, dtype=torch.float16)
        qhist = torch.zeros(HISTORY_STEPS, 9); ahist = torch.zeros(HISTORY_STEPS, 8)
        hmask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        amask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        cached = self.cache.load(episode)[f"agent_{arm}"]
        own = torch.from_numpy(cached[obs_idx])
        visual[obs_offset:, 0] = own; visual[obs_offset:, 1] = own
        with h5py.File(episode.path, "r") as handle:
            group = handle[episode.trajectory]
            image = np.asarray(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"][t])
            q = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][obs_idx], np.float32)
            past = clip_pd_action(np.asarray(group[f"actions/panda-{arm}"][action_idx], np.float32))
            future = clip_pd_action(np.asarray(group[f"actions/panda-{arm}"][t:min(t+ACTION_HORIZON, episode.length)], np.float32))
        if image.shape != (240, 320, 3): raise ValueError(f"MARS RGB drift: {image.shape}")
        qhist[obs_offset:] = (torch.from_numpy(q) - self.q_mean) / self.q_std; hmask[obs_offset:] = True
        if action_idx:
            ahist[action_offset:] = (torch.from_numpy(past) - self.a_mean) / self.a_std
            amask[action_offset:] = True
        normalized = (torch.from_numpy(future) - self.a_mean) / self.a_std
        target = torch.empty(ACTION_HORIZON, 8); target[:len(normalized)] = normalized
        target[len(normalized):] = normalized[-1]
        target_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool); target_mask[:len(normalized)] = True
        task_bytes, text_mask = task_text_tensor(local_task_text(episode.task,arm))
        rgb = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return {
            "global_rgb": rgb, "local_rgb": rgb.clone(), "history_visual_raw": visual,
            "history_qpos": qhist, "history_action": ahist, "history_mask": hmask,
            "action_history_mask": amask, "task_bytes": task_bytes,
            "task_text_mask": text_mask, "episode_reset": torch.tensor(t == 0),
            "action": target, "action_mask": target_mask,
        }


class MarsBalancedDistributedBatchSampler(Sampler[list[TeamTemporalRequest]]):
    def __init__(self, episodes: Sequence[MarsTemporalEpisode], updates: int, seed: int,
                 rank: int = 0, world_size: int = 1, start_update: int = 0):
        self.episodes=list(episodes); self.updates=updates; self.seed=seed
        self.rank=rank; self.world_size=world_size; self.start_update=start_update
        self.by_task=defaultdict(list)
        for i,e in enumerate(episodes): self.by_task[e.task].append(i)
        if EFFECTIVE_BATCH % world_size: raise ValueError("world size must divide 48")
    def __len__(self): return self.updates-self.start_update
    def requests_for_update(self, update: int) -> list[TeamTemporalRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng=random.Random(self.seed+1_000_003*update); rows=[]
        for task in MARS_TASKS:
            for _ in range(SAMPLES_PER_TASK):
                i=rng.choice(self.by_task[task]); e=self.episodes[i]
                arm=rng.choice(e.arms); t=rng.randrange(e.length)
                key=hashlib.sha256(f"{e.cache_key}:{arm}:{t}".encode()).hexdigest()
                rows.append(TeamTemporalRequest(i,arm,t,key,task))
        rng.shuffle(rows)
        return rows
    def cursor_receipt(self, completed_update: int) -> dict:
        next_update=completed_update+1
        keys=(
            [row.sample_key for row in self.requests_for_update(next_update)]
            if next_update <= self.updates else []
        )
        return {
            "format_version":"before-we-act.mars.b0h-cursor/1",
            "seed":self.seed,"completed_update":completed_update,
            "next_update":next_update if keys else None,"next_sample_keys":keys,
            "effective_batch":EFFECTIVE_BATCH,"samples_per_task":SAMPLES_PER_TASK,
        }
    def validate_cursor(self, receipt: Mapping) -> None:
        expected=self.cursor_receipt(int(receipt["completed_update"]))
        if dict(receipt) != expected:
            raise ValueError("MARS B0-H resume sample cursor drifted")
    def __iter__(self) -> Iterator[list[TeamTemporalRequest]]:
        for update in range(self.start_update+1,self.updates+1):
            yield self.requests_for_update(update)[self.rank::self.world_size]


__all__ = ["MARS_TASKS", "MarsBalancedDistributedBatchSampler", "MarsTemporalDataset",
           "MarsTemporalEpisode", "load_mars_episodes", "compute_normalization", "local_task_text",
           "PD_ACTION_LOW", "PD_ACTION_HIGH", "clip_pd_action", "validate_mars_normalization"]
