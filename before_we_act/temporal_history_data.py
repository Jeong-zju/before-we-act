"""Frozen temporal-history sample contract over the original 720 demonstrations.

The action candidates consume the original 640x480 RGB corpus.  Measurement's
60x80 probe recordings are deliberately excluded.  The original corpus has no
oracle sidecars, so social supervision is represented by an explicit false
mask instead of silently substituting pseudo-labels or a different dataset.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


SIX_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
TASK_TEXT = {
    "lift_barrier": "Lift the barrier together",
    "camera_alignment": "Align the cameras together",
    "long_pipeline_delivery": "Deliver the long pipeline together",
    "take_photo": "Take a photo together",
    "pass_shoe": "Pass the shoe between robots",
    "place_food": "Place the food together",
}
HISTORY_STEPS = 16
ACTION_HORIZON = 100
EFFECTIVE_BATCH = 48
SAMPLES_PER_TASK = 8
TASK_TEXT_BYTES = 64
PAD_BYTE = 256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TemporalEpisode:
    path: str
    relative_path: str
    task: str
    task_text: str
    arms: tuple[int, ...]
    length: int
    seed: int
    episode_index: int
    manifest_path: str
    manifest_sha256: str
    hdf5_sha256: str


@dataclass(frozen=True)
class TeamTemporalRequest:
    episode_list_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


def load_temporal_episodes(
    manifest_paths: Sequence[str | Path], split: str = "train"
) -> list[TemporalEpisode]:
    """Load and verify the original six-task/full-resolution projection."""

    records: list[TemporalEpisode] = []
    seen_tasks: set[str] = set()
    task_texts: dict[str, str] = {}
    per_task: Counter[str] = Counter()
    for source in manifest_paths:
        manifest_path = Path(source).resolve(strict=True)
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        task = str(manifest.get("task", {}).get("id"))
        if task not in SIX_TASKS:
            raise ValueError(f"unsupported Step-2 task in {manifest_path}: {task}")
        seen_tasks.add(task)
        task_text = str(manifest.get("task", {}).get("text") or TASK_TEXT[task]).strip()
        if task_text != TASK_TEXT[task]:
            raise ValueError(
                f"canonical task text drift for {task}: {task_text!r} != {TASK_TEXT[task]!r}"
            )
        previous = task_texts.setdefault(task, task_text)
        if previous != task_text:
            raise ValueError(f"task text changed inside {manifest_path}")
        action_dimension = int(manifest.get("action", {}).get("dimension", 0))
        if action_dimension <= 0 or action_dimension % 8:
            raise ValueError(f"invalid per-agent action layout: {manifest_path}")
        arms = tuple(range(action_dimension // 8))
        manifest_hash = hashlib.sha256(raw).hexdigest()
        for row in manifest.get("episodes", []):
            if row.get("split") != split:
                continue
            relative_path = str(row["hdf5_path"])
            path = (manifest_path.parent / relative_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            record = TemporalEpisode(
                path=str(path),
                relative_path=relative_path,
                task=task,
                task_text=task_text,
                arms=arms,
                length=int(row["steps"]),
                seed=int(row["seed"]),
                episode_index=int(row["episode_index"]),
                manifest_path=str(manifest_path),
                manifest_sha256=manifest_hash,
                hdf5_sha256=str(row["hdf5_sha256"]),
            )
            if record.length < 1:
                raise ValueError(f"empty episode: {path}")
            records.append(record)
            per_task[task] += 1
    if seen_tasks != set(SIX_TASKS):
        raise ValueError(f"expected all six Step-2 tasks, got {sorted(seen_tasks)}")
    expected = Counter({task: 120 for task in SIX_TASKS})
    if per_task != expected or len(records) != 720:
        raise ValueError(
            f"formal Step-2 corpus must be exactly 720 train episodes: {per_task}"
        )
    return records


def episode_receipt(episodes: Sequence[TemporalEpisode]) -> dict:
    manifests: dict[str, str] = {}
    for episode in episodes:
        prior = manifests.setdefault(episode.manifest_path, episode.manifest_sha256)
        if prior != episode.manifest_sha256:
            raise ValueError(f"manifest identity drift: {episode.manifest_path}")
    records = [asdict(item) for item in episodes]
    return {
        "format_version": "before-we-act.step2.dataset_projection/1",
        "source": "original_training_manifests",
        "image_contract": "original uint8 RGB 640x480; no Measurement compact RGB",
        "tasks": list(SIX_TASKS),
        "task_texts": {task: TASK_TEXT[task] for task in SIX_TASKS},
        "training_manifests": manifests,
        "episodes": len(episodes),
        "episodes_per_task": {task: 120 for task in SIX_TASKS},
        "history_steps": HISTORY_STEPS,
        "action_horizon": ACTION_HORIZON,
        "social_supervision": {
            "available": False,
            "mask": False,
            "reason": "the original 720 trajectories have no oracle sidecars",
            "step2_policy_use": "B0-H consumes no B/P/T target",
        },
        "records_sha256": canonical_sha256(records),
    }


def task_text_tensor(value: str) -> tuple[torch.Tensor, torch.Tensor]:
    raw = value.encode("utf-8")
    if len(raw) > TASK_TEXT_BYTES:
        raise ValueError(f"task text exceeds {TASK_TEXT_BYTES} frozen bytes: {value!r}")
    tokens = torch.full((TASK_TEXT_BYTES,), PAD_BYTE, dtype=torch.long)
    mask = torch.zeros(TASK_TEXT_BYTES, dtype=torch.bool)
    if raw:
        tokens[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        mask[: len(raw)] = True
    return tokens, mask


class _VisualCache:
    def __init__(self, root: str | Path, limit: int = 64):
        self.root = Path(root)
        self.limit = int(limit)
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def path_for(self, episode: TemporalEpisode) -> Path:
        return self.root / episode.task / f"{episode.hdf5_sha256}.npz"

    def load(self, episode: TemporalEpisode) -> dict[str, np.ndarray]:
        key = episode.hdf5_sha256
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        path = self.path_for(episode)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen visual cache: {path}")
        with np.load(path, allow_pickle=False) as source:
            source_hash = str(source["source_hdf5_sha256"].item())
            if source_hash != episode.hdf5_sha256:
                raise ValueError(f"visual cache source mismatch: {path}")
            result = {
                name: np.asarray(source[name])
                for name in source.files
                if name.startswith("view_")
            }
        expected = {"view_global", *(f"view_agent_{arm}" for arm in episode.arms)}
        if set(result) != expected:
            raise ValueError(f"visual cache views differ for {path}: {sorted(result)}")
        for name, value in result.items():
            if value.shape != (episode.length, 768):
                raise ValueError(
                    f"visual cache shape differs for {path}/{name}: {value.shape}"
                )
            if value.dtype != np.float16 or not np.isfinite(value).all():
                raise ValueError(f"invalid visual cache values: {path}/{name}")
        self.values[key] = result
        while self.limit > 0 and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return result


class TeamTemporalDataset(Dataset):
    """One legal 16-step history and one 100-step own-action target."""

    MODEL_INPUT_FIELDS = frozenset(
        {
            "global_rgb",
            "local_rgb",
            "history_visual_raw",
            "history_qpos",
            "history_action",
            "history_mask",
            "action_history_mask",
            "task_bytes",
            "task_text_mask",
            "episode_reset",
        }
    )
    TARGET_FIELDS = frozenset({"action", "action_mask"})
    AUDIT_ONLY_FIELDS = frozenset(
        {
            "task",
            "sample_key",
            "episode_index",
            "time_index",
            "agent_slot",
            "manifest_sha256",
            "hdf5_sha256",
            "social_supervision_mask",
        }
    )

    def __init__(
        self,
        episodes: Sequence[TemporalEpisode],
        stats: Mapping[str, np.ndarray | torch.Tensor],
        visual_cache_root: str | Path,
        *,
        cache_limit: int = 64,
    ) -> None:
        self.episodes = list(episodes)
        self.q_mean = torch.as_tensor(stats["q_mean"], dtype=torch.float32)
        self.q_std = torch.as_tensor(stats["q_std"], dtype=torch.float32)
        self.a_mean = torch.as_tensor(stats["a_mean"], dtype=torch.float32)
        self.a_std = torch.as_tensor(stats["a_std"], dtype=torch.float32)
        if self.q_mean.shape != (9,) or self.a_mean.shape != (8,):
            raise ValueError("Step-2 expects 9D qpos and 8D per-agent action")
        if torch.any(self.q_std <= 0) or torch.any(self.a_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.visual_cache = _VisualCache(visual_cache_root, cache_limit)

    def __len__(self) -> int:
        return sum(item.length * len(item.arms) for item in self.episodes)

    def __getitem__(self, request: TeamTemporalRequest | tuple) -> dict:
        if not isinstance(request, TeamTemporalRequest):
            request = TeamTemporalRequest(*request)
        episode = self.episodes[request.episode_list_index]
        if request.task != episode.task or request.arm not in episode.arms:
            raise ValueError(f"sample request identity mismatch: {request}")
        if not 0 <= request.time_index < episode.length:
            raise IndexError(request.time_index)
        cache = self.visual_cache.load(episode)
        time_index = request.time_index
        observation_first = max(0, time_index - (HISTORY_STEPS - 1))
        observation_indices = list(range(observation_first, time_index + 1))
        observation_offset = HISTORY_STEPS - len(observation_indices)
        action_first = max(0, time_index - HISTORY_STEPS)
        action_indices = list(range(action_first, time_index))
        action_offset = HISTORY_STEPS - len(action_indices)

        history_visual = torch.zeros(HISTORY_STEPS, 2, 768, dtype=torch.float16)
        history_qpos = torch.zeros(HISTORY_STEPS, 9, dtype=torch.float32)
        history_action = torch.zeros(HISTORY_STEPS, 8, dtype=torch.float32)
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)

        with h5py.File(episode.path, "r") as handle:
            data = handle["data"]
            images = data["observation"]["images"]
            local_key = f"agent_{request.arm}"
            if local_key not in images:
                if episode.task != "place_food":
                    raise KeyError(f"missing {local_key} outside place_food fallback")
                local_key = "global"
            global_rgb = np.asarray(images["global"][time_index])
            local_rgb = np.asarray(images[local_key][time_index])
            if global_rgb.shape != (480, 640, 3) or local_rgb.shape != (480, 640, 3):
                raise ValueError(
                    f"original 640x480 RGB required: {episode.path}, "
                    f"global={global_rgb.shape}, local={local_rgb.shape}"
                )
            qpos_source = data["observation"]["agents"][f"panda_{request.arm}"]["qpos"]
            qpos = np.asarray(qpos_source[observation_indices], dtype=np.float32)
            action_source = data["action"]["agents"][f"panda_{request.arm}"]["commanded"]
            past_action = np.asarray(action_source[action_indices], dtype=np.float32)
            future_end = min(time_index + ACTION_HORIZON, episode.length)
            future = np.asarray(action_source[time_index:future_end], dtype=np.float32)

        local_cache_key = (
            "view_global" if local_key == "global" else f"view_agent_{request.arm}"
        )
        if local_cache_key not in cache:
            if episode.task != "place_food":
                raise KeyError(f"cache lacks {local_cache_key}")
            local_cache_key = "view_global"
        history_visual[observation_offset:, 0] = torch.from_numpy(
            cache["view_global"][observation_indices]
        )
        history_visual[observation_offset:, 1] = torch.from_numpy(
            cache[local_cache_key][observation_indices]
        )
        history_qpos[observation_offset:] = (
            torch.from_numpy(qpos) - self.q_mean
        ) / self.q_std
        history_mask[observation_offset:] = True
        if action_indices:
            history_action[action_offset:] = (
                torch.from_numpy(past_action) - self.a_mean
            ) / self.a_std
            action_history_mask[action_offset:] = True

        valid = len(future)
        padded = torch.empty(ACTION_HORIZON, 8, dtype=torch.float32)
        normalized_future = (torch.from_numpy(future) - self.a_mean) / self.a_std
        padded[:valid] = normalized_future
        padded[valid:] = normalized_future[-1]
        action_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        action_mask[:valid] = True
        task_bytes, task_text_mask = task_text_tensor(episode.task_text)
        return {
            "global_rgb": torch.from_numpy(global_rgb).permute(2, 0, 1).contiguous(),
            "local_rgb": torch.from_numpy(local_rgb).permute(2, 0, 1).contiguous(),
            "history_visual_raw": history_visual,
            "history_qpos": history_qpos,
            "history_action": history_action,
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "task_bytes": task_bytes,
            "task_text_mask": task_text_mask,
            "episode_reset": torch.tensor(time_index == 0, dtype=torch.bool),
            "action": padded,
            "action_mask": action_mask,
            "task": episode.task,
            "sample_key": request.sample_key,
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.int64),
            "time_index": torch.tensor(time_index, dtype=torch.int64),
            "agent_slot": torch.tensor(request.arm, dtype=torch.int64),
            "manifest_sha256": episode.manifest_sha256,
            "hdf5_sha256": episode.hdf5_sha256,
            "social_supervision_mask": torch.tensor(False, dtype=torch.bool),
        }


class ExactSixTaskDistributedBatchSampler(Sampler[list[TeamTemporalRequest]]):
    """Exactly 8 samples/task globally, split deterministically across ranks."""

    def __init__(
        self,
        episodes: Sequence[TemporalEpisode],
        *,
        updates: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_update: int = 0,
    ) -> None:
        if updates < 1 or not 0 <= start_update <= updates:
            raise ValueError("invalid Step-2 update interval")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank")
        if EFFECTIVE_BATCH % world_size:
            raise ValueError(f"world size must divide effective batch {EFFECTIVE_BATCH}")
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(SIX_TASKS):
            raise ValueError(f"expected six task buckets, got {sorted(self.by_task)}")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[TeamTemporalRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.seed + 1_000_003 * update)
        requests: list[TeamTemporalRequest] = []
        for task in SIX_TASKS:
            candidates = self.by_task[task]
            for _ in range(SAMPLES_PER_TASK):
                episode_list_index = candidates[rng.randrange(len(candidates))]
                episode = self.episodes[episode_list_index]
                arm = episode.arms[rng.randrange(len(episode.arms))]
                time_index = rng.randrange(episode.length)
                identity = (
                    f"{episode.manifest_sha256}:{episode.hdf5_sha256}:"
                    f"{episode.episode_index}:{task}:{arm}:{time_index}"
                )
                requests.append(
                    TeamTemporalRequest(
                        episode_list_index=episode_list_index,
                        arm=arm,
                        time_index=time_index,
                        sample_key=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                        task=task,
                    )
                )
        rng.shuffle(requests)
        counts = Counter(item.task for item in requests)
        expected = Counter({task: SAMPLES_PER_TASK for task in SIX_TASKS})
        if len(requests) != EFFECTIVE_BATCH or counts != expected:
            raise AssertionError(f"Step-2 sampler balance failure: {counts}")
        return requests

    def __iter__(self) -> Iterator[list[TeamTemporalRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            global_requests = self.requests_for_update(update)
            yield global_requests[self.rank :: self.world_size]

    def cursor_receipt(self, completed_update: int) -> dict:
        if not 0 <= completed_update <= self.updates:
            raise ValueError(completed_update)
        next_update = completed_update + 1
        keys = (
            [item.sample_key for item in self.requests_for_update(next_update)]
            if next_update <= self.updates
            else []
        )
        return {
            "format_version": "before-we-act.step2.sample_cursor/1",
            "seed": self.seed,
            "completed_update": completed_update,
            "next_update": next_update if keys else None,
            "next_sample_keys": keys,
            "effective_batch": EFFECTIVE_BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
        }

    def validate_cursor(self, receipt: Mapping) -> int:
        completed = int(receipt["completed_update"])
        expected = self.cursor_receipt(completed)
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"sample cursor mismatch at {key}")
        return completed


__all__ = [
    "ACTION_HORIZON",
    "EFFECTIVE_BATCH",
    "ExactSixTaskDistributedBatchSampler",
    "HISTORY_STEPS",
    "PAD_BYTE",
    "SAMPLES_PER_TASK",
    "SIX_TASKS",
    "TemporalEpisode",
    "TASK_TEXT",
    "TASK_TEXT_BYTES",
    "TeamTemporalDataset",
    "TeamTemporalRequest",
    "canonical_sha256",
    "episode_receipt",
    "load_temporal_episodes",
    "sha256_file",
    "task_text_tensor",
]
