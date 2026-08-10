"""Frozen six-task data contract shared by the R11 candidates.

The sampler deliberately mirrors W10's per-update random seed and hierarchy,
while exposing deterministic micro-batches for models that cannot fit the full
effective batch of 48 on one GPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


SIX_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
FUTURE_OFFSETS = (1, 4, 8, 16)
SAMPLES_PER_TASK = 8
EFFECTIVE_BATCH = len(SIX_TASKS) * SAMPLES_PER_TASK


@dataclass(frozen=True)
class R11Episode:
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
class R11SampleRequest:
    episode_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_text_from_manifest(manifest: Mapping, row: Mapping) -> str:
    value = row.get("task_text")
    if value is None:
        value = manifest.get("task", {}).get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every R11 episode must have a non-empty manifest task_text")
    return value.strip()


def load_r11_episodes(
    manifest_paths: Sequence[str | Path], split: str = "train"
) -> list[R11Episode]:
    """Read and strictly validate the frozen six-task manifest projection."""

    records: list[R11Episode] = []
    seen_tasks: set[str] = set()
    task_texts: dict[str, str] = {}
    for source in manifest_paths:
        manifest_path = Path(source).resolve()
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        task = manifest.get("task", {}).get("id")
        if task not in SIX_TASKS:
            raise ValueError(f"unsupported R11 task in {manifest_path}: {task}")
        seen_tasks.add(task)
        action_dim = int(manifest.get("action", {}).get("dimension", 0))
        if action_dim <= 0 or action_dim % 8:
            raise ValueError(f"invalid per-agent action layout in {manifest_path}: {action_dim}")
        arms = tuple(range(action_dim // 8))
        manifest_sha256 = _sha256_bytes(raw)
        for row in manifest.get("episodes", []):
            if row.get("split") != split:
                continue
            task_text = _task_text_from_manifest(manifest, row)
            previous = task_texts.setdefault(task, task_text)
            if task_text != previous:
                raise ValueError(
                    f"task text drift for {task}: {previous!r} versus {task_text!r}"
                )
            relative_path = str(row["hdf5_path"])
            path = (manifest_path.parent / relative_path).resolve()
            record = R11Episode(
                path=str(path),
                relative_path=relative_path,
                task=task,
                task_text=task_text,
                arms=arms,
                length=int(row["steps"]),
                seed=int(row["seed"]),
                episode_index=int(row["episode_index"]),
                manifest_path=str(manifest_path),
                manifest_sha256=manifest_sha256,
                hdf5_sha256=str(row["hdf5_sha256"]),
            )
            if record.length <= 0:
                raise ValueError(f"empty episode in {manifest_path}: {relative_path}")
            records.append(record)
    if seen_tasks != set(SIX_TASKS):
        raise ValueError(f"expected all six R11 tasks, got {sorted(seen_tasks)}")
    missing = [record.path for record in records if not Path(record.path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} HDF5 files; first={missing[0]}")
    return records


def frozen_task_texts(episodes: Sequence[R11Episode]) -> dict[str, str]:
    values: dict[str, str] = {}
    for episode in episodes:
        previous = values.setdefault(episode.task, episode.task_text)
        if previous != episode.task_text:
            raise ValueError(f"task text drift for {episode.task}")
    if set(values) != set(SIX_TASKS):
        raise ValueError("task text receipt does not cover all six tasks")
    return {task: values[task] for task in SIX_TASKS}


def _resize_uint8(images: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    original_shape = images.shape
    flat = images.reshape(-1, *original_shape[-3:]).float()
    resized = F.interpolate(flat, size=image_size, mode="bilinear", antialias=True)
    return resized.round().clamp_(0, 255).to(torch.uint8).reshape(
        *original_shape[:-3], 3, *image_size
    )


class R11EpisodeDataset(Dataset):
    """Current dual-view observation, four future frames and a 100-step action."""

    def __init__(
        self,
        episodes: Sequence[R11Episode],
        stats: Mapping[str, np.ndarray | torch.Tensor],
        *,
        horizon: int = 100,
        future_offsets: Sequence[int] = FUTURE_OFFSETS,
        image_size: tuple[int, int] | None = None,
    ):
        if horizon != 100:
            raise ValueError("R11 common action horizon must remain 100")
        if tuple(future_offsets) != FUTURE_OFFSETS:
            raise ValueError(f"R11 future offsets must remain {FUTURE_OFFSETS}")
        self.episodes = list(episodes)
        self.horizon = horizon
        self.future_offsets = tuple(future_offsets)
        self.image_size = image_size
        self.q_mean = torch.as_tensor(stats["q_mean"], dtype=torch.float32)
        self.q_std = torch.as_tensor(stats["q_std"], dtype=torch.float32)
        self.a_mean = torch.as_tensor(stats["a_mean"], dtype=torch.float32)
        self.a_std = torch.as_tensor(stats["a_std"], dtype=torch.float32)
        if self.q_mean.shape != (9,) or self.a_mean.shape != (8,):
            raise ValueError("R11 expects 9D own qpos and 8D per-agent actions")
        if torch.any(self.q_std <= 0) or torch.any(self.a_std <= 0):
            raise ValueError("normalization standard deviations must be positive")

    def __len__(self) -> int:
        return sum(record.length * len(record.arms) for record in self.episodes)

    def __getitem__(self, request: R11SampleRequest | tuple) -> dict:
        if not isinstance(request, R11SampleRequest):
            request = R11SampleRequest(*request)
        episode = self.episodes[request.episode_index]
        if request.task != episode.task or request.arm not in episode.arms:
            raise ValueError(f"sample request identity mismatch: {request}")
        if not 0 <= request.time_index < episode.length:
            raise IndexError(request.time_index)

        with h5py.File(episode.path, "r") as handle:
            data = handle["data"]
            image_group = data["observation"]["images"]
            local_key = f"agent_{request.arm}"
            if local_key not in image_group:
                if episode.task != "place_food":
                    raise KeyError(f"missing {local_key} outside declared place_food fallback")
                local_key = "global"
            current = np.stack(
                [
                    np.asarray(image_group["global"][request.time_index]),
                    np.asarray(image_group[local_key][request.time_index]),
                ]
            )
            future = np.zeros(
                (len(self.future_offsets), 2, *current.shape[1:]), dtype=np.uint8
            )
            future_mask = np.zeros(len(self.future_offsets), dtype=np.bool_)
            for index, offset in enumerate(self.future_offsets):
                future_index = request.time_index + offset
                if future_index < episode.length:
                    future[index, 0] = image_group["global"][future_index]
                    future[index, 1] = image_group[local_key][future_index]
                    future_mask[index] = True
            qpos = np.asarray(
                data["observation"]["agents"][f"panda_{request.arm}"]["qpos"]
                [request.time_index],
                dtype=np.float32,
            )
            qpos_source = data["observation"]["agents"][f"panda_{request.arm}"]["qpos"]
            future_qpos = np.zeros((len(self.future_offsets), 9), dtype=np.float32)
            for index, offset in enumerate(self.future_offsets):
                future_index = request.time_index + offset
                if future_index < episode.length:
                    future_qpos[index] = qpos_source[future_index]
            action_source = data["action"]["agents"][f"panda_{request.arm}"]["commanded"]
            action_end = min(request.time_index + self.horizon, episode.length)
            action = np.asarray(
                action_source[request.time_index:action_end], dtype=np.float32
            )

        current_tensor = torch.from_numpy(current).permute(0, 3, 1, 2).contiguous()
        future_tensor = torch.from_numpy(future).permute(0, 1, 4, 2, 3).contiguous()
        if self.image_size is not None:
            current_tensor = _resize_uint8(current_tensor, self.image_size)
            future_tensor = _resize_uint8(future_tensor, self.image_size)

        action_tensor = torch.empty((self.horizon, 8), dtype=torch.float32)
        valid_steps = int(action.shape[0])
        action_tensor[:valid_steps] = torch.from_numpy(action)
        action_tensor[valid_steps:] = torch.from_numpy(action[-1])
        action_mask = torch.zeros(self.horizon, dtype=torch.bool)
        action_mask[:valid_steps] = True
        future_qpos_tensor = (
            torch.from_numpy(future_qpos) - self.q_mean.unsqueeze(0)
        ) / self.q_std.unsqueeze(0)
        future_qpos_tensor[~torch.from_numpy(future_mask)] = 0
        return {
            "current_rgb": current_tensor,
            "future_rgb": future_tensor,
            "future_mask": torch.from_numpy(future_mask),
            "qpos": (torch.from_numpy(qpos) - self.q_mean) / self.q_std,
            "future_qpos": future_qpos_tensor,
            "action": (action_tensor - self.a_mean) / self.a_std,
            "action_mask": action_mask,
            "task": episode.task,
            "task_text": episode.task_text,
            "agent": request.arm,
            "time_index": request.time_index,
            "episode_index": episode.episode_index,
            "manifest_sha256": episode.manifest_sha256,
            "hdf5_sha256": episode.hdf5_sha256,
            "sample_key": request.sample_key,
        }


class ExactSixTaskAccumulationSampler(Sampler[list[R11SampleRequest]]):
    """Yield micro-batches that sum to exactly eight samples per task/update."""

    def __init__(
        self,
        episodes: Sequence[R11Episode],
        *,
        updates: int,
        seed: int,
        micro_batch_size: int,
        start_update: int = 0,
    ):
        if updates < 1 or not 0 <= start_update <= updates:
            raise ValueError("invalid update interval")
        if micro_batch_size < 1 or EFFECTIVE_BATCH % micro_batch_size:
            raise ValueError(
                f"micro_batch_size must divide effective batch {EFFECTIVE_BATCH}"
            )
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.seed = int(seed)
        self.micro_batch_size = int(micro_batch_size)
        self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(SIX_TASKS):
            raise ValueError(f"expected six task buckets, got {sorted(self.by_task)}")

    @property
    def accumulation_steps(self) -> int:
        return EFFECTIVE_BATCH // self.micro_batch_size

    def __len__(self) -> int:
        return (self.updates - self.start_update) * self.accumulation_steps

    def requests_for_update(self, update: int) -> list[R11SampleRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.seed + 1_000_003 * update)
        requests: list[R11SampleRequest] = []
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
                    R11SampleRequest(
                        episode_index=episode_list_index,
                        arm=arm,
                        time_index=time_index,
                        sample_key=_sha256_bytes(identity.encode("utf-8")),
                        task=task,
                    )
                )
        rng.shuffle(requests)
        counts = Counter(item.task for item in requests)
        if counts != Counter({task: SAMPLES_PER_TASK for task in SIX_TASKS}):
            raise AssertionError(f"internal sampler balance failure: {counts}")
        return requests

    def microbatches_for_update(self, update: int) -> list[list[R11SampleRequest]]:
        requests = self.requests_for_update(update)
        return [
            requests[index : index + self.micro_batch_size]
            for index in range(0, EFFECTIVE_BATCH, self.micro_batch_size)
        ]

    def __iter__(self) -> Iterator[list[R11SampleRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield from self.microbatches_for_update(update)

    def cursor_receipt(self, completed_update: int) -> dict:
        if not 0 <= completed_update <= self.updates:
            raise ValueError(completed_update)
        next_update = completed_update + 1
        next_keys = (
            [item.sample_key for item in self.requests_for_update(next_update)]
            if next_update <= self.updates
            else []
        )
        return {
            "format_version": "before-we-act.r11.sample_cursor/1",
            "seed": self.seed,
            "completed_update": completed_update,
            "next_update": next_update if next_keys else None,
            "next_sample_keys": next_keys,
            "effective_batch": EFFECTIVE_BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
        }

    def validate_resume_receipt(self, receipt: Mapping) -> int:
        completed = int(receipt["completed_update"])
        expected = self.cursor_receipt(completed)
        for field in (
            "seed",
            "completed_update",
            "next_update",
            "next_sample_keys",
            "effective_batch",
            "samples_per_task",
        ):
            if receipt.get(field) != expected[field]:
                raise ValueError(f"resume sample cursor mismatch at {field}")
        return completed


def episode_receipt(episodes: Sequence[R11Episode]) -> dict:
    """Serializable manifest/text projection stored in every R11 checkpoint."""

    manifests: dict[str, str] = {}
    for episode in episodes:
        existing = manifests.setdefault(episode.manifest_path, episode.manifest_sha256)
        if existing != episode.manifest_sha256:
            raise ValueError(f"manifest identity drift: {episode.manifest_path}")
    return {
        "format_version": "before-we-act.r11.dataset_projection/1",
        "tasks": list(SIX_TASKS),
        "task_texts": frozen_task_texts(episodes),
        "training_manifests": manifests,
        "episodes": len(episodes),
        "future_offsets": list(FUTURE_OFFSETS),
        "action_horizon": 100,
        "records_sha256": _sha256_bytes(
            json.dumps(
                [asdict(episode) for episode in episodes],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    }
