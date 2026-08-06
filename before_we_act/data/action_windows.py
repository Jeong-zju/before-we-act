from __future__ import annotations

from collections import defaultdict
import random
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset, Sampler

from .raw_team_windows import TASKS


class CachedActionWindows(Dataset):
    """Causal legal inputs plus normalized future joint-action supervision."""

    def __init__(
        self,
        cache_path: str | Path,
        split: str,
        *,
        spatial_cache_path: str | Path | None = None,
        recovery_cache_path: str | Path | None = None,
    ) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("round") != "R12":
            raise ValueError("unsupported R12 action cache")
        if split not in ("train", "validation"):
            raise ValueError("R12 cache split must be train or validation")
        self.metadata = payload["metadata"]
        if (
            self.metadata.get("protocol_variant") != "causal_lag1_coldstart_dense_v2"
            or self.metadata.get("action_history_lag") != 1
            or self.metadata.get("cold_start_steps") != [0, 1, 2]
        ):
            raise ValueError("R12 cache is not the dense causal cold-start protocol")
        self.stats: Mapping[str, torch.Tensor] = payload["stats"]
        self.data: Mapping[str, torch.Tensor] = payload[split]
        size = int(self.data["visual"].shape[0])
        if not size or any(int(value.shape[0]) != size for value in self.data.values()):
            raise ValueError("R12 cache tensors have inconsistent lengths")
        if self.data["joint_actions"].shape[1:] != (100, 4, 8):
            raise ValueError("R12 joint action cache shape differs")
        self.spatial_data: Mapping[str, torch.Tensor] | None = None
        self.spatial_metadata: Mapping[str, object] | None = None
        self.recovery_data: Mapping[str, torch.Tensor] | None = None
        if spatial_cache_path is not None:
            spatial = torch.load(
                spatial_cache_path, map_location="cpu", weights_only=False
            )
            if (
                spatial.get("schema_version") != 1
                or spatial.get("round") != "R12-R3"
                or spatial.get("metadata", {}).get("protocol_variant")
                != "current_dinov3_vitb16_4x4_per_fixed_view_v1"
            ):
                raise ValueError("unsupported R12-R3 spatial cache")
            values: Mapping[str, torch.Tensor] = spatial[split]
            if tuple(values["spatial_tokens"].shape) != (size, 5, 16, 768):
                raise ValueError("R12-R3 spatial token cache shape differs")
            if tuple(values["spatial_view_mask"].shape) != (size, 5):
                raise ValueError("R12-R3 spatial view mask shape differs")
            if not torch.equal(values["task_index"], self.data["task_index"]):
                raise ValueError("R12-R3 spatial/action cache row identity differs")
            if not bool(torch.isfinite(values["spatial_tokens"]).all()):
                raise ValueError("R12-R3 spatial cache contains non-finite values")
            self.spatial_data = values
            self.spatial_metadata = spatial["metadata"]
        if recovery_cache_path is not None:
            if split != "train" or self.spatial_data is None:
                raise ValueError("R12-R3 recovery data is train-only and requires spatial data")
            recovery = torch.load(
                recovery_cache_path, map_location="cpu", weights_only=False
            )
            if (
                recovery.get("schema_version") != 1
                or recovery.get("round") != "R12-R3"
                or recovery.get("metadata", {}).get("protocol_variant")
                != "r12r2_student_on_policy_w10_teacher_recovery_v1"
            ):
                raise ValueError("unsupported R12-R3 recovery cache")
            values = recovery["train"]
            recovery_size = int(values["visual"].shape[0])
            required_shapes = {
                "visual": (recovery_size, 3, 16, 15),
                "view_mask": (recovery_size, 3, 5),
                "qpos": (recovery_size, 3, 4, 9),
                "actions": (recovery_size, 3, 4, 8),
                "agent_mask": (recovery_size, 4),
                "joint_actions": (recovery_size, 100, 4, 8),
                "action_step_mask": (recovery_size, 100),
                "spatial_tokens": (recovery_size, 5, 16, 768),
                "spatial_view_mask": (recovery_size, 5),
                "task_index": (recovery_size,),
            }
            for key, shape in required_shapes.items():
                if tuple(values[key].shape) != shape:
                    raise ValueError(f"R12-R3 recovery tensor {key} shape differs")
            if not recovery_size or not bool(torch.isfinite(values["spatial_tokens"]).all()):
                raise ValueError("R12-R3 recovery cache is empty or non-finite")
            self.recovery_data = values
        self.base_size = size
        recovery_size = 0 if self.recovery_data is None else len(self.recovery_data["task_index"])
        self.task_index = torch.cat(
            [
                self.data["task_index"],
                torch.empty(0, dtype=torch.long)
                if self.recovery_data is None
                else self.recovery_data["task_index"],
            ]
        )
        self.source_index = torch.cat(
            [
                torch.zeros(size, dtype=torch.long),
                torch.ones(recovery_size, dtype=torch.long),
            ]
        )

    def __len__(self) -> int:
        return int(len(self.task_index))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < self.base_size:
            row = {key: value[index] for key, value in self.data.items()}
            spatial = self.spatial_data
            spatial_index = index
        else:
            if self.recovery_data is None:
                raise IndexError(index)
            spatial_index = index - self.base_size
            row = {
                key: value[spatial_index]
                for key, value in self.recovery_data.items()
                if key not in ("spatial_tokens", "spatial_view_mask", "source_policy")
            }
            spatial = self.recovery_data
        if self.spatial_data is not None:
            row.update(
                {
                    "spatial_tokens": spatial["spatial_tokens"][spatial_index],
                    "spatial_view_mask": spatial["spatial_view_mask"][spatial_index],
                }
            )
        return row


class ExactFiveTaskWindowSampler(Sampler[list[int]]):
    """Deterministic one-sample-per-task batches shared by all four routes."""

    def __init__(
        self,
        task_indices: torch.Tensor,
        updates: int,
        seed: int,
        start_update: int = 0,
        source_indices: torch.Tensor | None = None,
        recovery_probability: float = 0.0,
    ) -> None:
        self.updates = int(updates)
        self.seed = int(seed)
        self.start_update = int(start_update)
        self.recovery_probability = float(recovery_probability)
        if not 0.0 <= self.recovery_probability <= 1.0:
            raise ValueError("R12 recovery sampling probability must be in [0,1]")
        if source_indices is None:
            source_indices = torch.zeros_like(task_indices)
        if tuple(source_indices.shape) != tuple(task_indices.shape):
            raise ValueError("R12 task/source indices shape differs")
        self.by_task_source: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, task in enumerate(task_indices.tolist()):
            self.by_task_source[(int(task), int(source_indices[index]))].append(index)
        if {task for task, source in self.by_task_source if source == 0} != set(range(len(TASKS))):
            raise ValueError("R12 action cache must contain all five task buckets")
        if self.recovery_probability and {
            task for task, source in self.by_task_source if source == 1
        } != set(range(len(TASKS))):
            raise ValueError("R12 recovery cache must contain all five task buckets")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def __iter__(self):
        for update in range(self.start_update + 1, self.updates + 1):
            rng = random.Random(self.seed + 1_000_003 * update)
            batch = []
            for task in range(len(TASKS)):
                source = 1 if rng.random() < self.recovery_probability else 0
                batch.append(rng.choice(self.by_task_source[(task, source)]))
            rng.shuffle(batch)
            yield batch
