"""RoboFactory-only heterogeneous multi-task windows for Phase M2.

Each source task keeps its audited M1 conversion/training manifest.  This
wrapper composes those manifests without copying trajectory files, pads only
the explicit per-agent state/action slots, and returns dimension masks so the
model and losses can never treat padding as a physical actuator.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from data.robofactory import ROBOFACTORY_M1_PROFILE, ROBOFACTORY_M1_SCHEMA_VERSION
from train.generic_m1_trajectory_dataset import (
    GenericM1ManifestIndex,
    GenericM1WindowDataset,
)


ROBOFACTORY_M2_DATASET_PROTOCOL = "wam.robofactory.multitask/3"
PAD_TOKEN_ID = 0
UTF8_VOCAB_SIZE = 257
NATIVE_ROBOFACTORY_TASKS = {
    "camera_alignment": ("CameraAlignment-rf", "camera_alignment.yaml"),
    "lift_barrier": ("LiftBarrier-rf", "lift_barrier.yaml"),
    "long_pipeline_delivery": (
        "LongPipelineDelivery-rf",
        "long_pipeline_delivery.yaml",
    ),
    "pass_shoe": ("PassShoe-rf", "pass_shoe.yaml"),
    "pick_meat": ("PickMeat-rf", "pick_meat.yaml"),
    "place_food": ("PlaceFood-rf", "place_food.yaml"),
    "stack_cube": ("StackCube-rf", "stack_cube.yaml"),
    "strike_cube": ("StrikeCube-rf", "strike_cube.yaml"),
    "take_photo": ("TakePhoto-rf", "take_photo.yaml"),
    "three_robots_stack_cube": (
        "ThreeRobotsStackCube-rf",
        "three_robots_stack_cube.yaml",
    ),
    "two_robots_stack_cube": (
        "TwoRobotsStackCube-rf",
        "two_robots_stack_cube.yaml",
    ),
}


@dataclass(frozen=True)
class RoboFactoryM2TaskContract:
    task_id: str
    task_text: str
    manifest_path: Path
    manifest_sha256: str
    state_dim: int
    action_dim: int
    action_horizon: int
    agent_count: int
    camera_order: tuple[str, ...]
    action_codec_sha256: str
    normalization_sha256: str
    source_conversion_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "agent_count": self.agent_count,
            "camera_order": list(self.camera_order),
            "action_codec_sha256": self.action_codec_sha256,
            "normalization_sha256": self.normalization_sha256,
            "source_conversion_sha256": self.source_conversion_sha256,
        }


def encode_task_text(text: str, *, max_tokens: int) -> Tensor:
    """Encode natural-language task text as bounded UTF-8 bytes.

    Zero is reserved for padding.  Byte values use ids 1..256, which makes the
    tokenizer deterministic, dependency-free, multilingual, and checkpoint
    stable.  It is deliberately not a claim of pretrained language semantics.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        raise ValueError("task text cannot be empty")
    payload = normalized.encode("utf-8")
    if not payload:
        raise ValueError("task text produced no UTF-8 bytes")
    values = torch.zeros(max_tokens, dtype=torch.long)
    encoded = torch.tensor([value + 1 for value in payload[:max_tokens]], dtype=torch.long)
    values[: encoded.numel()] = encoded
    return values


class RoboFactoryMultitaskDataset(Dataset[dict[str, Tensor]]):
    """Compose audited RoboFactory task datasets into one masked M2 dataset."""

    BASE_SAMPLE_KEYS = frozenset(
        {
            "states",
            "state_valid_mask",
            "past_actions",
            "past_action_valid_mask",
            "images",
            "image_valid_mask",
            "action_targets",
            "action_target_valid_mask",
            "future_states",
            "future_state_valid_mask",
            "future_images",
            "future_image_novelty_mask",
            "future_visual_valid_mask",
            "future_horizons",
        }
    )

    def __init__(
        self,
        manifests: Sequence[str | Path | GenericM1ManifestIndex],
        *,
        split: str,
        state_history: int = 16,
        action_horizon: int = 16,
        task_action_horizons: Mapping[str, int] | None = None,
        visual_history: int = 4,
        future_horizons: Sequence[int] = (1, 4, 8, 16),
        cameras: Sequence[str] = ("global",),
        max_state_dim: int = 72,
        max_action_dim: int = 32,
        max_agents: int = 4,
        max_text_tokens: int = 96,
        stride: int = 1,
        hdf5_cache_size: int = 16,
        sample_keys: Collection[str] | None = None,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
        verify_normalization: bool = True,
    ) -> None:
        if not manifests:
            raise ValueError("M2 requires at least one RoboFactory manifest")
        if max_state_dim <= 0 or max_action_dim <= 0 or max_agents <= 0:
            raise ValueError("M2 maximum dimensions must be positive")
        if max_state_dim != 18 * max_agents or max_action_dim != 8 * max_agents:
            raise ValueError(
                "M2 maxima must describe complete 18D-state/8D-action Panda slots"
            )
        if max_text_tokens <= 0:
            raise ValueError("max_text_tokens must be positive")
        camera_slots = tuple(str(value) for value in cameras)
        expected_slots = ("global",) + tuple(
            f"agent_{index}" for index in range(max_agents)
        )
        if camera_slots != expected_slots[: len(camera_slots)]:
            raise ValueError(
                "M2 camera slots must be the canonical prefix "
                "['global', 'agent_0', ...]"
            )
        if len(camera_slots) != max_agents + 1:
            raise ValueError(
                "M2 camera slots must reserve global plus every maximum agent slot"
            )
        requested = self.BASE_SAMPLE_KEYS if sample_keys is None else frozenset(sample_keys)
        unknown = requested.difference(self.BASE_SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"unknown M2 sample keys: {sorted(unknown)}")
        required = {
            "states",
            "state_valid_mask",
            "past_actions",
            "past_action_valid_mask",
            "images",
            "image_valid_mask",
            "action_targets",
            "action_target_valid_mask",
            "future_states",
            "future_state_valid_mask",
            "future_images",
            "future_image_novelty_mask",
            "future_visual_valid_mask",
            "future_horizons",
        }
        if requested != required:
            raise ValueError("M2 currently requires the complete world-action sample contract")
        maximum_action_horizon = int(action_horizon)
        if maximum_action_horizon <= 0:
            raise ValueError("M2 maximum action horizon must be positive")
        declared_task_horizons = (
            None
            if task_action_horizons is None
            else {
                str(task_id): int(horizon)
                for task_id, horizon in task_action_horizons.items()
            }
        )
        if declared_task_horizons is not None:
            if not declared_task_horizons or any(
                not task_id or horizon <= 0 or horizon > maximum_action_horizon
                for task_id, horizon in declared_task_horizons.items()
            ):
                raise ValueError(
                    "task_action_horizons must map task ids to positive values "
                    "within action_horizon"
                )
            if max(declared_task_horizons.values()) != maximum_action_horizon:
                raise ValueError(
                    "data.action_horizon must equal max(task_action_horizons)"
                )

        indexes: list[GenericM1ManifestIndex] = []
        for value in manifests:
            if isinstance(value, GenericM1ManifestIndex):
                index = value
            else:
                index = GenericM1ManifestIndex.from_path(
                    value,
                    verify_hdf5_sha256=verify_hdf5_sha256,
                    verify_hdf5_contract=verify_hdf5_contract,
                    verify_normalization=verify_normalization,
                )
            indexes.append(index)

        task_ids: set[str] = set()
        datasets: list[GenericM1WindowDataset] = []
        contracts: list[RoboFactoryM2TaskContract] = []
        text_tokens: list[Tensor] = []
        state_means: list[Tensor] = []
        state_stds: list[Tensor] = []
        action_means: list[Tensor] = []
        action_stds: list[Tensor] = []
        for index in indexes:
            self._validate_manifest(index)
            task_ids_in_manifest = {record.task_id for record in index.episodes}
            texts_in_manifest = {record.task_text for record in index.episodes}
            if len(task_ids_in_manifest) != 1 or len(texts_in_manifest) != 1:
                raise ValueError("each M2 source manifest must represent exactly one task/text")
            task_id = next(iter(task_ids_in_manifest))
            task_text = next(iter(texts_in_manifest))
            if task_id in task_ids:
                raise ValueError(f"duplicate M2 task id {task_id!r}")
            task_ids.add(task_id)
            task_action_horizon = (
                maximum_action_horizon
                if declared_task_horizons is None
                else declared_task_horizons.get(task_id)
            )
            if task_action_horizon is None:
                raise ValueError(
                    f"task_action_horizons is missing task {task_id!r}"
                )
            if index.state_dim > max_state_dim or index.action_dim > max_action_dim:
                raise ValueError(
                    f"task {task_id!r} dimensions ({index.state_dim},{index.action_dim}) "
                    f"exceed M2 maxima ({max_state_dim},{max_action_dim})"
                )
            if index.state_dim % 18 or index.action_dim % 8:
                raise ValueError(
                    f"task {task_id!r} is not a RoboFactory Panda slot layout"
                )
            state_agents = index.state_dim // 18
            action_agents = index.action_dim // 8
            if state_agents != action_agents or state_agents > max_agents:
                raise ValueError(f"task {task_id!r} has inconsistent agent slots")
            task_cameras = self._available_task_cameras(
                index.camera_order,
                state_agents=state_agents,
                camera_slots=camera_slots,
                task_id=task_id,
            )
            source_conversion_sha256 = self._validate_native_source(
                index,
                task_id,
                expected_camera_order=task_cameras,
            )
            dataset = GenericM1WindowDataset(
                index,
                split=split,
                state_history=state_history,
                action_chunk=action_horizon,
                cameras=task_cameras,
                visual_history=visual_history,
                future_horizons=future_horizons,
                stride=stride,
                hdf5_cache_size=hdf5_cache_size,
                allow_incomplete_horizon=True,
                allow_incomplete_visual_history=True,
            )
            stats = index.load_normalization()
            state_mean = torch.zeros(max_state_dim, dtype=torch.float32)
            state_std = torch.ones(max_state_dim, dtype=torch.float32)
            state_mean[: index.state_dim] = torch.from_numpy(stats.state_mean.copy())
            state_std[: index.state_dim] = torch.from_numpy(stats.state_std.copy())
            state_means.append(state_mean)
            state_stds.append(state_std)
            action_mean = torch.zeros(max_action_dim, dtype=torch.float32)
            action_std = torch.ones(max_action_dim, dtype=torch.float32)
            action_mean[: index.action_dim] = torch.from_numpy(stats.action_mean.copy())
            action_std[: index.action_dim] = torch.from_numpy(stats.action_std.copy())
            if (
                not torch.isfinite(action_mean).all()
                or not torch.isfinite(action_std).all()
                or not bool(action_std[: index.action_dim].gt(0.0).all())
            ):
                raise ValueError(f"task {task_id!r} action normalization is invalid")
            action_means.append(action_mean)
            action_stds.append(action_std)
            codec = index.action_codec
            assert codec is not None
            contracts.append(
                RoboFactoryM2TaskContract(
                    task_id=task_id,
                    task_text=task_text,
                    manifest_path=index.manifest_path,
                    manifest_sha256=index.manifest_sha256,
                    state_dim=index.state_dim,
                    action_dim=index.action_dim,
                    action_horizon=int(task_action_horizon),
                    agent_count=state_agents,
                    camera_order=task_cameras,
                    action_codec_sha256=codec.semantic_sha256,
                    normalization_sha256=index.normalization_sha256,
                    source_conversion_sha256=source_conversion_sha256,
                )
            )
            text_tokens.append(encode_task_text(task_text, max_tokens=max_text_tokens))
            datasets.append(dataset)

        if declared_task_horizons is not None:
            unknown_horizons = set(declared_task_horizons).difference(task_ids)
            if unknown_horizons:
                raise ValueError(
                    "task_action_horizons contains tasks absent from manifests: "
                    f"{sorted(unknown_horizons)}"
                )
        image_shapes = {dataset.image_shape_hwc for dataset in datasets}
        if len(image_shapes) != 1:
            raise ValueError(f"M2 task image shapes differ: {sorted(image_shapes)}")
        self.datasets = tuple(datasets)
        self.contracts = tuple(contracts)
        self.task_vocabulary = tuple(contract.task_id for contract in contracts)
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_vocabulary)}
        self.split = str(split)
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.max_agents = int(max_agents)
        self.max_text_tokens = int(max_text_tokens)
        self.state_history = int(state_history)
        self.action_horizon = maximum_action_horizon
        self.task_action_horizons = {
            contract.task_id: contract.action_horizon
            for contract in self.contracts
        }
        self.visual_history = int(visual_history)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.camera_order = camera_slots
        self.camera_agent_index = torch.tensor(
            [max_agents]
            + list(range(max_agents)),
            dtype=torch.long,
        )
        self.image_shape_hwc = next(iter(image_shapes))
        self._text_tokens = torch.stack(text_tokens)
        self._state_means = torch.stack(state_means)
        self._state_stds = torch.stack(state_stds)
        self._action_means = torch.stack(action_means)
        self._action_stds = torch.stack(action_stds)
        self._offsets = [0]
        for dataset in self.datasets:
            self._offsets.append(self._offsets[-1] + len(dataset))

    @staticmethod
    def _available_task_cameras(
        camera_order: Sequence[str],
        *,
        state_agents: int,
        camera_slots: Sequence[str],
        task_id: str,
    ) -> tuple[str, ...]:
        available = tuple(map(str, camera_order))
        maximum = tuple(camera_slots[: state_agents + 1])
        if not available or available != maximum[: len(available)]:
            raise ValueError(
                f"task {task_id!r} cameras must be a non-empty canonical "
                f"prefix of {list(maximum)}; got {list(available)}"
            )
        return available

    @staticmethod
    def _validate_manifest(index: GenericM1ManifestIndex) -> None:
        if (
            index.schema_profile != ROBOFACTORY_M1_PROFILE
            or index.schema_version != ROBOFACTORY_M1_SCHEMA_VERSION
        ):
            raise ValueError(
                "M2 accepts only native RoboFactory converted data; "
                f"got {index.schema_profile!r}/{index.schema_version!r}"
            )
        if index.action_codec is None:
            raise ValueError("M2 requires a controller-bound action codec per task")
        if index.action_domain != "canonical_unit_action":
            raise ValueError("M2 task actions must use canonical_unit_action")

    @staticmethod
    def _validate_native_source(
        index: GenericM1ManifestIndex,
        task_id: str,
        *,
        expected_camera_order: Sequence[str],
    ) -> str:
        """Bind M2 data to an upstream native task config and planner lineage."""

        native = NATIVE_ROBOFACTORY_TASKS.get(task_id)
        if native is None:
            raise ValueError(f"M2 task {task_id!r} is not in the native RoboFactory set")
        raw_source = index.raw_manifest.get("source")
        if not isinstance(raw_source, Mapping):
            raise ValueError("M2 requires a hashed RoboFactory conversion source")
        relative = str(raw_source.get("conversion_manifest_path", ""))
        if not relative or Path(relative).is_absolute():
            raise ValueError("M2 conversion manifest path must be relative")
        root = index.manifest_path.parent.resolve()
        conversion_path = (root / relative).resolve(strict=True)
        if not conversion_path.is_relative_to(root) or not conversion_path.is_file():
            raise ValueError("M2 conversion manifest escapes the dataset directory")
        payload = conversion_path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != str(raw_source.get("conversion_manifest_sha256", "")):
            raise ValueError("M2 conversion manifest SHA-256 mismatch")
        try:
            conversion = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("M2 conversion manifest is invalid JSON") from exc
        if not isinstance(conversion, Mapping):
            raise ValueError("M2 conversion manifest root must be an object")
        if (
            conversion.get("format_version")
            != "robofactory.conversion_manifest/2.0"
            or conversion.get("schema_profile") != ROBOFACTORY_M1_PROFILE
            or conversion.get("schema_version") != ROBOFACTORY_M1_SCHEMA_VERSION
            or conversion.get("task_id") != task_id
        ):
            raise ValueError("M2 conversion manifest identity is not RoboFactory M1")

        source = conversion.get("source")
        metadata = source.get("metadata") if isinstance(source, Mapping) else None
        env_info = metadata.get("env_info") if isinstance(metadata, Mapping) else None
        env_kwargs = env_info.get("env_kwargs") if isinstance(env_info, Mapping) else None
        expected_env, config_name = native
        config_value = (
            str(env_kwargs.get("config", "")).replace("\\", "/")
            if isinstance(env_kwargs, Mapping)
            else ""
        )
        native_config = any(
            config_value.endswith(f"robofactory/configs/{scene}/{config_name}")
            for scene in ("table", "robocasa")
        )
        if (
            not isinstance(source, Mapping)
            or source.get("env_id") != expected_env
            or not isinstance(metadata, Mapping)
            or metadata.get("source_type") != "motionplanning"
            or not isinstance(env_info, Mapping)
            or env_info.get("env_id") != expected_env
            or not isinstance(env_kwargs, Mapping)
            or env_kwargs.get("control_mode") != "pd_joint_pos"
            or env_kwargs.get("obs_mode") != "rgb"
            or env_kwargs.get("render_mode") != "sensors"
            or not native_config
        ):
            raise ValueError(
                "M2 source is not a native RoboFactory table/robocasa planner dataset"
            )
        layout = conversion.get("layout")
        if (
            not isinstance(layout, Mapping)
            or int(layout.get("state_size", -1)) != index.state_dim
            or int(layout.get("action_size", -1)) != index.action_dim
        ):
            raise ValueError("M2 conversion layout disagrees with the training manifest")
        semantics = conversion.get("data_semantics")
        vision = semantics.get("vision") if isinstance(semantics, Mapping) else None
        if not isinstance(vision, Mapping):
            raise ValueError("M2 requires native RGB camera semantics")
        target_order = list(expected_camera_order)
        source_order = vision.get("source_camera_order")
        if vision.get("camera_order") != target_order or not isinstance(
            source_order, list
        ):
            raise ValueError(
                "M2 source camera order differs from the declared canonical prefix"
            )
        expected_sources = [
            {"head_camera", "head_camera_global"},
            *[
                {f"head_camera_agent{index}"}
                for index in range(len(target_order) - 1)
            ],
        ]
        if len(source_order) != len(expected_sources) or any(
            str(source) not in allowed
            for source, allowed in zip(source_order, expected_sources, strict=True)
        ):
            raise ValueError(
                "M2 source cameras do not match the declared canonical prefix"
            )
        return actual_sha256

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        task_index, local_index = self.resolve_index(index)
        raw = self.datasets[task_index][local_index]
        contract = self.contracts[task_index]
        state_mask = torch.arange(self.max_state_dim) < contract.state_dim
        action_mask = torch.arange(self.max_action_dim) < contract.action_dim
        states = self._pad_last(raw["states"], self.max_state_dim)
        future_states = self._pad_last(raw["future_states"], self.max_state_dim)
        means = self._state_means[task_index]
        stds = self._state_stds[task_index]
        states = (states - means) / stds
        future_states = (future_states - means) / stds
        states = states.masked_fill(~state_mask, 0.0)
        future_states = future_states.masked_fill(~state_mask, 0.0)
        action_mean = self._action_means[task_index]
        action_std = self._action_stds[task_index]
        past_actions = (
            self._pad_last(raw["past_actions"], self.max_action_dim) - action_mean
        ) / action_std
        action_targets = (
            self._pad_last(raw["action_targets"], self.max_action_dim) - action_mean
        ) / action_std
        past_actions = past_actions.masked_fill(~action_mask, 0.0)
        action_targets = action_targets.masked_fill(~action_mask, 0.0)
        task_horizon_mask = (
            torch.arange(self.action_horizon) < contract.action_horizon
        )
        future_visual_task_mask = torch.tensor(
            [
                horizon <= contract.action_horizon
                for horizon in self.future_horizons
            ],
            dtype=torch.bool,
        )
        images = self._pad_camera_axis(raw["images"], len(self.camera_order), axis=1)
        image_valid = self._pad_camera_axis(
            raw["image_valid_mask"].bool(),
            len(self.camera_order),
            axis=1,
        )
        future_images = self._pad_camera_axis(
            raw["future_images"],
            len(self.camera_order),
            axis=1,
        )
        future_novelty = self._pad_camera_axis(
            raw["future_image_novelty_mask"].bool(),
            len(self.camera_order),
            axis=1,
        )
        raw_future_valid = raw["future_visual_valid_mask"].bool()
        if raw_future_valid.ndim == 1:
            raw_future_valid = raw_future_valid[:, None].expand(
                -1,
                len(contract.camera_order),
            )
        elif raw_future_valid.ndim != 2:
            raise ValueError("future_visual_valid_mask must be [F] or [F,Cam]")
        future_visual_valid = self._pad_camera_axis(
            raw_future_valid,
            len(self.camera_order),
            axis=1,
        )
        future_visual_valid = (
            future_visual_valid & future_visual_task_mask[:, None]
        )
        return {
            "dataset_index": torch.tensor(index, dtype=torch.long),
            "states": states,
            "state_valid_mask": raw["state_valid_mask"].bool(),
            "state_dimension_mask": state_mask,
            "past_actions": past_actions,
            "past_action_valid_mask": raw["past_action_valid_mask"].bool(),
            "action_dimension_mask": action_mask,
            "images": images,
            "image_valid_mask": image_valid,
            "camera_agent_index": self.camera_agent_index.clone(),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "embodiment_index": torch.tensor(contract.agent_count - 1, dtype=torch.long),
            "task_text_tokens": self._text_tokens[task_index].clone(),
            "action_targets": action_targets,
            "action_target_valid_mask": (
                raw["action_target_valid_mask"].bool() & task_horizon_mask
            ),
            "action_horizon_mask": task_horizon_mask,
            "future_states": future_states,
            "future_state_valid_mask": (
                raw["future_state_valid_mask"].bool() & task_horizon_mask
            ),
            "future_images": future_images,
            "future_image_novelty_mask": future_novelty,
            "future_visual_valid_mask": future_visual_valid,
            "future_horizons": raw["future_horizons"],
        }

    @staticmethod
    def _pad_last(value: Tensor, width: int) -> Tensor:
        if value.shape[-1] > width:
            raise ValueError("tensor exceeds configured padded width")
        if value.shape[-1] == width:
            return value
        output = value.new_zeros((*value.shape[:-1], width))
        output[..., : value.shape[-1]] = value
        return output

    @staticmethod
    def _pad_camera_axis(value: Tensor, width: int, *, axis: int) -> Tensor:
        normalized_axis = axis if axis >= 0 else value.ndim + axis
        if normalized_axis < 0 or normalized_axis >= value.ndim:
            raise ValueError("camera padding axis is outside the tensor")
        current = int(value.shape[normalized_axis])
        if current > width:
            raise ValueError("tensor exceeds configured camera slots")
        if current == width:
            return value
        shape = list(value.shape)
        shape[normalized_axis] = width
        output = value.new_zeros(shape)
        slices = [slice(None)] * value.ndim
        slices[normalized_axis] = slice(0, current)
        output[tuple(slices)] = value
        return output

    def resolve_index(self, index: int) -> tuple[int, int]:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        task_index = bisect_right(self._offsets, normalized) - 1
        return task_index, normalized - self._offsets[task_index]

    def task_indices(self, task_index: int) -> range:
        return range(self._offsets[task_index], self._offsets[task_index + 1])

    def estimate_ram_preload_bytes(self) -> int:
        return sum(dataset.estimate_ram_preload_bytes() for dataset in self.datasets)

    def preload_to_ram(
        self,
        *,
        shared_memory: bool,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        total_episodes = sum(len(dataset.records) for dataset in self.datasets)
        completed_episodes = 0
        total_bytes = 0
        if progress_callback is not None:
            progress_callback(0, total_episodes, 0)
        for dataset in self.datasets:
            task_start = completed_episodes

            def update(current: int, _total: int, bytes_loaded: int) -> None:
                if progress_callback is not None:
                    progress_callback(task_start + current, total_episodes, total_bytes + bytes_loaded)

            report = dataset.preload_to_ram(
                shared_memory=shared_memory,
                progress_callback=update,
            )
            reports.append(report)
            completed_episodes += len(dataset.records)
            total_bytes += int(report["bytes"])
        return {
            "enabled": True,
            "shared_memory": bool(shared_memory),
            "tasks": len(self.datasets),
            "episodes": total_episodes,
            "bytes": total_bytes,
            "task_reports": reports,
        }

    def clear_ram_preload(self) -> None:
        for dataset in self.datasets:
            dataset.clear_ram_preload()

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def summary(self) -> dict[str, Any]:
        windows = {
            contract.task_id: len(dataset)
            for contract, dataset in zip(self.contracts, self.datasets, strict=True)
        }
        return {
            "dataset_protocol": ROBOFACTORY_M2_DATASET_PROTOCOL,
            "split": self.split,
            "tasks": len(self.datasets),
            "task_vocabulary": list(self.task_vocabulary),
            "windows": len(self),
            "windows_by_task": windows,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "max_agents": self.max_agents,
            "state_history": self.state_history,
            "action_horizon": self.action_horizon,
            "task_action_horizons": dict(self.task_action_horizons),
            "action_space": "per_task_zscore_canonical_unit_action",
            "visual_history": self.visual_history,
            "visual_history_alignment": "deployable_suffix_left_padding",
            "future_horizons": list(self.future_horizons),
            "camera_order": list(self.camera_order),
            "image_shape_hwc": list(self.image_shape_hwc),
            "contracts": [contract.to_dict() for contract in self.contracts],
        }

    def lineage_sha256(self) -> str:
        payload = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class CoverageTemperatureDistributedSampler(Sampler[int]):
    """Coverage-first sampler followed by square-root task mixing.

    Coverage epochs contain every indexed decision exactly once before any
    temperature-sampled filler. Later epochs draw tasks with
    ``p(task) ∝ window_count**temperature_alpha`` and cycle through a shuffled
    per-task permutation before repeating a local window.
    """

    def __init__(
        self,
        dataset: RoboFactoryMultitaskDataset,
        *,
        samples_per_epoch: int | None = None,
        coverage_epochs: int = 1,
        temperature_alpha: float = 0.5,
        seed: int = 0,
        rank: int = 0,
        replicas: int = 1,
    ) -> None:
        if replicas <= 0 or rank < 0 or rank >= replicas:
            raise ValueError("invalid distributed sampler rank/replicas")
        requested = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
        if requested <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if int(coverage_epochs) < 0:
            raise ValueError("coverage_epochs cannot be negative")
        if not 0.0 <= float(temperature_alpha) <= 1.0:
            raise ValueError("temperature_alpha must be in [0,1]")
        if int(coverage_epochs) and requested < len(dataset):
            raise ValueError(
                "coverage-first sampling requires samples_per_epoch >= dataset size"
            )
        self.dataset = dataset
        self.coverage_epochs = int(coverage_epochs)
        self.temperature_alpha = float(temperature_alpha)
        self.seed = int(seed)
        self.rank = int(rank)
        self.replicas = int(replicas)
        self.epoch = 0
        self.start_offset = 0
        self.global_samples = int(math.ceil(requested / replicas) * replicas)
        self.local_samples = self.global_samples // replicas

    def __len__(self) -> int:
        return self.local_samples - self.start_offset

    def set_epoch(self, epoch: int, *, start_offset: int = 0) -> None:
        normalized_epoch = int(epoch)
        normalized_offset = int(start_offset)
        if normalized_epoch < 0:
            raise ValueError("sampler epoch cannot be negative")
        if not 0 <= normalized_offset <= self.local_samples:
            raise ValueError("sampler start_offset lies outside the local epoch")
        self.epoch = normalized_epoch
        self.start_offset = normalized_offset

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices: list[int]
        if self.epoch < self.coverage_epochs:
            indices = torch.randperm(
                len(self.dataset), generator=generator
            ).tolist()
            indices.extend(
                self._temperature_indices(
                    self.global_samples - len(indices), generator=generator
                )
            )
        else:
            indices = self._temperature_indices(
                self.global_samples, generator=generator
            )
        local_indices = indices[self.rank :: self.replicas]
        return iter(local_indices[self.start_offset :])

    def _temperature_indices(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> list[int]:
        if count <= 0:
            return []
        sizes = torch.tensor(
            [len(dataset) for dataset in self.dataset.datasets],
            dtype=torch.float64,
        )
        probabilities = sizes.pow(self.temperature_alpha)
        probabilities /= probabilities.sum()
        assignments = torch.multinomial(
            probabilities,
            count,
            replacement=True,
            generator=generator,
        )
        result = [-1] * count
        for task_index in range(len(sizes)):
            positions = assignments.eq(task_index).nonzero().flatten().tolist()
            if not positions:
                continue
            local_count = int(sizes[task_index])
            sampled: list[int] = []
            while len(sampled) < len(positions):
                sampled.extend(
                    torch.randperm(local_count, generator=generator).tolist()
                )
            start = self.dataset._offsets[task_index]
            for position, local_index in zip(
                positions, sampled[: len(positions)], strict=True
            ):
                result[position] = start + local_index
        if any(index < 0 for index in result):  # pragma: no cover - invariant.
            raise RuntimeError("temperature sampler failed to assign an index")
        return result

    def summary(self) -> dict[str, Any]:
        sizes = np.asarray(
            [len(dataset) for dataset in self.dataset.datasets],
            dtype=np.float64,
        )
        probabilities = np.power(sizes, self.temperature_alpha)
        probabilities /= probabilities.sum()
        return {
            "strategy": "coverage_then_temperature_without_replacement_cycles",
            "coverage_epochs": self.coverage_epochs,
            "temperature_alpha": self.temperature_alpha,
            "samples_per_epoch": self.global_samples,
            "task_probabilities": {
                contract.task_id: float(probability)
                for contract, probability in zip(
                    self.dataset.contracts, probabilities, strict=True
                )
            },
        }


# Compatibility name for downstream imports. The semantics are intentionally
# upgraded from equal-task replacement to coverage-first temperature sampling.
TaskBalancedDistributedSampler = CoverageTemperatureDistributedSampler


__all__ = [
    "PAD_TOKEN_ID",
    "NATIVE_ROBOFACTORY_TASKS",
    "ROBOFACTORY_M2_DATASET_PROTOCOL",
    "RoboFactoryM2TaskContract",
    "RoboFactoryMultitaskDataset",
    "CoverageTemperatureDistributedSampler",
    "TaskBalancedDistributedSampler",
    "UTF8_VOCAB_SIZE",
    "encode_task_text",
]
