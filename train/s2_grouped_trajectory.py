"""S2 grouped-trajectory adapter preserving agent and shared-view axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from train.robofactory_multitask_dataset import RoboFactoryMultitaskDataset


S2_FUTURE_HORIZONS = (1, 25, 50, 100)
S2_MAX_AGENTS = 4
S2_STATE_DIM = 18
S2_ACTION_DIM = 8


@dataclass(frozen=True)
class S2SampleLineage:
    task_id: str
    task_index: int
    episode_index: int
    episode_seed: int
    decision_t: int


class S2GroupedTrajectoryDataset(Dataset[dict[str, Tensor]]):
    """Add stable episode lineage to the audited multi-task M1 windows."""

    def __init__(
        self,
        manifests: Sequence[str | Path],
        *,
        split: str,
        stride: int = 1,
        hdf5_cache_size: int = 4,
        verify_hdf5_sha256: bool = True,
        use_projected_future_cache: bool = False,
    ) -> None:
        self.source = RoboFactoryMultitaskDataset(
            manifests,
            split=split,
            state_history=1,
            action_horizon=max(S2_FUTURE_HORIZONS),
            task_action_horizons=None,
            visual_history=1,
            future_horizons=S2_FUTURE_HORIZONS,
            cameras=("global", "agent_0", "agent_1", "agent_2", "agent_3"),
            max_state_dim=S2_MAX_AGENTS * S2_STATE_DIM,
            max_action_dim=S2_MAX_AGENTS * S2_ACTION_DIM,
            max_agents=S2_MAX_AGENTS,
            max_text_tokens=16,
            stride=stride,
            hdf5_cache_size=hdf5_cache_size,
            sample_keys=(
                RoboFactoryMultitaskDataset.BASE_SAMPLE_KEYS.difference(
                    {"future_images", "future_image_novelty_mask"}
                )
                if use_projected_future_cache
                else None
            ),
            verify_hdf5_sha256=verify_hdf5_sha256,
        )
        self.contracts = self.source.contracts
        self.task_vocabulary = self.source.task_vocabulary
        self.split = str(split)
        self._hierarchical_indices_cache = self._build_hierarchical_indices()
        self._restore_hierarchical_indices_view()

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = dict(self.source[index])
        lineage = self.sample_lineage(index)
        sample.update(
            {
                "episode_index": torch.tensor(
                    lineage.episode_index, dtype=torch.long
                ),
                "episode_seed": torch.tensor(
                    lineage.episode_seed, dtype=torch.long
                ),
                "decision_t": torch.tensor(lineage.decision_t, dtype=torch.long),
            }
        )
        return sample

    def sample_lineage(self, index: int) -> S2SampleLineage:
        task_index, local_index = self.source.resolve_index(index)
        task_dataset = self.source.datasets[task_index]
        window = task_dataset._window(local_index)
        record = task_dataset.records[window.record_index]
        return S2SampleLineage(
            task_id=self.contracts[task_index].task_id,
            task_index=task_index,
            episode_index=int(record.episode_index),
            episode_seed=int(record.seed),
            decision_t=int(window.decision_t),
        )

    def task_indices(self, task_index: int) -> range:
        return self.source.task_indices(task_index)

    def hierarchical_indices(
        self,
    ) -> Mapping[str, Mapping[int, tuple[int, ...]]]:
        """Return the cached ``task -> episode -> time`` dataset indices.

        The innermost tuples are ordered by ``decision_t``.  The returned
        nested mappings are read-only so a sampler cannot accidentally change
        the shared dataset contract.  Building the cache only reads window
        metadata; it never calls ``__getitem__`` or opens episode payloads.
        """

        return self._hierarchical_indices_view

    def _build_hierarchical_indices(
        self,
    ) -> dict[str, dict[int, tuple[int, ...]]]:
        hierarchy: dict[str, dict[int, tuple[int, ...]]] = {}
        for task_index, (contract, task_dataset) in enumerate(
            zip(self.contracts, self.source.datasets, strict=True)
        ):
            task_id = str(contract.task_id)
            task_range = self.source.task_indices(task_index)
            windows = getattr(task_dataset, "_index", None)
            if windows is None:
                # Compatibility with an older metadata adapter.  This still
                # avoids materializing samples or opening HDF5 payloads.
                windows = tuple(
                    task_dataset._window(local_index)
                    for local_index in range(len(task_dataset))
                )
            if len(windows) != len(task_range):
                raise RuntimeError(
                    f"task {task_id!r} window metadata length drifted"
                )

            episode_windows: defaultdict[int, list[tuple[int, int]]] = (
                defaultdict(list)
            )
            for local_index, window in enumerate(windows):
                record = task_dataset.records[int(window.record_index)]
                episode_index = int(record.episode_index)
                decision_t = int(window.decision_t)
                dataset_index = task_range.start + local_index
                episode_windows[episode_index].append(
                    (decision_t, dataset_index)
                )

            task_hierarchy: dict[int, tuple[int, ...]] = {}
            for episode_index in sorted(episode_windows):
                ordered = sorted(episode_windows[episode_index])
                decision_times = [decision_t for decision_t, _ in ordered]
                if len(decision_times) != len(set(decision_times)):
                    raise RuntimeError(
                        f"task {task_id!r} episode {episode_index} has "
                        "duplicate decision_t windows"
                    )
                task_hierarchy[episode_index] = tuple(
                    dataset_index for _, dataset_index in ordered
                )
            if not task_hierarchy:
                raise RuntimeError(f"task {task_id!r} has no hierarchical windows")
            hierarchy[task_id] = task_hierarchy
        return hierarchy

    def _restore_hierarchical_indices_view(self) -> None:
        self._hierarchical_indices_view = MappingProxyType(
            {
                task_id: MappingProxyType(episodes)
                for task_id, episodes in self._hierarchical_indices_cache.items()
            }
        )

    def __getstate__(self) -> dict[str, Any]:
        # ``mappingproxy`` is intentionally read-only but not pickleable.
        # Workers reconstruct the cheap views around the cached plain dicts.
        state = dict(self.__dict__)
        state.pop("_hierarchical_indices_view", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        if not hasattr(self, "_hierarchical_indices_cache"):
            self._hierarchical_indices_cache = self._build_hierarchical_indices()
        self._restore_hierarchical_indices_view()

    def close(self) -> None:
        self.source.close()

    def summary(self) -> dict[str, Any]:
        value = self.source.summary()
        value.update(
            {
                "adapter": "s2_grouped_trajectory",
                "grouped_state_shape": [S2_MAX_AGENTS, S2_STATE_DIM],
                "grouped_action_shape": [
                    S2_MAX_AGENTS,
                    max(S2_FUTURE_HORIZONS),
                    S2_ACTION_DIM,
                ],
                "future_horizons": list(S2_FUTURE_HORIZONS),
                "global_view_is_separate": True,
                "raw_future_images_materialized": (
                    "future_images" in self.source.sample_keys
                ),
                "hierarchical_sampling": {
                    "order": ["task", "episode", "time", "all_valid_agent"],
                    "tasks": len(self._hierarchical_indices_cache),
                    "episodes": sum(
                        len(episodes)
                        for episodes in self._hierarchical_indices_cache.values()
                    ),
                    "team_windows": sum(
                        len(indices)
                        for episodes in self._hierarchical_indices_cache.values()
                        for indices in episodes.values()
                    ),
                },
            }
        )
        return value


def grouped_s2_batch(
    batch: Mapping[str, Tensor], *, require_future_images: bool = True
) -> dict[str, Tensor]:
    """Convert padded flat slots to the explicit ``[B,A,...]`` S2 contract."""

    states = batch["states"]
    actions = batch["action_targets"]
    images = batch["images"]
    future_states = batch["future_states"]
    future_images = batch.get("future_images")
    if states.ndim != 3 or states.shape[1:] != (
        1,
        S2_MAX_AGENTS * S2_STATE_DIM,
    ):
        raise ValueError("S2 states must be [B,1,72]")
    if actions.ndim != 3 or actions.shape[1:] != (
        max(S2_FUTURE_HORIZONS),
        S2_MAX_AGENTS * S2_ACTION_DIM,
    ):
        raise ValueError("S2 action targets must be [B,100,32]")
    if images.ndim != 6 or images.shape[1] != 1 or images.shape[2] != 5:
        raise ValueError("S2 current RGB must be [B,1,5,3,H,W]")
    if future_states.shape[1:] != (
        max(S2_FUTURE_HORIZONS),
        S2_MAX_AGENTS * S2_STATE_DIM,
    ):
        raise ValueError("S2 future state path must be [B,100,72]")
    if require_future_images:
        if (
            not isinstance(future_images, Tensor)
            or future_images.ndim != 6
            or future_images.shape[1:3] != (len(S2_FUTURE_HORIZONS), 5)
        ):
            raise ValueError("S2 future RGB must be [B,F,5,3,H,W]")
    elif future_images is not None:
        raise ValueError("projected-future-cache batches must omit raw future RGB")

    batch_size = states.shape[0]
    current_state = states[:, -1].reshape(
        batch_size, S2_MAX_AGENTS, S2_STATE_DIM
    )
    candidate_actions = actions.reshape(
        batch_size,
        max(S2_FUTURE_HORIZONS),
        S2_MAX_AGENTS,
        S2_ACTION_DIM,
    ).permute(0, 2, 1, 3).contiguous()
    agent_count = batch["embodiment_index"] + 1
    slot_valid = (
        torch.arange(S2_MAX_AGENTS, device=agent_count.device)[None]
        < agent_count[:, None]
    )
    current_image_valid = batch["image_valid_mask"][:, -1]
    physical_agent_camera_valid = slot_valid & current_image_valid[:, 1:5]
    global_fallback_valid = (
        slot_valid
        & ~physical_agent_camera_valid
        & current_image_valid[:, 0, None]
    )
    agent_valid = physical_agent_camera_valid | global_fallback_valid
    agent_observations = images[:, -1, 1:5].clone()
    if bool(global_fallback_valid.any()):
        global_observations = images[:, -1, 0, None].expand(
            -1, S2_MAX_AGENTS, -1, -1, -1
        )
        agent_observations[global_fallback_valid] = global_observations[
            global_fallback_valid
        ]

    future_indices = torch.tensor(
        [value - 1 for value in S2_FUTURE_HORIZONS],
        dtype=torch.long,
        device=future_states.device,
    )
    selected_future = future_states.index_select(1, future_indices).reshape(
        batch_size,
        len(S2_FUTURE_HORIZONS),
        S2_MAX_AGENTS,
        S2_STATE_DIM,
    ).permute(0, 2, 1, 3).contiguous()
    future_state_delta = selected_future - current_state[:, :, None]
    future_state_time_valid = batch["future_state_valid_mask"].index_select(
        1, future_indices
    )
    future_state_valid = (
        agent_valid[:, :, None] & future_state_time_valid[:, None]
    )
    future_visual_valid = batch["future_visual_valid_mask"].bool()
    agent_future_visual_valid = (
        physical_agent_camera_valid[:, :, None]
        & future_visual_valid[:, :, 1:5].permute(0, 2, 1)
    )
    shared_future_visual_valid = future_visual_valid[:, :, 0]
    action_valid_mask = batch.get("action_target_valid_mask")
    if action_valid_mask is None:
        # Compatibility for synthetic contract fixtures created before S3.
        action_valid_mask = torch.ones(
            batch_size,
            max(S2_FUTURE_HORIZONS),
            dtype=torch.bool,
            device=actions.device,
        )

    output = {
        "dataset_index": batch["dataset_index"],
        "task_index": batch["task_index"],
        "episode_index": batch["episode_index"],
        "episode_seed": batch["episode_seed"],
        "decision_t": batch["decision_t"],
        "current_state": current_state,
        "candidate_actions": candidate_actions,
        "action_valid_mask": action_valid_mask.bool(),
        "agent_observations": agent_observations,
        "shared_observation": images[:, -1, 0],
        "shared_observation_valid_mask": current_image_valid[:, 0],
        "valid_agent_mask": agent_valid,
        "agent_camera_valid_mask": physical_agent_camera_valid,
        "agent_global_fallback_mask": global_fallback_valid,
        "future_state_delta": future_state_delta,
        "future_state_valid_mask": future_state_valid,
        "future_agent_visual_valid_mask": agent_future_visual_valid,
        "future_shared_visual_valid_mask": shared_future_visual_valid,
        "future_horizons": batch["future_horizons"],
    }
    if isinstance(future_images, Tensor):
        output["future_agent_observations"] = future_images[:, :, 1:5].permute(
            0, 2, 1, 3, 4, 5
        ).contiguous()
        output["future_shared_observations"] = future_images[:, :, 0]
    validate_grouped_s2_contract(output)
    return output


def validate_grouped_s2_contract(batch: Mapping[str, Tensor]) -> None:
    state = batch["current_state"]
    if state.ndim != 3 or state.shape[1:] != (
        S2_MAX_AGENTS,
        S2_STATE_DIM,
    ):
        raise ValueError("current agent state must be [B,4,18]")
    batch_size = state.shape[0]
    expected = {
        "candidate_actions": (
            batch_size,
            S2_MAX_AGENTS,
            max(S2_FUTURE_HORIZONS),
            S2_ACTION_DIM,
        ),
        "action_valid_mask": (
            batch_size,
            max(S2_FUTURE_HORIZONS),
        ),
        "valid_agent_mask": (batch_size, S2_MAX_AGENTS),
        "agent_camera_valid_mask": (batch_size, S2_MAX_AGENTS),
        "agent_global_fallback_mask": (batch_size, S2_MAX_AGENTS),
        "future_state_delta": (
            batch_size,
            S2_MAX_AGENTS,
            len(S2_FUTURE_HORIZONS),
            S2_STATE_DIM,
        ),
        "future_state_valid_mask": (
            batch_size,
            S2_MAX_AGENTS,
            len(S2_FUTURE_HORIZONS),
        ),
        "future_agent_visual_valid_mask": (
            batch_size,
            S2_MAX_AGENTS,
            len(S2_FUTURE_HORIZONS),
        ),
        "future_shared_visual_valid_mask": (
            batch_size,
            len(S2_FUTURE_HORIZONS),
        ),
        "shared_observation_valid_mask": (batch_size,),
    }
    for name, shape in expected.items():
        if tuple(batch[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if batch["agent_observations"].shape[:2] != (
        batch_size,
        S2_MAX_AGENTS,
    ):
        raise ValueError("agent observations must retain [B,A,...]")
    if batch["shared_observation"].shape[0] != batch_size:
        raise ValueError("shared observation must retain a separate batch slot")


__all__ = [
    "S2_ACTION_DIM",
    "S2_FUTURE_HORIZONS",
    "S2_MAX_AGENTS",
    "S2_STATE_DIM",
    "S2GroupedTrajectoryDataset",
    "S2SampleLineage",
    "grouped_s2_batch",
    "validate_grouped_s2_contract",
]
