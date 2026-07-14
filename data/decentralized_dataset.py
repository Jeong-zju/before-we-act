"""Ego-indexed training windows for the FE-PC-WAM schema."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.local_observation import (
    LocalObservationSpec,
    flatten_observation_mapping,
)
from data.schema import (
    LEGACY_CONTACT_SEMANTICS,
    LEGACY_FORCE_SEMANTICS,
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    read_local_observations,
    spec_from_hdf5,
)


@dataclass(frozen=True)
class DecentralizedSampleSpec:
    file_idx: int
    decision_t: int
    ego_id: int


class ProjectedDatasetView(Dataset):
    """A DataLoader-compatible, zero-copy key projection of a dataset.

    The view owns no HDF5 handles and delegates each read to
    :meth:`DecentralizedTransitionDataset.get_selected`.  Keeping ``index``
    visible also makes the view compatible with file-grouped batch samplers.
    """

    def __init__(
        self,
        dataset: "DecentralizedTransitionDataset",
        keys: Sequence[str],
    ) -> None:
        self.dataset = dataset
        self.keys = tuple(dict.fromkeys(str(key) for key in keys))
        if not self.keys:
            raise ValueError("projected dataset keys cannot be empty")
        unknown = set(self.keys) - set(dataset.sample_keys)
        if unknown:
            raise KeyError(f"unknown dataset projection keys: {sorted(unknown)}")

    @property
    def index(self) -> List[DecentralizedSampleSpec]:
        return self.dataset.index

    @property
    def paths(self) -> list[Path]:
        return self.dataset.paths

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_selected(index, self.keys)


class DecentralizedTransitionDataset(Dataset):
    """Return one ego robot's deployable history and future supervision.

    A sample at decision index ``t`` uses observations ending at ``o_t``.  The
    action beside each history row is the *previous own action*: observation
    ``o_tau`` is paired with ``a_(tau-1)``.  It never contains ``a_t``.

    ``padding_mask`` follows the PyTorch transformer convention: ``True`` marks
    left-padding that must be ignored.  Privileged keys are targets only; the
    ``flat_observation_history`` retains the complete deployable packet for
    diagnostics.  ``model_history``/``local_history`` exclude the separately
    encoded six-dimensional object estimate and concatenate only onboard/task
    features with the previous ego action.  The teammate's deployable
    observation group is never opened.
    """

    INPUT_KEYS = frozenset(
        {
            "ego_id",
            "local_history",
            "history_mask",
            "object_observation_history",
            "object_valid_history",
            "object_confidence_history",
            "object_age_history",
        }
    )
    SAMPLE_KEYS = frozenset(
        {
            "ego_id",
            "decision_t",
            "flat_observation_history",
            "model_observation_history",
            "prev_action_history",
            "model_history",
            "local_history",
            "padding_mask",
            "history_mask",
            "history_valid_mask",
            "prev_action_valid_mask",
            "object_observation_history",
            "object_valid_history",
            "object_confidence_history",
            "object_age_history",
            "object_observation",
            "object_valid",
            "object_confidence",
            "object_age",
            "ego_future_observation",
            "future_model_observation",
            "future_object_observation",
            "future_object_valid",
            "future_object_confidence",
            "future_object_age",
            "target_local_force",
            "target_local_contact",
            "ego_future_action",
            "privileged_teammate_future_action",
            "target_object_pose_world",
            "target_object_pose_ego",
            "target_robot_pose_world",
            "target_teammate_pose_ego",
            "target_reward",
            "target_done",
            "target_success",
            "target_failure",
            "target_failure_reason",
            "target_phase",
            "target_force_proxy",
            "target_contact",
            "target_grasp",
            "target_private_event_type",
            "target_private_event_informed_agent",
            "target_private_event_maneuver",
            "target_private_event_error",
            "target_return",
            "target_collision",
            "target_force_violation",
            "target_maneuver",
            "target_progress",
            "branch_valid",
            "branch_plan_pair",
            "branch_action",
            "branch_return",
            "branch_success",
            "branch_constraint_violation",
        }
    )

    def __init__(
        self,
        data_dir: str | Path,
        *,
        history: int = 8,
        horizon: int = 16,
        stride: int = 1,
        ego_ids: Iterable[int] = (0, 1),
        max_episodes: int = -1,
        expected_schema_version: str = SCHEMA_VERSION,
        hdf5_cache_size: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.ego_ids = tuple(int(value) for value in ego_ids)
        self.expected_schema_version = str(expected_schema_version)
        self.hdf5_cache_size = int(hdf5_cache_size)
        self._hdf5_cache_pid = os.getpid()
        self._hdf5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        if self.history <= 0 or self.horizon <= 0 or self.stride <= 0:
            raise ValueError("history, horizon, and stride must be positive")
        if self.hdf5_cache_size < 0:
            raise ValueError("hdf5_cache_size cannot be negative")
        if not self.ego_ids or len(set(self.ego_ids)) != len(self.ego_ids):
            raise ValueError("ego_ids must be a non-empty unique sequence")

        self.paths = sorted(self.data_dir.glob("episode_*.hdf5"))
        if max_episodes > 0:
            self.paths = self.paths[:max_episodes]
        if not self.paths:
            raise FileNotFoundError(f"No episode_*.hdf5 found in {self.data_dir}")

        self.index: List[DecentralizedSampleSpec] = []
        self.spec: LocalObservationSpec | None = None
        self.action_dim: int | None = None
        num_agents: int | None = None
        self.local_contact_semantics: str | None = None
        self.local_force_semantics: str | None = None
        self.local_force_scale_newtons: float | None = None
        self.local_force_units: str | None = None
        self.local_sensor_provenance: str | None = None
        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as file:
                if str(file.attrs.get("schema_version", "")) != self.expected_schema_version:
                    raise ValueError(
                        f"{path} is not a {self.expected_schema_version} episode"
                    )
                file_spec = spec_from_hdf5(
                    file, expected_schema_version=self.expected_schema_version
                )
                contact_semantics = str(
                    file.attrs.get(
                        "local_contact_semantics", LEGACY_CONTACT_SEMANTICS
                    )
                )
                if self.local_contact_semantics is None:
                    self.local_contact_semantics = contact_semantics
                elif contact_semantics != self.local_contact_semantics:
                    raise ValueError("local contact semantics changed across episodes")
                force_semantics = str(
                    file.attrs.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
                )
                if self.local_force_semantics is None:
                    self.local_force_semantics = force_semantics
                elif force_semantics != self.local_force_semantics:
                    raise ValueError("local force semantics changed across episodes")
                raw_force_scale = file.attrs.get("local_force_scale_newtons")
                force_scale = (
                    None if raw_force_scale is None else float(raw_force_scale)
                )
                if force_scale is not None and (
                    not np.isfinite(force_scale) or force_scale <= 0.0
                ):
                    raise ValueError("local force scale must be finite and positive")
                force_units = file.attrs.get("local_force_units")
                force_units = None if force_units is None else str(force_units)
                sensor_provenance = file.attrs.get("local_sensor_provenance")
                sensor_provenance = (
                    None if sensor_provenance is None else str(sensor_provenance)
                )
                if (
                    contact_semantics == STRICT_LOCAL_CONTACT_SEMANTICS
                    or force_semantics == STRICT_LOCAL_FORCE_SEMANTICS
                ):
                    if (
                        contact_semantics != STRICT_LOCAL_CONTACT_SEMANTICS
                        or force_semantics != STRICT_LOCAL_FORCE_SEMANTICS
                        or force_scale is None
                        or force_units != LOCAL_FORCE_UNITS
                        or sensor_provenance != STRICT_LOCAL_SENSOR_PROVENANCE
                    ):
                        raise ValueError(
                            "strict local sensor episodes require paired semantics, "
                            "normalized units, scale, and explicit provenance"
                        )
                if file_idx == 0:
                    self.local_force_scale_newtons = force_scale
                elif (force_scale is None) != (
                    self.local_force_scale_newtons is None
                ) or (
                    force_scale is not None
                    and self.local_force_scale_newtons is not None
                    and not np.isclose(force_scale, self.local_force_scale_newtons)
                ):
                    raise ValueError("local force scale changed across episodes")
                if file_idx == 0:
                    self.local_force_units = force_units
                elif force_units != self.local_force_units:
                    raise ValueError("local force units changed across episodes")
                if file_idx == 0:
                    self.local_sensor_provenance = sensor_provenance
                elif sensor_provenance != self.local_sensor_provenance:
                    raise ValueError("local sensor provenance changed across episodes")
                if self.spec is None:
                    self.spec = file_spec
                elif file_spec != self.spec:
                    raise ValueError(f"observation spec changed across episodes: {path}")

                file_num_agents = int(file.attrs["num_agents"])
                if num_agents is None:
                    num_agents = file_num_agents
                elif file_num_agents != num_agents:
                    raise ValueError("num_agents changed across episodes")
                for ego_id in self.ego_ids:
                    if ego_id < 0 or ego_id >= file_num_agents:
                        raise ValueError(f"ego_id {ego_id} unavailable in {path}")

                transitions = int(file.attrs["num_transitions"])
                file_action_dim = int(file[f"transitions/actions/agent_{self.ego_ids[0]}"].shape[1])
                if self.action_dim is None:
                    self.action_dim = file_action_dim
                elif file_action_dim != self.action_dim:
                    raise ValueError("action_dim changed across episodes")

                max_t = transitions - self.horizon
                for decision_t in range(0, max_t + 1, self.stride):
                    for ego_id in self.ego_ids:
                        self.index.append(
                            DecentralizedSampleSpec(
                                file_idx=file_idx,
                                decision_t=decision_t,
                                ego_id=ego_id,
                            )
                        )

        if not self.index:
            raise RuntimeError("No valid  windows. Reduce horizon or collect longer episodes.")
        assert self.spec is not None and self.action_dim is not None
        assert self.local_contact_semantics is not None
        assert self.local_force_semantics is not None

    @property
    def observation_dim(self) -> int:
        assert self.spec is not None
        return self.spec.flat_dim

    @property
    def local_history_dim(self) -> int:
        assert self.action_dim is not None
        return self.spec.model_observation_dim + self.action_dim

    @property
    def model_observation_dim(self) -> int:
        return self.spec.model_observation_dim

    @property
    def input_feature_names(self) -> list[str]:
        assert self.spec is not None and self.action_dim is not None
        return self.spec.model_feature_names() + [
            f"previous_ego_action_{index}" for index in range(self.action_dim)
        ]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def sample_keys(self) -> frozenset[str]:
        return self.SAMPLE_KEYS

    def project(self, keys: Sequence[str]) -> ProjectedDatasetView:
        """Return a lightweight dataset that reads and returns only ``keys``."""

        return ProjectedDatasetView(self, keys)

    @contextmanager
    def _open_hdf5(self, path: str | Path) -> Iterator[h5py.File]:
        """Reuse a small number of read-only files inside each loader worker.

        Research-v2 reads several groups for every sample.  Reopening the same
        HDF5 file for each group dominates training time on millions of windows.
        The cache is process-local, bounded, and reset after a fork/spawn so an
        h5py handle is never shared across worker processes.
        """

        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        pid = os.getpid()
        if pid != self._hdf5_cache_pid:
            self._close_hdf5_cache()
            self._hdf5_cache_pid = pid
        key = str(Path(path).resolve())
        file = self._hdf5_cache.pop(key, None)
        if file is None or not file.id.valid:
            file = h5py.File(key, "r")
        self._hdf5_cache[key] = file
        while len(self._hdf5_cache) > self.hdf5_cache_size:
            _, stale = self._hdf5_cache.popitem(last=False)
            stale.close()
        yield file

    def _close_hdf5_cache(self) -> None:
        cache = getattr(self, "_hdf5_cache", {})
        for file in cache.values():
            try:
                file.close()
            except Exception:
                pass
        cache.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_hdf5_cache"] = OrderedDict()
        state["_hdf5_cache_pid"] = os.getpid()
        return state

    def __del__(self):
        self._close_hdf5_cache()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_spec = self.index[index]
        path = self.paths[sample_spec.file_idx]
        t = sample_spec.decision_t
        ego_id = sample_spec.ego_id
        teammate_id = 1 - ego_id
        history_start = max(0, t - self.history + 1)
        history_count = t - history_start + 1
        pad_count = self.history - history_count
        future_slice = slice(t, t + self.horizon)
        post_observation_slice = slice(t + 1, t + self.horizon + 1)

        flat_observation_history = np.zeros(
            (self.history, self.observation_dim), dtype=np.float32
        )
        model_observation_history = np.zeros(
            (self.history, self.spec.model_observation_dim), dtype=np.float32
        )
        object_observation_history = np.zeros((self.history, 3), dtype=np.float32)
        object_valid_history = np.zeros(self.history, dtype=np.bool_)
        object_confidence_history = np.zeros(self.history, dtype=np.float32)
        object_age_history = np.zeros(self.history, dtype=np.float32)
        previous_action_history = np.zeros(
            (self.history, self.action_dim), dtype=np.float32
        )
        padding_mask = np.ones(self.history, dtype=np.bool_)
        previous_action_valid = np.zeros(self.history, dtype=np.bool_)

        with self._open_hdf5(path) as file:
            # This is the sole observation read.  In particular, no
            # observations/agent_{teammate_id} path is accessed.
            ego_mapping = read_local_observations(
                file,
                ego_id,
                self.spec,
                slice(history_start, t + 1),
            )
            flat_observation_history[pad_count:] = flatten_observation_mapping(
                ego_mapping, self.spec
            )
            model_observation_history[pad_count:] = _flatten_selected_fields(
                ego_mapping, self.spec.model_field_names()
            )
            object_observation_history[pad_count:] = ego_mapping[
                "estimates/object/pose"
            ]
            object_valid_history[pad_count:] = (
                ego_mapping["estimates/object/valid"].reshape(-1) > 0.5
            )
            object_confidence_history[pad_count:] = ego_mapping[
                "estimates/object/confidence"
            ].reshape(-1)
            object_age_history[pad_count:] = ego_mapping[
                "estimates/object/age"
            ].reshape(-1)
            padding_mask[pad_count:] = False

            future_ego_mapping = read_local_observations(
                file,
                ego_id,
                self.spec,
                post_observation_slice,
            )
            ego_future_observation = flatten_observation_mapping(
                future_ego_mapping, self.spec
            )
            future_model_observation = _flatten_selected_fields(
                future_ego_mapping, self.spec.model_field_names()
            )

            ego_actions_ds = file[f"transitions/actions/agent_{ego_id}"]
            real_times = np.arange(history_start, t + 1, dtype=np.int64)
            action_rows = real_times > 0
            if np.any(action_rows):
                action_indices = real_times[action_rows] - 1
                # HDF5 supports a sorted integer selection.  These indices are
                # exactly a_(tau-1), never the current a_t.
                previous_action_history[pad_count + np.flatnonzero(action_rows)] = ego_actions_ds[
                    action_indices
                ]
                previous_action_valid[pad_count + np.flatnonzero(action_rows)] = True

            ego_future_action = ego_actions_ds[future_slice]
            # The teammate action is an explicitly privileged intention label,
            # not an input or continuously available state.
            teammate_future_action = file[
                f"transitions/actions/agent_{teammate_id}"
            ][future_slice]

            privileged_obs = file["privileged/observations"]
            privileged_tr = file["privileged/transitions"]
            target_object_pose_world = privileged_obs["object_pose_world"][
                post_observation_slice
            ]
            target_object_pose_ego = privileged_obs["object_pose_ego"][
                post_observation_slice, ego_id
            ]
            target_robot_pose_world = privileged_obs["robot_pose_world"][
                post_observation_slice
            ]
            target_teammate_pose_ego = privileged_obs["teammate_pose_ego"][
                post_observation_slice, ego_id
            ]

            result = {
                "ego_id": torch.tensor(ego_id, dtype=torch.long),
                "decision_t": torch.tensor(t, dtype=torch.long),
                "flat_observation_history": torch.from_numpy(flat_observation_history),
                "model_observation_history": torch.from_numpy(model_observation_history),
                "prev_action_history": torch.from_numpy(previous_action_history),
                "model_history": torch.from_numpy(
                    np.concatenate(
                        [model_observation_history, previous_action_history], axis=-1
                    ).astype(np.float32)
                ),
                "padding_mask": torch.from_numpy(padding_mask),
                "history_mask": torch.from_numpy(~padding_mask),
                "history_valid_mask": torch.from_numpy(~padding_mask),
                "prev_action_valid_mask": torch.from_numpy(previous_action_valid),
                "object_observation_history": torch.from_numpy(object_observation_history),
                "object_valid_history": torch.from_numpy(object_valid_history),
                "object_confidence_history": torch.from_numpy(object_confidence_history),
                "object_age_history": torch.from_numpy(object_age_history),
                "object_observation": torch.from_numpy(object_observation_history[-1]),
                "object_valid": torch.tensor(object_valid_history[-1], dtype=torch.bool),
                "object_confidence": torch.tensor(
                    object_confidence_history[-1], dtype=torch.float32
                ),
                "object_age": torch.tensor(object_age_history[-1], dtype=torch.float32),
                "ego_future_observation": torch.tensor(
                    ego_future_observation, dtype=torch.float32
                ),
                "future_model_observation": torch.tensor(
                    future_model_observation, dtype=torch.float32
                ),
                "future_object_observation": torch.tensor(
                    future_ego_mapping["estimates/object/pose"], dtype=torch.float32
                ),
                "future_object_valid": torch.tensor(
                    future_ego_mapping["estimates/object/valid"].reshape(-1) > 0.5,
                    dtype=torch.bool,
                ),
                "future_object_confidence": torch.tensor(
                    future_ego_mapping["estimates/object/confidence"].reshape(-1),
                    dtype=torch.float32,
                ),
                "future_object_age": torch.tensor(
                    future_ego_mapping["estimates/object/age"].reshape(-1),
                    dtype=torch.float32,
                ),
                "target_local_force": torch.tensor(
                    future_ego_mapping["local/force"].reshape(self.horizon, -1)[:, 0],
                    dtype=torch.float32,
                ),
                "target_local_contact": torch.tensor(
                    future_ego_mapping["local/contact"].reshape(-1),
                    dtype=torch.float32,
                ),
                "ego_future_action": torch.tensor(
                    ego_future_action, dtype=torch.float32
                ),
                "privileged_teammate_future_action": torch.tensor(
                    teammate_future_action, dtype=torch.float32
                ),
                "target_object_pose_world": torch.tensor(
                    target_object_pose_world, dtype=torch.float32
                ),
                "target_object_pose_ego": torch.tensor(
                    target_object_pose_ego, dtype=torch.float32
                ),
                "target_robot_pose_world": torch.tensor(
                    target_robot_pose_world, dtype=torch.float32
                ),
                "target_teammate_pose_ego": torch.tensor(
                    target_teammate_pose_ego, dtype=torch.float32
                ),
            }

            for key in (
                "reward",
                "done",
                "success",
                "failure",
                "failure_reason",
                "phase",
                "force_proxy",
                "contact",
                "grasp",
            ):
                value = privileged_tr[key][future_slice]
                dtype = torch.long if key in {
                    "failure_reason",
                    "phase",
                    "private_event_type",
                    "private_event_informed_agent",
                    "private_event_maneuver",
                } else torch.float32
                result[f"target_{key}"] = torch.tensor(value, dtype=dtype)

            private_defaults = {
                "private_event_type": -1,
                "private_event_informed_agent": -1,
                "private_event_maneuver": 0,
                "private_event_error": 0.0,
            }
            for key, default in private_defaults.items():
                if key in privileged_tr:
                    value = privileged_tr[key][future_slice]
                else:
                    value = np.full((self.horizon, 1), default)
                dtype = torch.long if key != "private_event_error" else torch.float32
                result[f"target_{key}"] = torch.tensor(value, dtype=dtype)

            reward_values = privileged_tr["reward"][future_slice].reshape(-1)
            result["target_return"] = torch.tensor(
                float(np.sum(reward_values)), dtype=torch.float32
            )
            result["target_collision"] = result["target_contact"].clone()
            result["target_force_violation"] = (
                result["target_force_proxy"].reshape(-1) > 1.0
            ).to(torch.float32)
            result["target_maneuver"] = (
                result["target_private_event_maneuver"].reshape(-1)[0] + 1
            ).clamp(0, 2).to(torch.long)
            branch_count = 6
            if "branch_action" in privileged_tr:
                branch_action = np.asarray(
                    privileged_tr["branch_action"][t], dtype=np.float32
                )
                branch_count = int(branch_action.shape[0])
                fitted_action = np.zeros(
                    (branch_count, 2, self.horizon, self.action_dim),
                    dtype=np.float32,
                )
                copied_horizon = min(self.horizon, int(branch_action.shape[2]))
                fitted_action[:, :, :copied_horizon] = branch_action[
                    :, :, :copied_horizon
                ]
                branch_valid = np.asarray(
                    privileged_tr["branch_valid"][t], dtype=np.bool_
                )
                branch_plan_pair = privileged_tr["branch_plan_pair"][t]
                branch_return = privileged_tr["branch_return"][t]
                branch_success = privileged_tr["branch_success"][t]
                branch_constraint = privileged_tr[
                    "branch_constraint_violation"
                ][t]
            else:
                fitted_action = np.zeros(
                    (branch_count, 2, self.horizon, self.action_dim),
                    dtype=np.float32,
                )
                branch_valid = np.zeros(branch_count, dtype=np.bool_)
                branch_plan_pair = np.zeros((branch_count, 2), dtype=np.float32)
                branch_return = np.zeros(branch_count, dtype=np.float32)
                branch_success = np.zeros(branch_count, dtype=np.float32)
                branch_constraint = np.zeros(branch_count, dtype=np.float32)
            result["branch_valid"] = torch.tensor(branch_valid, dtype=torch.bool)
            result["branch_plan_pair"] = torch.tensor(
                branch_plan_pair, dtype=torch.float32
            )
            result["branch_action"] = torch.tensor(fitted_action, dtype=torch.float32)
            result["branch_return"] = torch.tensor(branch_return, dtype=torch.float32)
            result["branch_success"] = torch.tensor(branch_success, dtype=torch.float32)
            result["branch_constraint_violation"] = torch.tensor(
                branch_constraint, dtype=torch.float32
            )

            # Progress is a privileged post-transition training target.  Keep
            # the public shape [H] rather than the HDF5 scalar-column shape.
            result["target_progress"] = torch.tensor(
                privileged_tr["progress"][future_slice].reshape(-1),
                dtype=torch.float32,
            )

            # Compatibility alias: both names intentionally contain identical
            # non-object model input, never the full flat packet.
            result["local_history"] = result["model_history"]

        return result

    def get_selected(
        self,
        index: int,
        keys: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Read only the HDF5 fields needed to materialize ``keys``.

        This is intentionally separate from :meth:`__getitem__`: callers that
        rely on the historical full sample contract see no behavior change,
        while stage-specific DataLoaders can avoid constructing future,
        privileged, and branch payloads that their model never consumes.
        """

        requested = frozenset(str(key) for key in keys)
        if not requested:
            raise ValueError("selected dataset keys cannot be empty")
        unknown = requested - DecentralizedTransitionDataset.SAMPLE_KEYS
        if unknown:
            raise KeyError(f"unknown base dataset keys: {sorted(unknown)}")

        sample_spec = self.index[index]
        path = self.paths[sample_spec.file_idx]
        t = sample_spec.decision_t
        ego_id = sample_spec.ego_id
        teammate_id = 1 - ego_id
        history_start = max(0, t - self.history + 1)
        history_count = t - history_start + 1
        pad_count = self.history - history_count
        history_slice = slice(history_start, t + 1)
        future_slice = slice(t, t + self.horizon)
        post_observation_slice = slice(t + 1, t + self.horizon + 1)

        result: Dict[str, torch.Tensor] = {}
        if "ego_id" in requested:
            result["ego_id"] = torch.tensor(ego_id, dtype=torch.long)
        if "decision_t" in requested:
            result["decision_t"] = torch.tensor(t, dtype=torch.long)

        padding_mask = np.ones(self.history, dtype=np.bool_)
        padding_mask[pad_count:] = False
        for name, value in (
            ("padding_mask", padding_mask),
            ("history_mask", ~padding_mask),
            ("history_valid_mask", ~padding_mask),
        ):
            if name in requested:
                result[name] = torch.from_numpy(value.copy())
        if "prev_action_valid_mask" in requested:
            valid = np.zeros(self.history, dtype=np.bool_)
            real_times = np.arange(history_start, t + 1, dtype=np.int64)
            valid[pad_count:] = real_times > 0
            result["prev_action_valid_mask"] = torch.from_numpy(valid)

        metadata_only = {
            "ego_id",
            "decision_t",
            "padding_mask",
            "history_mask",
            "history_valid_mask",
            "prev_action_valid_mask",
        }
        if requested <= metadata_only:
            return result

        with self._open_hdf5(path) as file:
            history_all = "flat_observation_history" in requested
            history_model = bool(
                requested
                & {
                    "model_observation_history",
                    "model_history",
                    "local_history",
                }
            )
            object_history_outputs = {
                "object_observation_history": "estimates/object/pose",
                "object_valid_history": "estimates/object/valid",
                "object_confidence_history": "estimates/object/confidence",
                "object_age_history": "estimates/object/age",
            }
            object_current_outputs = {
                "object_observation": "estimates/object/pose",
                "object_valid": "estimates/object/valid",
                "object_confidence": "estimates/object/confidence",
                "object_age": "estimates/object/age",
            }
            history_fields: set[str] = set()
            if history_all:
                history_fields.update(self.spec.field_shapes())
            if history_model:
                history_fields.update(self.spec.model_field_names())
            for output, field_name in object_history_outputs.items():
                if output in requested:
                    history_fields.add(field_name)
            # Current values reuse a history read when the same field is already
            # requested; otherwise only o_t is touched.
            for output, field_name in object_current_outputs.items():
                if output in requested and field_name in history_fields:
                    continue
                if output in requested:
                    current = _read_selected_local_observations(
                        file, ego_id, (field_name,), slice(t, t + 1)
                    )[field_name][0]
                    result[output] = _object_tensor(output, current)

            history_mapping: dict[str, np.ndarray] = {}
            if history_fields:
                history_mapping = _read_selected_local_observations(
                    file, ego_id, history_fields, history_slice
                )
                padded_mapping = {
                    name: _left_pad(value, self.history, pad_count)
                    for name, value in history_mapping.items()
                }
                if "flat_observation_history" in requested:
                    result["flat_observation_history"] = torch.from_numpy(
                        flatten_observation_mapping(padded_mapping, self.spec)
                    )
                model_observation_history: np.ndarray | None = None
                if history_model:
                    model_observation_history = _flatten_selected_fields(
                        padded_mapping, self.spec.model_field_names()
                    )
                    if "model_observation_history" in requested:
                        result["model_observation_history"] = torch.from_numpy(
                            model_observation_history
                        )
                for output, field_name in object_history_outputs.items():
                    if output not in requested:
                        continue
                    value = padded_mapping[field_name]
                    if output == "object_valid_history":
                        tensor = torch.from_numpy(value.reshape(-1) > 0.5)
                    elif output in {
                        "object_confidence_history",
                        "object_age_history",
                    }:
                        tensor = torch.from_numpy(value.reshape(-1))
                    else:
                        tensor = torch.from_numpy(value)
                    result[output] = tensor
                for output, field_name in object_current_outputs.items():
                    if output in requested and output not in result:
                        result[output] = _object_tensor(
                            output, padded_mapping[field_name][-1]
                        )
            else:
                model_observation_history = None

            need_previous_action = bool(
                requested
                & {
                    "prev_action_history",
                    "model_history",
                    "local_history",
                }
            )
            previous_action_history: np.ndarray | None = None
            if need_previous_action:
                previous_action_history = np.zeros(
                    (self.history, self.action_dim), dtype=np.float32
                )
                real_times = np.arange(history_start, t + 1, dtype=np.int64)
                action_rows = real_times > 0
                if np.any(action_rows):
                    action_indices = real_times[action_rows] - 1
                    previous_action_history[
                        pad_count + np.flatnonzero(action_rows)
                    ] = file[f"transitions/actions/agent_{ego_id}"][action_indices]
                if "prev_action_history" in requested:
                    result["prev_action_history"] = torch.from_numpy(
                        previous_action_history
                    )
            if requested & {"model_history", "local_history"}:
                assert model_observation_history is not None
                assert previous_action_history is not None
                combined = torch.from_numpy(
                    np.concatenate(
                        [model_observation_history, previous_action_history], axis=-1
                    ).astype(np.float32)
                )
                if "model_history" in requested:
                    result["model_history"] = combined
                if "local_history" in requested:
                    result["local_history"] = combined

            future_all = "ego_future_observation" in requested
            future_model = "future_model_observation" in requested
            future_local_outputs = {
                "future_object_observation": "estimates/object/pose",
                "future_object_valid": "estimates/object/valid",
                "future_object_confidence": "estimates/object/confidence",
                "future_object_age": "estimates/object/age",
                "target_local_force": "local/force",
                "target_local_contact": "local/contact",
            }
            future_fields: set[str] = set()
            if future_all:
                future_fields.update(self.spec.field_shapes())
            if future_model:
                future_fields.update(self.spec.model_field_names())
            for output, field_name in future_local_outputs.items():
                if output in requested:
                    future_fields.add(field_name)
            if future_fields:
                future_mapping = _read_selected_local_observations(
                    file, ego_id, future_fields, post_observation_slice
                )
                if future_all:
                    result["ego_future_observation"] = torch.from_numpy(
                        flatten_observation_mapping(future_mapping, self.spec)
                    )
                if future_model:
                    result["future_model_observation"] = torch.from_numpy(
                        _flatten_selected_fields(
                            future_mapping, self.spec.model_field_names()
                        )
                    )
                for output, field_name in future_local_outputs.items():
                    if output not in requested:
                        continue
                    value = future_mapping[field_name]
                    if output == "future_object_valid":
                        tensor = torch.from_numpy(value.reshape(-1) > 0.5)
                    elif output in {
                        "future_object_confidence",
                        "future_object_age",
                        "target_local_contact",
                    }:
                        tensor = torch.tensor(value.reshape(-1), dtype=torch.float32)
                    elif output == "target_local_force":
                        tensor = torch.tensor(
                            value.reshape(self.horizon, -1)[:, 0],
                            dtype=torch.float32,
                        )
                    else:
                        tensor = torch.tensor(value, dtype=torch.float32)
                    result[output] = tensor

            if "ego_future_action" in requested:
                result["ego_future_action"] = torch.tensor(
                    file[f"transitions/actions/agent_{ego_id}"][future_slice],
                    dtype=torch.float32,
                )
            if "privileged_teammate_future_action" in requested:
                result["privileged_teammate_future_action"] = torch.tensor(
                    file[f"transitions/actions/agent_{teammate_id}"][future_slice],
                    dtype=torch.float32,
                )

            privileged_observation_outputs = {
                "target_object_pose_world": ("object_pose_world", post_observation_slice),
                "target_object_pose_ego": (
                    "object_pose_ego",
                    (post_observation_slice, ego_id),
                ),
                "target_robot_pose_world": ("robot_pose_world", post_observation_slice),
                "target_teammate_pose_ego": (
                    "teammate_pose_ego",
                    (post_observation_slice, ego_id),
                ),
            }
            privileged_obs = None
            for output, (field_name, selection) in privileged_observation_outputs.items():
                if output not in requested:
                    continue
                if privileged_obs is None:
                    privileged_obs = file["privileged/observations"]
                result[output] = torch.tensor(
                    privileged_obs[field_name][selection], dtype=torch.float32
                )

            direct_transition_outputs = {
                "target_reward": "reward",
                "target_done": "done",
                "target_success": "success",
                "target_failure": "failure",
                "target_failure_reason": "failure_reason",
                "target_phase": "phase",
                "target_force_proxy": "force_proxy",
                "target_contact": "contact",
                "target_grasp": "grasp",
                "target_private_event_type": "private_event_type",
                "target_private_event_informed_agent": "private_event_informed_agent",
                "target_private_event_maneuver": "private_event_maneuver",
                "target_private_event_error": "private_event_error",
            }
            transition_dependencies: set[str] = {
                field_name
                for output, field_name in direct_transition_outputs.items()
                if output in requested
            }
            if "target_return" in requested:
                transition_dependencies.add("reward")
            if "target_collision" in requested:
                transition_dependencies.add("contact")
            if "target_force_violation" in requested:
                transition_dependencies.add("force_proxy")
            if "target_maneuver" in requested:
                transition_dependencies.add("private_event_maneuver")
            if "target_progress" in requested:
                transition_dependencies.add("progress")

            legacy_branch_keys = requested & {
                "branch_valid",
                "branch_plan_pair",
                "branch_action",
                "branch_return",
                "branch_success",
                "branch_constraint_violation",
            }
            private_defaults = {
                "private_event_type": -1,
                "private_event_informed_agent": -1,
                "private_event_maneuver": 0,
                "private_event_error": 0.0,
            }
            transition_tensors: dict[str, torch.Tensor] = {}
            transition_values: dict[str, np.ndarray] = {}
            privileged_tr = (
                file["privileged/transitions"]
                if transition_dependencies or legacy_branch_keys
                else None
            )
            for field_name in transition_dependencies:
                assert privileged_tr is not None
                if field_name in privileged_tr:
                    value = privileged_tr[field_name][future_slice]
                elif field_name in private_defaults:
                    value = np.full(
                        (self.horizon, 1), private_defaults[field_name]
                    )
                else:
                    raise KeyError(f"missing privileged transition target {field_name}")
                transition_values[field_name] = np.asarray(value)
                dtype = (
                    torch.long
                    if field_name
                    in {
                        "failure_reason",
                        "phase",
                        "private_event_type",
                        "private_event_informed_agent",
                        "private_event_maneuver",
                    }
                    else torch.float32
                )
                transition_tensors[field_name] = torch.tensor(value, dtype=dtype)
            for output, field_name in direct_transition_outputs.items():
                if output in requested:
                    result[output] = transition_tensors[field_name]
            if "target_progress" in requested:
                result["target_progress"] = transition_tensors["progress"].reshape(-1)
            if "target_return" in requested:
                result["target_return"] = torch.tensor(
                    float(np.sum(transition_values["reward"].reshape(-1))),
                    dtype=torch.float32,
                )
            if "target_collision" in requested:
                result["target_collision"] = transition_tensors["contact"].clone()
            if "target_force_violation" in requested:
                result["target_force_violation"] = (
                    transition_tensors["force_proxy"].reshape(-1) > 1.0
                ).to(torch.float32)
            if "target_maneuver" in requested:
                result["target_maneuver"] = (
                    transition_tensors["private_event_maneuver"].reshape(-1)[0] + 1
                ).clamp(0, 2).to(torch.long)

            if legacy_branch_keys:
                assert privileged_tr is not None
                self._read_selected_legacy_branches(
                    result,
                    privileged_tr,
                    t=t,
                    keys=legacy_branch_keys,
                )

        missing = requested - set(result)
        if missing:
            raise RuntimeError(f"selected dataset implementation missed keys: {sorted(missing)}")
        return result

    def _read_selected_legacy_branches(
        self,
        result: Dict[str, torch.Tensor],
        privileged_tr: h5py.Group,
        *,
        t: int,
        keys: frozenset[str],
    ) -> None:
        has_branches = "branch_action" in privileged_tr
        branch_count = (
            int(privileged_tr["branch_action"].shape[1]) if has_branches else 6
        )
        if "branch_valid" in keys:
            value = (
                privileged_tr["branch_valid"][t]
                if has_branches
                else np.zeros(branch_count, dtype=np.bool_)
            )
            result["branch_valid"] = torch.tensor(value, dtype=torch.bool)
        if "branch_plan_pair" in keys:
            value = (
                privileged_tr["branch_plan_pair"][t]
                if has_branches
                else np.zeros((branch_count, 2), dtype=np.float32)
            )
            result["branch_plan_pair"] = torch.tensor(value, dtype=torch.float32)
        if "branch_action" in keys:
            fitted_action = np.zeros(
                (branch_count, 2, self.horizon, self.action_dim), dtype=np.float32
            )
            if has_branches:
                branch_action = np.asarray(
                    privileged_tr["branch_action"][t], dtype=np.float32
                )
                copied_horizon = min(self.horizon, int(branch_action.shape[2]))
                fitted_action[:, :, :copied_horizon] = branch_action[
                    :, :, :copied_horizon
                ]
            result["branch_action"] = torch.tensor(
                fitted_action, dtype=torch.float32
            )
        scalar_sources = {
            "branch_return": "branch_return",
            "branch_success": "branch_success",
            "branch_constraint_violation": "branch_constraint_violation",
        }
        for output, field_name in scalar_sources.items():
            if output not in keys:
                continue
            value = (
                privileged_tr[field_name][t]
                if has_branches
                else np.zeros(branch_count, dtype=np.float32)
            )
            result[output] = torch.tensor(value, dtype=torch.float32)


# A shorter alias for downstream code that prefers the conceptual name.
DecentralizedEgoDataset = DecentralizedTransitionDataset


def _read_selected_local_observations(
    file: h5py.File,
    agent_id: int,
    field_names: Iterable[str],
    selection,
) -> dict[str, np.ndarray]:
    """Read an explicit subset of one ego observation group."""

    root = file[f"observations/agent_{agent_id}/deployable"]
    return {
        name: np.asarray(root[name][selection], dtype=np.float32)
        for name in field_names
    }


def _left_pad(value: np.ndarray, length: int, pad_count: int) -> np.ndarray:
    array = np.asarray(value)
    padded = np.zeros((length, *array.shape[1:]), dtype=array.dtype)
    padded[pad_count:] = array
    return padded


def _object_tensor(name: str, value: np.ndarray) -> torch.Tensor:
    array = np.asarray(value)
    if name == "object_valid":
        return torch.tensor(bool(array.reshape(-1)[0] > 0.5), dtype=torch.bool)
    if name in {"object_confidence", "object_age"}:
        return torch.tensor(float(array.reshape(-1)[0]), dtype=torch.float32)
    return torch.tensor(array, dtype=torch.float32)


def _flatten_selected_fields(mapping, field_names: list[str]) -> np.ndarray:
    arrays = [np.asarray(mapping[name], dtype=np.float32) for name in field_names]
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    dataset = DecentralizedTransitionDataset(
        args.data_dir, history=args.history, horizon=args.horizon
    )
    print("episodes:", len(dataset.paths))
    print("samples:", len(dataset))
    print("observation_dim:", dataset.observation_dim)
    print("local_history_dim:", dataset.local_history_dim)
    for key, value in dataset[0].items():
        print(key, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
