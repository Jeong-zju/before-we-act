"""Ego-indexed training windows for the FE-PC-WAM schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

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

    def __init__(
        self,
        data_dir: str | Path,
        *,
        history: int = 8,
        horizon: int = 16,
        stride: int = 1,
        ego_ids: Iterable[int] = (0, 1),
        max_episodes: int = -1,
    ):
        self.data_dir = Path(data_dir)
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.ego_ids = tuple(int(value) for value in ego_ids)
        if self.history <= 0 or self.horizon <= 0 or self.stride <= 0:
            raise ValueError("history, horizon, and stride must be positive")
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
                if str(file.attrs.get("schema_version", "")) != SCHEMA_VERSION:
                    raise ValueError(f"{path} is not a {SCHEMA_VERSION} episode")
                file_spec = spec_from_hdf5(file)
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

        with h5py.File(path, "r") as file:
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
                dtype = torch.long if key in {"failure_reason", "phase"} else torch.float32
                result[f"target_{key}"] = torch.tensor(value, dtype=dtype)

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


# A shorter alias for downstream code that prefers the conceptual name.
DecentralizedEgoDataset = DecentralizedTransitionDataset


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
