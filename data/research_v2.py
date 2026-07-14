"""Versioned matched-intervention data contract for FE-PC-WAM Research-v2.

The deployable trajectory layout deliberately remains compatible with the
strict V1 local-observation representation.  Research-v2 adds paired branch
groups containing forced actions and their matching future *ego-local*
observations.  Group identifiers are target-only metadata and are never
returned through :attr:`ResearchV2Dataset.INPUT_KEYS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch

from data.decentralized_dataset import DecentralizedSampleSpec, DecentralizedTransitionDataset
from data.local_observation import LocalObservationSpec, flatten_observation_mapping
from data.schema import Episode, save_episode


RESEARCH_V2_SCHEMA_VERSION = "fe_pc_wam/research_v2"
RESEARCH_V2_DATA_CONTRACT = "matched_action_interventions/ego_local_future_v1"


@dataclass(frozen=True)
class MatchedBranchGroup:
    """Counterfactual branches evaluated from one simulator snapshot.

    Actions use canonical global agent order ``[agent_0, agent_1]`` on disk.
    Local observations have shape ``[N, H, *field_shape]`` per agent.  The
    dataset is solely responsible for presenting actions in ego-first order.
    """

    group_id: int
    decision_t: int
    plan_pairs: np.ndarray
    actions: np.ndarray
    valid_mask: np.ndarray
    future_local_observations: Mapping[int, Mapping[str, np.ndarray]]
    reward: np.ndarray
    progress: np.ndarray
    contact: np.ndarray
    force: np.ndarray
    success: np.ndarray
    constraint: np.ndarray
    terminal: np.ndarray


def save_research_v2_episode(
    path: str | Path,
    episode: Episode,
    spec: LocalObservationSpec,
    branch_groups: list[MatchedBranchGroup],
) -> None:
    """Write one atomic-compatible V2 episode and validate matched branches."""

    _validate_branch_groups(branch_groups, spec)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".v2tmp")
    save_episode(temporary, episode, spec)
    with h5py.File(temporary, "r+") as file:
        file.attrs["schema_version"] = RESEARCH_V2_SCHEMA_VERSION
        file.attrs["research_v2_data_contract"] = RESEARCH_V2_DATA_CONTRACT
        file.attrs["branch_group_count"] = len(branch_groups)
        root = file.create_group("research_v2")
        root.attrs["policy_input_allowed"] = False
        root.attrs["group_id_is_model_feature"] = False
        branches = root.create_group("branches")
        for index, group in enumerate(branch_groups):
            target = branches.create_group(f"group_{index:04d}")
            target.attrs["group_id"] = int(group.group_id)
            target.attrs["decision_t"] = int(group.decision_t)
            target.create_dataset("plan_pairs", data=np.asarray(group.plan_pairs, dtype=np.float32))
            target.create_dataset("actions", data=np.asarray(group.actions, dtype=np.float32))
            target.create_dataset("valid_mask", data=np.asarray(group.valid_mask, dtype=np.bool_))
            local = target.create_group("future_local")
            for agent_id in (0, 1):
                agent = local.create_group(f"agent_{agent_id}")
                for field_name in spec.field_shapes():
                    _write_nested(agent, field_name, group.future_local_observations[agent_id][field_name])
            outcomes = target.create_group("outcomes")
            for name in ("reward", "progress", "contact", "force", "success", "constraint", "terminal"):
                outcomes.create_dataset(name, data=np.asarray(getattr(group, name)))
    temporary.replace(destination)


class ResearchV2Dataset(DecentralizedTransitionDataset):
    """Ego windows plus matched branch supervision under the V2 contract."""

    INPUT_KEYS = DecentralizedTransitionDataset.INPUT_KEYS
    CURRENT_TARGET_KEYS = frozenset(
        {
            "target_current_self_state",
            "target_current_object_pose",
            "target_current_teammate_pose",
            "target_current_task_progress",
            "target_current_maneuver",
        }
    )
    MATCHED_BRANCH_KEYS = frozenset(
        {
            "branch_group_id",
            "branch_matched_action",
            "branch_valid_mask",
            "branch_future_observation",
            "branch_future_model_observation",
            "branch_future_object_observation",
            "branch_future_object_valid",
            "branch_future_object_confidence",
            "branch_future_object_age",
            "branch_target_reward",
            "branch_target_progress",
            "branch_target_contact",
            "branch_target_force",
            "branch_target_success",
            "branch_target_constraint",
            "branch_target_terminal",
        }
    )
    SAMPLE_KEYS = (
        DecentralizedTransitionDataset.SAMPLE_KEYS
        | CURRENT_TARGET_KEYS
        | MATCHED_BRANCH_KEYS
    )

    def __init__(self, *args, **kwargs):
        kwargs["expected_schema_version"] = RESEARCH_V2_SCHEMA_VERSION
        kwargs.setdefault("hdf5_cache_size", 8)
        super().__init__(*args, **kwargs)
        self._groups_by_file_and_t: dict[tuple[int, int], str] = {}
        self.branch_candidates = 0
        retained_branch_samples: dict[int, list[DecentralizedSampleSpec]] = {}
        for file_idx, path in enumerate(self.paths):
            with h5py.File(path, "r") as file:
                if str(file.attrs.get("research_v2_data_contract", "")) != RESEARCH_V2_DATA_CONTRACT:
                    raise ValueError(f"{path} lacks the matched Research-v2 data contract")
                branch_root = file.get("research_v2/branches")
                if branch_root is None:
                    continue
                for group_name, group in branch_root.items():
                    decision_t = int(group.attrs["decision_t"])
                    key = (file_idx, decision_t)
                    if key in self._groups_by_file_and_t:
                        raise ValueError(f"duplicate branch group for decision_t={decision_t} in {path}")
                    self._groups_by_file_and_t[key] = group_name
                    candidates = int(group["actions"].shape[0])
                    if self.branch_candidates == 0:
                        self.branch_candidates = candidates
                    elif candidates != self.branch_candidates:
                        raise ValueError("branch candidate count changed across Research-v2 data")
                    max_t = int(file.attrs["num_transitions"]) - self.horizon
                    if decision_t <= max_t and decision_t % self.stride:
                        retained_branch_samples.setdefault(file_idx, []).extend(
                            DecentralizedSampleSpec(file_idx, decision_t, ego_id)
                            for ego_id in self.ego_ids
                        )
        # A larger training stride is an important throughput control on D1/D2,
        # but matched interventions are too valuable to drop merely because a
        # decision time is off-stride.  Always retain both ego views for them.
        if retained_branch_samples:
            merged: list[DecentralizedSampleSpec] = []
            cursor = 0
            for file_idx in range(len(self.paths)):
                start = cursor
                while cursor < len(self.index) and self.index[cursor].file_idx == file_idx:
                    cursor += 1
                file_samples = self.index[start:cursor]
                file_samples.extend(retained_branch_samples.get(file_idx, ()))
                file_samples.sort(key=lambda sample: (sample.decision_t, sample.ego_id))
                merged.extend(file_samples)
            self.index = merged

    def get_selected(
        self,
        index: int,
        keys,
    ) -> dict[str, torch.Tensor]:
        """Materialize a key projection without touching unused branch groups."""

        requested = frozenset(str(key) for key in keys)
        if not requested:
            raise ValueError("selected dataset keys cannot be empty")
        unknown = requested - self.SAMPLE_KEYS
        if unknown:
            raise KeyError(f"unknown Research-v2 dataset keys: {sorted(unknown)}")

        base_keys = requested & DecentralizedTransitionDataset.SAMPLE_KEYS
        result = (
            super().get_selected(index, tuple(base_keys)) if base_keys else {}
        )
        sample = self.index[index]
        current_keys = requested & self.CURRENT_TARGET_KEYS
        if current_keys:
            self._read_selected_current_targets(result, sample, current_keys)
        branch_keys = requested & self.MATCHED_BRANCH_KEYS
        if branch_keys:
            self._read_selected_matched_branches(result, sample, branch_keys)
        missing = requested - set(result)
        if missing:
            raise RuntimeError(
                f"Research-v2 selected reader missed keys: {sorted(missing)}"
            )
        return result

    def _read_selected_current_targets(
        self,
        result: dict[str, torch.Tensor],
        sample: DecentralizedSampleSpec,
        keys: frozenset[str],
    ) -> None:
        with self._open_hdf5(self.paths[sample.file_idx]) as file:
            observations = file["privileged/observations"]
            t = sample.decision_t
            ego_id = sample.ego_id
            direct = {
                "target_current_self_state": "base_twist_ego",
                "target_current_object_pose": "object_pose_ego",
                "target_current_teammate_pose": "teammate_pose_ego",
            }
            for output, field_name in direct.items():
                if output in keys:
                    result[output] = torch.tensor(
                        observations[field_name][t, ego_id], dtype=torch.float32
                    )
            if "target_current_task_progress" in keys:
                if "progress" in observations:
                    value = observations["progress"][t]
                else:
                    value = file["privileged/transitions/progress"][max(0, t - 1)]
                result["target_current_task_progress"] = torch.tensor(
                    np.asarray(value).reshape(-1)[:1], dtype=torch.float32
                )
            if "target_current_maneuver" in keys:
                if "private_event_truth" in observations:
                    maneuver = observations["private_event_truth"][t, 3] + 1
                else:
                    maneuver = (
                        file["privileged/transitions/private_event_maneuver"][t]
                        .reshape(-1)[0]
                        + 1
                    )
                result["target_current_maneuver"] = torch.tensor(
                    int(np.clip(maneuver, 0, 2)), dtype=torch.long
                )

    def _read_selected_matched_branches(
        self,
        result: dict[str, torch.Tensor],
        sample: DecentralizedSampleSpec,
        keys: frozenset[str],
    ) -> None:
        candidates = self.branch_candidates or 1
        group_name = self._groups_by_file_and_t.get(
            (sample.file_idx, sample.decision_t)
        )
        if group_name is None:
            result.update(self._empty_matched_branches(candidates, keys=keys))
            return

        ego_id = sample.ego_id
        peer_id = 1 - ego_id
        with self._open_hdf5(self.paths[sample.file_idx]) as file:
            group = file[f"research_v2/branches/{group_name}"]
            if "branch_group_id" in keys:
                result["branch_group_id"] = torch.tensor(
                    (int(sample.file_idx) << 33)
                    | (int(group.attrs["group_id"]) << 1)
                    | int(sample.ego_id),
                    dtype=torch.long,
                )
            if "branch_matched_action" in keys:
                actions = np.asarray(group["actions"], dtype=np.float32)
                if actions.shape[1:] != (2, self.horizon, self.action_dim):
                    raise ValueError(
                        "Research-v2 branch action shape does not match dataset horizon"
                    )
                result["branch_matched_action"] = torch.from_numpy(
                    actions[:, [ego_id, peer_id]]
                )
            if "branch_valid_mask" in keys:
                result["branch_valid_mask"] = torch.from_numpy(
                    np.asarray(group["valid_mask"], dtype=np.bool_)
                )

            local_outputs = {
                "branch_future_object_observation": "estimates/object/pose",
                "branch_future_object_valid": "estimates/object/valid",
                "branch_future_object_confidence": "estimates/object/confidence",
                "branch_future_object_age": "estimates/object/age",
            }
            local_fields: set[str] = set()
            if "branch_future_observation" in keys:
                local_fields.update(self.spec.field_shapes())
            if "branch_future_model_observation" in keys:
                local_fields.update(self.spec.model_field_names())
            for output, field_name in local_outputs.items():
                if output in keys:
                    local_fields.add(field_name)
            if local_fields:
                local = group[f"future_local/agent_{ego_id}"]
                mapping = {
                    name: np.asarray(local[name], dtype=np.float32)
                    for name in local_fields
                }
                if "branch_future_observation" in keys:
                    result["branch_future_observation"] = torch.from_numpy(
                        flatten_observation_mapping(mapping, self.spec)
                    )
                if "branch_future_model_observation" in keys:
                    result["branch_future_model_observation"] = torch.from_numpy(
                        np.concatenate(
                            [
                                mapping[name]
                                for name in self.spec.model_field_names()
                            ],
                            axis=-1,
                        ).astype(np.float32)
                    )
                for output, field_name in local_outputs.items():
                    if output not in keys:
                        continue
                    value = mapping[field_name]
                    if output == "branch_future_object_valid":
                        tensor = torch.from_numpy(
                            value.reshape(candidates, self.horizon) > 0.5
                        )
                    elif output in {
                        "branch_future_object_confidence",
                        "branch_future_object_age",
                    }:
                        tensor = torch.from_numpy(
                            value.reshape(candidates, self.horizon)
                        )
                    else:
                        tensor = torch.from_numpy(value)
                    result[output] = tensor

            outcome_outputs = {
                "branch_target_reward": "reward",
                "branch_target_progress": "progress",
                "branch_target_contact": "contact",
                "branch_target_force": "force",
                "branch_target_success": "success",
                "branch_target_constraint": "constraint",
                "branch_target_terminal": "terminal",
            }
            requested_outcomes = {
                output: field_name
                for output, field_name in outcome_outputs.items()
                if output in keys
            }
            if requested_outcomes:
                outcomes = group["outcomes"]
                for output, field_name in requested_outcomes.items():
                    value = np.asarray(outcomes[field_name], dtype=np.float32)
                    if output in {"branch_target_contact", "branch_target_force"}:
                        value = value[..., ego_id]
                    result[output] = torch.from_numpy(value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(index)
        sample = self.index[index]
        with self._open_hdf5(self.paths[sample.file_idx]) as file:
            observations = file["privileged/observations"]
            transitions = file["privileged/transitions"]
            t = sample.decision_t
            ego_id = sample.ego_id
            result.update(
                {
                    "target_current_self_state": torch.tensor(
                        observations["base_twist_ego"][t, ego_id], dtype=torch.float32
                    ),
                    "target_current_object_pose": torch.tensor(
                        observations["object_pose_ego"][t, ego_id], dtype=torch.float32
                    ),
                    "target_current_teammate_pose": torch.tensor(
                        observations["teammate_pose_ego"][t, ego_id], dtype=torch.float32
                    ),
                    "target_current_task_progress": torch.tensor(
                        (
                            observations["progress"][t]
                            if "progress" in observations
                            else transitions["progress"][max(0, t - 1)]
                        ).reshape(-1)[:1],
                        dtype=torch.float32,
                    ),
                    "target_current_maneuver": torch.tensor(
                        int(
                            np.clip(
                                (
                                    observations["private_event_truth"][t, 3] + 1
                                    if "private_event_truth" in observations
                                    else result["target_maneuver"].item()
                                ),
                                0,
                                2,
                            )
                        ),
                        dtype=torch.long,
                    ),
                }
            )
        group_name = self._groups_by_file_and_t.get((sample.file_idx, sample.decision_t))
        candidates = self.branch_candidates or 1
        if group_name is None:
            result.update(self._empty_matched_branches(candidates))
            return result

        ego_id = sample.ego_id
        peer_id = 1 - ego_id
        with self._open_hdf5(self.paths[sample.file_idx]) as file:
            group = file[f"research_v2/branches/{group_name}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            if actions.shape[1:] != (2, self.horizon, self.action_dim):
                raise ValueError("Research-v2 branch action shape does not match dataset horizon")
            # The only ego canonicalization point.  It is intentionally tested
            # for both agents to prevent the historical agent-1 reversal bug.
            ego_actions = actions[:, [ego_id, peer_id]]
            valid_mask = np.asarray(group["valid_mask"], dtype=np.bool_)
            local = group[f"future_local/agent_{ego_id}"]
            mapping = {
                name: np.asarray(local[name], dtype=np.float32)
                for name in self.spec.field_shapes()
            }
            flat = flatten_observation_mapping(mapping, self.spec)
            model_observation = np.concatenate(
                [mapping[name] for name in self.spec.model_field_names()], axis=-1
            ).astype(np.float32)
            outcomes = group["outcomes"]
            contact = np.asarray(outcomes["contact"], dtype=np.float32)[..., ego_id]
            force = np.asarray(outcomes["force"], dtype=np.float32)[..., ego_id]
            payload = {
                "branch_group_id": torch.tensor(
                    (int(sample.file_idx) << 33)
                    | (int(group.attrs["group_id"]) << 1)
                    | int(sample.ego_id),
                    dtype=torch.long,
                ),
                "branch_matched_action": torch.from_numpy(ego_actions),
                "branch_valid_mask": torch.from_numpy(valid_mask),
                "branch_future_observation": torch.from_numpy(flat),
                "branch_future_model_observation": torch.from_numpy(model_observation),
                "branch_future_object_observation": torch.from_numpy(mapping["estimates/object/pose"]),
                "branch_future_object_valid": torch.from_numpy(
                    mapping["estimates/object/valid"].reshape(candidates, self.horizon) > 0.5
                ),
                "branch_future_object_confidence": torch.from_numpy(
                    mapping["estimates/object/confidence"].reshape(candidates, self.horizon)
                ),
                "branch_future_object_age": torch.from_numpy(
                    mapping["estimates/object/age"].reshape(candidates, self.horizon)
                ),
                "branch_target_reward": torch.from_numpy(np.asarray(outcomes["reward"], dtype=np.float32)),
                "branch_target_progress": torch.from_numpy(np.asarray(outcomes["progress"], dtype=np.float32)),
                "branch_target_contact": torch.from_numpy(contact),
                "branch_target_force": torch.from_numpy(force),
                "branch_target_success": torch.from_numpy(np.asarray(outcomes["success"], dtype=np.float32)),
                "branch_target_constraint": torch.from_numpy(np.asarray(outcomes["constraint"], dtype=np.float32)),
                "branch_target_terminal": torch.from_numpy(np.asarray(outcomes["terminal"], dtype=np.float32)),
            }
        result.update(payload)
        return result

    def _empty_matched_branches(
        self,
        candidates: int,
        *,
        keys: frozenset[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        N, H, A = candidates, self.horizon, int(self.action_dim)
        payload = {
            "branch_group_id": torch.tensor(-1, dtype=torch.long),
            "branch_matched_action": torch.zeros(N, 2, H, A),
            "branch_valid_mask": torch.zeros(N, H, dtype=torch.bool),
            "branch_future_observation": torch.zeros(N, H, self.observation_dim),
            "branch_future_model_observation": torch.zeros(N, H, self.model_observation_dim),
            "branch_future_object_observation": torch.zeros(N, H, 3),
            "branch_future_object_valid": torch.zeros(N, H, dtype=torch.bool),
            "branch_future_object_confidence": torch.zeros(N, H),
            "branch_future_object_age": torch.zeros(N, H),
            "branch_target_reward": torch.zeros(N, H),
            "branch_target_progress": torch.zeros(N, H),
            "branch_target_contact": torch.zeros(N, H),
            "branch_target_force": torch.zeros(N, H),
            "branch_target_success": torch.zeros(N),
            "branch_target_constraint": torch.zeros(N),
            "branch_target_terminal": torch.zeros(N),
        }
        return payload if keys is None else {key: payload[key] for key in keys}


def audit_research_v2_file(path: str | Path) -> dict[str, int | bool]:
    """Read-only structural audit used before training or artifact export."""

    with h5py.File(path, "r") as file:
        if str(file.attrs.get("schema_version", "")) != RESEARCH_V2_SCHEMA_VERSION:
            raise ValueError("not a Research-v2 episode")
        if bool(file["research_v2"].attrs.get("policy_input_allowed", True)):
            raise ValueError("Research-v2 branch root must be target-only")
        if bool(file["research_v2"].attrs.get("group_id_is_model_feature", True)):
            raise ValueError("branch group id must not be a model feature")
        groups = file["research_v2/branches"]
        for group in groups.values():
            actions = np.asarray(group["actions"])
            valid = np.asarray(group["valid_mask"])
            if actions.ndim != 4 or actions.shape[1] != 2 or valid.shape != actions.shape[:1] + actions.shape[2:3]:
                raise ValueError("invalid matched branch shapes")
            if np.asarray(group["plan_pairs"]).shape != (actions.shape[0], 2):
                raise ValueError("invalid branch plan-pair shape")
            if not np.isfinite(actions).all():
                raise ValueError("non-finite matched branch action")
            for agent_id in (0, 1):
                local = group[f"future_local/agent_{agent_id}"]
                for name in spec_from_file_fields(file):
                    if not np.isfinite(np.asarray(local[name])).all():
                        raise ValueError("non-finite branch local observation")
            for value in group["outcomes"].values():
                if not np.isfinite(np.asarray(value)).all():
                    raise ValueError("non-finite branch outcome")
        return {"passed": True, "branch_groups": len(groups)}


def _validate_branch_groups(groups: list[MatchedBranchGroup], spec: LocalObservationSpec) -> None:
    seen_ids: set[int] = set()
    seen_steps: set[int] = set()
    candidate_count: int | None = None
    for group in groups:
        if group.group_id in seen_ids or group.decision_t in seen_steps:
            raise ValueError("branch group ids and decision steps must be unique within an episode")
        seen_ids.add(int(group.group_id))
        seen_steps.add(int(group.decision_t))
        actions = np.asarray(group.actions)
        if actions.ndim != 4 or actions.shape[1] != 2:
            raise ValueError("branch actions must have shape [N,2,H,A]")
        N, _, H, _ = actions.shape
        if np.asarray(group.plan_pairs).shape != (N, 2):
            raise ValueError("branch plan_pairs must have shape [N,2]")
        if candidate_count is None:
            candidate_count = N
        elif candidate_count != N:
            raise ValueError("candidate count changed within episode")
        valid = np.asarray(group.valid_mask)
        if valid.shape != (N, H):
            raise ValueError("branch valid_mask must have shape [N,H]")
        for agent_id in (0, 1):
            mapping = group.future_local_observations.get(agent_id)
            if mapping is None or set(mapping) != set(spec.field_shapes()):
                raise ValueError("each branch requires exact ego-local observation fields")
            for name, shape in spec.field_shapes().items():
                value = np.asarray(mapping[name])
                if value.shape != (N, H, *shape) or not np.isfinite(value).all():
                    raise ValueError(f"invalid branch local observation field {name}")
        for name in ("reward", "progress"):
            value = np.asarray(getattr(group, name))
            if value.shape != (N, H) or not np.isfinite(value).all():
                raise ValueError(f"branch {name} must have shape [N,H]")
        for name in ("contact", "force"):
            value = np.asarray(getattr(group, name))
            if value.shape != (N, H, 2) or not np.isfinite(value).all():
                raise ValueError(f"branch {name} must have shape [N,H,2]")
        for name in ("success", "constraint", "terminal"):
            value = np.asarray(getattr(group, name))
            if value.shape != (N,) or not np.isfinite(value).all():
                raise ValueError(f"branch {name} must have shape [N]")


def _write_nested(root: h5py.Group, name: str, value: np.ndarray) -> None:
    parts = name.split("/")
    group = root
    for part in parts[:-1]:
        group = group.require_group(part)
    group.create_dataset(parts[-1], data=np.asarray(value, dtype=np.float32))


def spec_from_file_fields(file: h5py.File) -> tuple[str, ...]:
    """Read canonical local field names without treating data as model input."""

    import json

    raw = file["schema/local_observation"].attrs["field_order_json"]
    return tuple(json.loads(str(raw)))
