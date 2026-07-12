"""Training pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data.decentralized_dataset import DecentralizedTransitionDataset
from data.local_observation import LocalObservationPacket, LocalObservationSpec, PoseEstimate
from data.schema import (
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    Episode,
    save_episode,
)
from models.plan_tokenizer import PlanCodeSupport
from policies.decentralized import DecentralizedPolicyConfig
from policies.runtime import (
    DecentralizedRuntime,
    RuntimeConfig,
    _require_compatible_dataset_contract,
)
from train.train_decentralized import (
    TrainingConfig,
    smoke_config,
    train_stage,
)
from train.checkpoint import (
    IncompatibleCheckpoint,
    CONTRACT_TAG,
    file_sha256,
    load_checkpoint,
)


def test_incompatible_checkpoint_is_rejected_explicitly(tmp_path):
    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}, "stage": "wam"}, legacy_path)
    with pytest.raises(IncompatibleCheckpoint, match="Incompatible checkpoints must be retrained"):
        load_checkpoint(legacy_path)


def test_runtime_rejects_checkpoint_local_sensor_contract_mismatch():
    common = {
        "schema_version": SCHEMA_VERSION,
        "history": 2,
        "horizon": 2,
        "model_observation_dim": 9,
        "local_history_dim": 13,
        "action_dim": 4,
        "local_observation_spec": {
            "joint_dim": 0,
            "force_dim": 1,
            "base_twist_dim": 3,
        },
        "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
        "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
        "local_force_units": LOCAL_FORCE_UNITS,
        "local_force_scale_newtons": 1000.0,
        "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
    }
    mismatched = dict(common, local_force_scale_newtons=500.0)

    with pytest.raises(ValueError, match="local_force_scale_newtons"):
        _require_compatible_dataset_contract(
            ({"dataset": common}, {"dataset": mismatched})
        )


def test_all_training_stages_smoke_with_empirical_support_and_lineage(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_episode(data_dir / "episode_000000.hdf5", transitions=5)

    outputs = {
        name: tmp_path / f"{name}.pt"
        for name in ("plan", "belief", "wam", "intention", "wam_robust")
    }

    def config(stage: str) -> TrainingConfig:
        return smoke_config(
            TrainingConfig(
                stage=stage,
                data_dir=str(data_dir),
                output=str(outputs[stage]),
                plan_checkpoint=str(outputs["plan"]) if stage != "plan" else None,
                belief_checkpoint=(
                    str(outputs["belief"])
                    if stage in {"wam", "intention", "wam_robust"}
                    else None
                ),
                wam_checkpoint=(
                    str(outputs["wam"])
                    if stage in {"intention", "wam_robust"}
                    else None
                ),
                intention_checkpoint=(
                    str(outputs["intention"]) if stage == "wam_robust" else None
                ),
                history=2,
                horizon=2,
                batch_size=4,
                min_active_codes=1,
                min_usage_ratio=0.01,
                seed=11,
                device="cpu",
            )
        )

    for stage in ("plan", "belief", "wam", "intention", "wam_robust"):
        assert train_stage(config(stage)) == outputs[stage]

    dataset = DecentralizedTransitionDataset(data_dir, history=2, horizon=2)
    plan = load_checkpoint(outputs["plan"], expected_stage="plan")
    support = PlanCodeSupport.from_dict(plan["plan_code_support"])
    assert plan["contract_tag"] == CONTRACT_TAG
    assert plan["extra"]["encoder_input"] == "ego_future_action_only"
    assert int(support.counts.sum()) == len(dataset)
    assert torch.equal(
        support.active_codes,
        torch.nonzero(support.counts >= support.min_count).flatten(),
    )
    assert torch.all(support.residual_std > 0)
    assert plan["metrics"]["dead_codes"] == support.codebook_size - plan["metrics"]["used_codes"]
    assert plan["metrics"]["usage_ratio"] == pytest.approx(
        plan["metrics"]["used_codes"] / support.codebook_size
    )
    assert plan["metrics"]["hard_usage_ratio"] == plan["metrics"]["usage_ratio"]
    assert plan["metrics"]["hard_entropy"] == plan["metrics"]["entropy"]
    assert plan["metrics"]["hard_perplexity"] == plan["metrics"]["perplexity"]
    assert plan["metrics"]["perplexity"] == pytest.approx(
        np.exp(plan["metrics"]["entropy"])
    )

    belief = load_checkpoint(outputs["belief"], expected_stage="belief")
    assert belief["extra"]["slot_role_order"] == [
        "self",
        "object-belief",
        "teammate-belief",
        "task-context",
    ]
    assert belief["extra"]["privileged_values_are_forward_inputs"] is False
    assert belief["upstream"]["plan"]["sha256"] == file_sha256(outputs["plan"])

    wam = load_checkpoint(outputs["wam"], expected_stage="wam")
    assert wam["extra"]["teammate_private_state_input"] is False
    assert wam["upstream"]["belief"]["sha256"] == file_sha256(outputs["belief"])

    intention = load_checkpoint(outputs["intention"], expected_stage="intention")
    assert intention["extra"]["teammate_plan_is_target_only"] is True
    assert intention["upstream"]["wam"]["sha256"] == file_sha256(outputs["wam"])

    robust = load_checkpoint(outputs["wam_robust"], expected_stage="wam_robust")
    assert robust["extra"]["true_teammate_plan_used_as_input_for_non_oracle_rows"] is False
    assert robust["upstream"]["intention"]["sha256"] == file_sha256(
        outputs["intention"]
    )
    conditioning_rows = sum(
        robust["metrics"][f"conditioning_{name}"]
        for name in ("oracle", "inferred", "missing_prior", "corrupted")
    )
    assert conditioning_rows == 4

    runtime = DecentralizedRuntime.from_checkpoints(
        plan_checkpoint=outputs["plan"],
        belief_checkpoint=outputs["belief"],
        wam_checkpoint=outputs["wam_robust"],
        intention_checkpoint=outputs["intention"],
        config=RuntimeConfig(
            device="cpu",
            policy=DecentralizedPolicyConfig(
                num_candidates=2,
                num_teammate_hypotheses=2,
                residual_sigma_points=1,
                communication_mode="no_comm",
            ),
        ),
    )
    packet = _runtime_packet()
    pair_decision = runtime.step((packet, packet))
    assert pair_decision.joint_action.shape == (8,)
    assert torch.isfinite(pair_decision.joint_action).all()
    assert pair_decision.routed_messages == 0


def _write_synthetic_episode(path: Path, transitions: int) -> None:
    spec = LocalObservationSpec(joint_dim=0, force_dim=1)
    observations = transitions + 1
    time = np.arange(observations, dtype=np.float32)
    local_observations: dict[int, dict[str, np.ndarray]] = {}
    actions: dict[int, np.ndarray] = {}
    for agent_id in (0, 1):
        fields = {
            name: np.zeros((observations, *shape), dtype=np.float32)
            for name, shape in spec.field_shapes().items()
        }
        fields["self/base_twist"][:, 0] = 0.1 * time + 0.05 * agent_id
        fields["self/base_twist"][:, 1] = 0.05 * np.sin(time + agent_id)
        fields["self/base_twist"][:, 2] = 0.02 * np.cos(time)
        fields["local/force"][:, 0] = 0.1 + 0.02 * time
        fields["local/contact"][:, 0] = (time >= 1).astype(np.float32)
        fields["local/grasp"][:, 0] = (time >= 2).astype(np.float32)
        fields["task/goal"][:, 0] = -0.2 * agent_id
        fields["task/goal"][:, 1] = 3.0 - 0.1 * time
        fields["estimates/object/pose"][:, 0] = 0.2 * agent_id
        fields["estimates/object/pose"][:, 1] = 0.5 + 0.05 * time
        fields["estimates/object/pose"][:, 2] = 0.01 * time
        fields["estimates/object/valid"][:, 0] = 1.0
        fields["estimates/object/confidence"][:, 0] = 0.9
        fields["estimates/object/age"][:, 0] = 0.0
        local_observations[agent_id] = fields

        transition_time = np.arange(transitions, dtype=np.float32)
        actions[agent_id] = np.stack(
            [
                0.2 + 0.05 * transition_time + 0.03 * agent_id,
                0.1 * np.sin(transition_time + agent_id),
                0.05 * np.cos(transition_time + agent_id),
                np.clip(0.2 * transition_time, 0.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32)

    robot_pose = np.zeros((observations, 2, 3), dtype=np.float32)
    robot_pose[:, 0, 1] = 0.1 * time
    robot_pose[:, 1, 0] = 0.5
    robot_pose[:, 1, 1] = 0.1 * time
    object_pose = np.stack(
        [np.full_like(time, 0.25), 0.1 * time, np.zeros_like(time)], axis=-1
    )
    teammate_pose = np.zeros((observations, 2, 3), dtype=np.float32)
    teammate_pose[:, 0, 0] = 0.5
    teammate_pose[:, 1, 0] = -0.5
    transition_time = np.arange(transitions, dtype=np.float32)
    episode = Episode(
        local_observations=local_observations,
        actions=actions,
        privileged_observations={
            "time": time[:, None],
            "robot_pose_world": robot_pose,
            "object_pose_world": object_pose,
            "object_pose_ego": np.zeros((observations, 2, 3), dtype=np.float32),
            "teammate_pose_ego": teammate_pose,
            "base_twist_ego": np.zeros((observations, 2, 3), dtype=np.float32),
            "global_state": np.zeros((observations, 12), dtype=np.float32),
        },
        privileged_transitions={
            "reward": np.ones((transitions, 1), dtype=np.float32),
            "done": (transition_time == transitions - 1).astype(np.float32)[:, None],
            "success": (transition_time == transitions - 1).astype(np.float32)[:, None],
            "failure": np.zeros((transitions, 1), dtype=np.float32),
            "failure_reason": np.zeros((transitions, 1), dtype=np.int32),
            "phase": np.zeros((transitions, 1), dtype=np.int32),
            "progress": (transition_time / transitions)[:, None],
            "force_proxy": (0.1 + 0.01 * transition_time)[:, None],
            "contact": np.ones((transitions, 1), dtype=np.float32),
            "grasp": (transition_time >= 1).astype(np.float32)[:, None],
        },
        metadata={
            "source": "unit_test",
            "split": path.parent.name,
            "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": LOCAL_FORCE_UNITS,
            "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
            "local_force_scale_newtons": 1000.0,
        },
    )
    save_episode(path, episode, spec, compression=None)


def _runtime_packet() -> LocalObservationPacket:
    return LocalObservationPacket(
        base_twist=np.zeros(3, dtype=np.float32),
        joint_position=np.zeros(0, dtype=np.float32),
        joint_velocity=np.zeros(0, dtype=np.float32),
        joint_torque=np.zeros(0, dtype=np.float32),
        local_force=np.zeros(1, dtype=np.float32),
        contact=np.zeros(1, dtype=np.float32),
        grasp=np.zeros(1, dtype=np.float32),
        object_estimate=PoseEstimate(
            pose=np.zeros(3, dtype=np.float32),
            valid=np.zeros(1, dtype=np.float32),
            confidence=np.zeros(1, dtype=np.float32),
            age=np.ones(1, dtype=np.float32),
        ),
        task_goal=np.asarray([0.0, 3.0, 0.0], dtype=np.float32),
    )
