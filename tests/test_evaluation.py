"""Evaluation tests."""

from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest
import torch

from data.local_observation import LocalObservationSpec
from data.schema import (
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    Episode,
    save_episode,
)
from eval.evaluate import (
    COMMUNICATION_MODES,
    EvaluationContractError,
    compare_communication_modes,
)
from models.plan_tokenizer import build_plan_code_support
from scripts.audit_contract import (
    EXPECTED_DEPLOYABLE_INPUT_KEYS,
    ContractAuditError,
    audit_candidate_codes,
    audit_checkpoint_file,
    audit_hdf5_file,
    audit_plan_code_support,
    run_contract_audit,
)
from train.checkpoint import make_checkpoint, save_checkpoint


def test_paired_mode_evaluation_aggregates_round_trip_diagnostics_and_marks_oracle():
    patterns = {
        "no_comm": [0, 0],
        "always_reply": [1, 1],
        "selective_vpi": [0, 1],
        "periodic": [1, 0],
        "random": [0, 1],
        "oracle_upper_bound": [1, 1],
    }
    records = {
        mode: [_evaluation_record(seed, pattern) for seed in (3, 7)]
        for mode, pattern in patterns.items()
    }

    report = compare_communication_modes(records)
    selective = report["modes"]["selective_vpi"]
    always = report["modes"]["always_reply"]
    no_comm = report["modes"]["no_comm"]
    oracle = report["modes"]["oracle_upper_bound"]

    assert report["mode_order"] == list(COMMUNICATION_MODES)
    assert report["paired_inputs_verified"] is True
    assert report["input_digest_verified"] is True
    assert selective["request_rate"] == pytest.approx(0.5)
    assert selective["reply_rate"] == pytest.approx(0.5)
    assert selective["actual_request_reply_bits_total"] == 64.0
    assert always["actual_request_reply_bits_total"] == 128.0
    assert no_comm["actual_request_reply_bits_total"] == 0.0
    assert selective["G_improvement_mean"] == pytest.approx(1.0)
    assert selective["replan_rate"] == pytest.approx(0.5)
    assert selective["action_change_l2_mean"] == pytest.approx(0.25)
    assert oracle["deployable"] is False
    assert "Privileged" in oracle["non_deployable_reason"]
    assert "NON-DEPLOYABLE" in report["oracle_notice"]


def test_paired_mode_evaluation_rejects_changed_episode_inputs():
    records = {
        mode: [_evaluation_record(5, [0, 0])]
        for mode in COMMUNICATION_MODES
    }
    changed = copy.deepcopy(records)
    changed["random"][0]["input_digest"] = "different-world-state"

    with pytest.raises(EvaluationContractError, match="input digests differ"):
        compare_communication_modes(changed)


def test_contract_audit_accepts_artifacts_and_empirical_candidates(tmp_path):
    episode_path = tmp_path / "episode_000000.hdf5"
    _write_episode(episode_path)
    checkpoint_path, support = _write_plan_checkpoint(tmp_path / "plan.pt")

    report = run_contract_audit(
        [episode_path],
        [checkpoint_path],
        candidate_codes=[1, 3, 1],
    )

    assert report["passed"] is True
    assert report["episodes"][0]["firewall_passed"] is True
    usage = report["checkpoints"][0]["plan_code_support"]
    assert usage["used_codes"] == 2
    assert usage["active_codes"] == [1, 3]
    assert usage["usage_ratio"] == pytest.approx(0.5)
    assert usage["perplexity"] == pytest.approx(np.exp(usage["entropy"]))
    assert report["candidate_codes"]["all_candidates_within_empirical_support"] is True

    direct = audit_plan_code_support(support)
    assert direct["encoded_segments"] == 4
    with pytest.raises(ContractAuditError, match="outside empirical active support"):
        audit_candidate_codes([1, 2], support)


def test_contract_audit_fails_on_firewall_drift_incompatibility_and_missing_support(
    tmp_path,
):
    episode_path = tmp_path / "episode_000000.hdf5"
    _write_episode(episode_path)
    with h5py.File(episode_path, "r+") as file:
        file["observations/agent_0/deployable"].create_dataset(
            "estimates/teammate/pose", data=np.zeros((3, 3), dtype=np.float32)
        )
    with pytest.raises(ContractAuditError, match="deployable fields mismatch"):
        audit_hdf5_file(episode_path)

    legacy = tmp_path / "legacy.pt"
    torch.save({"stage": "plan", "model": {}}, legacy)
    with pytest.raises(ContractAuditError, match="Incompatible checkpoints must be retrained"):
        audit_checkpoint_file(legacy)

    missing_support = make_checkpoint(
        stage="plan",
        model_class="ActionOnlyPlanTokenizer",
        model_config={"horizon": 2, "action_dim": 4},
        model_state_dict={},
        training_config={},
        dataset_metadata={"schema_version": SCHEMA_VERSION},
        metrics={},
        plan_code_support=None,
        extra={
            "encoder_input": "ego_future_action_only",
            "hardcoded_plan_codes_allowed": False,
        },
    )
    missing_path = tmp_path / "missing_support.pt"
    save_checkpoint(missing_path, missing_support)
    with pytest.raises(ContractAuditError, match="no empirical plan_code_support"):
        audit_checkpoint_file(missing_path)


def _evaluation_record(seed: int, request_pattern: list[int]) -> dict:
    return {
        "seed": seed,
        "episode_id": f"episode-{seed}",
        "input_digest": f"fixed-input-{seed}",
        "decision_count": 2,
        "success": True,
        "safe": True,
        "request": request_pattern,
        "reply": request_pattern,
        "request_bits": [8, 8],
        "reply_bits": [24, 24],
        "communication_delay": request_pattern,
        "VPI": [0.25, 0.75],
        "code_surprise": [0.1, 0.3],
        "residual_surprise": [0.2, 0.4],
        "plan_surprise": [0.3, 0.7],
        "G_before": [3.0, 2.0],
        "G_after": [2.0, 1.0],
        "replanned": request_pattern,
        "action_change_l2": [0.0, 0.5],
    }


def _write_episode(path) -> None:
    spec = LocalObservationSpec()
    transitions = 2
    observations = transitions + 1
    local_observations = {}
    actions = {}
    for agent_id in (0, 1):
        fields = {
            name: np.zeros((observations, *shape), dtype=np.float32)
            for name, shape in spec.field_shapes().items()
        }
        fields["estimates/object/valid"][:] = 1.0
        fields["estimates/object/confidence"][:] = 0.8
        fields["task/goal"][:, 1] = 2.0
        local_observations[agent_id] = fields
        actions[agent_id] = np.zeros((transitions, 4), dtype=np.float32)
    episode = Episode(
        local_observations=local_observations,
        actions=actions,
        privileged_observations={
            "object_pose_world": np.zeros((observations, 3), dtype=np.float32)
        },
        privileged_transitions={
            "success": np.zeros((transitions, 1), dtype=np.float32)
        },
        metadata={
            "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": LOCAL_FORCE_UNITS,
            "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
            "local_force_scale_newtons": 1000.0,
        },
    )
    save_episode(path, episode, spec)


def _write_plan_checkpoint(path):
    codes = torch.tensor([1, 1, 1, 3], dtype=torch.long)
    residual = torch.tensor(
        [[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [1.0, 1.0]],
        dtype=torch.float32,
    )
    support = build_plan_code_support(codes, residual, codebook_size=4).to_dict()
    usage = audit_plan_code_support(support)
    checkpoint = make_checkpoint(
        stage="plan",
        model_class="ActionOnlyPlanTokenizer",
        model_config={"horizon": 2, "action_dim": 4},
        model_state_dict={},
        training_config={},
        dataset_metadata={
            "schema_version": SCHEMA_VERSION,
            "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": LOCAL_FORCE_UNITS,
            "local_force_scale_newtons": 1000.0,
            "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
            "deployable_input_keys": sorted(EXPECTED_DEPLOYABLE_INPUT_KEYS),
            "input_feature_names": [
                "base_vx",
                "base_vy",
                "base_wz",
                "local_force_0",
                "local_contact",
                "local_grasp",
                "goal_ego_x",
                "goal_ego_y",
                "goal_ego_yaw",
                "previous_ego_action_0",
            ],
        },
        metrics={
            "used_codes": usage["used_codes"],
            "usage_ratio": usage["usage_ratio"],
            "entropy": usage["entropy"],
            "perplexity": usage["perplexity"],
        },
        plan_code_support=support,
        extra={
            "encoder_input": "ego_future_action_only",
            "hardcoded_plan_codes_allowed": False,
        },
    )
    save_checkpoint(path, checkpoint)
    return path, support
