from __future__ import annotations

from copy import deepcopy

from scripts.before_we_act.label_care_simulator_anomalies import label_horizon


def outcome(utility: float = 0.1) -> dict:
    return {
        "utility_main": utility,
        "hard_safety_violation": False,
        "first_success_step": None,
        "final_stage_id": "active",
        "observed_steps": 8,
        "final_factorized_predicates": {"done": False, "distance": 0.1},
    }


def family() -> dict:
    branches = []
    for repeat_id in (0, 1):
        for candidate_id in range(6):
            for regime in ("reactive", "replay"):
                branches.append(
                    {
                        "candidate_id": candidate_id,
                        "regime": regime,
                        "repeat_id": repeat_id,
                        "status": "VALID",
                        "candidate_valid": True,
                        "restore_observation_source": "captured_snapshot",
                        "restore_observation_max_abs_error": 0.0,
                        "outcomes": {"8": outcome()},
                    }
                )
    return {
        "snapshot_id": "test-family",
        "repeat_common_replay_support": [
            {"repeat_id": 0, "common_replay_support_steps": 8},
            {"repeat_id": 1, "common_replay_support_steps": 8},
        ],
        "candidate_legality": [
            {"candidate_id": candidate_id, "valid": True} for candidate_id in range(6)
        ],
        "restore_probe": {
            "restore_observation_source": "captured_snapshot",
            "restore_observation_max_abs_error": 0.0,
            "terminal_and_success_exact": True,
        },
        "branches": branches,
    }


def branch(row: dict, regime: str, repeat_id: int) -> dict:
    return next(
        value
        for value in row["branches"]
        if value["candidate_id"] == 0
        and value["regime"] == regime
        and value["repeat_id"] == repeat_id
    )


def test_stable_supported_horizon_is_usable() -> None:
    result = label_horizon(family(), 8)
    assert result["label"] == "USE"
    assert result["use_for_gate_analysis"] is True
    assert result["use_for_training"] is False


def test_reference_mode_mismatch_is_simulator_anomaly() -> None:
    row = family()
    branch(row, "replay", 0)["outcomes"]["8"]["hard_safety_violation"] = True
    result = label_horizon(row, 8)
    assert result["label"] == "DO_NOT_USE_SIMULATOR_ANOMALY"
    assert "REFERENCE_MODE_DISCRETE_MISMATCH" in result["reasons"]


def test_reference_repeat_noise_is_simulator_anomaly() -> None:
    row = family()
    for regime in ("reactive", "replay"):
        branch(row, regime, 1)["outcomes"]["8"]["utility_main"] = 0.2
    result = label_horizon(row, 8)
    assert result["label"] == "DO_NOT_USE_SIMULATOR_ANOMALY"
    assert result["reasons"] == ["REFERENCE_REPEAT_UTILITY_MISMATCH"]


def test_missing_common_support_is_not_mislabeled_as_simulator_anomaly() -> None:
    row = deepcopy(family())
    row["repeat_common_replay_support"][1]["common_replay_support_steps"] = 7
    result = label_horizon(row, 8)
    assert result["label"] == "DO_NOT_USE_UNSUPPORTED_HORIZON"
