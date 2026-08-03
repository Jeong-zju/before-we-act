from __future__ import annotations

import copy

import pytest

from train.s4_model_registry import (
    S4_MODEL_KINDS,
    S4_R7_BUDGET_MODES,
    S4_R7_MODEL_KINDS,
    S4_R8_MODEL_KINDS,
    validate_s4_candidate,
    validate_s4_r7_candidate,
    validate_s4_r7_pair,
    validate_s4_r8_candidate,
    validate_s4_r8_pair,
)


def _r7(candidate: str) -> dict[str, object]:
    p1 = candidate == "P1"
    return {
        "round": {
            "round_id": "s4-r7",
            "candidate_id": candidate,
            "model_kind": (
                "s4_r7_world_utility_coupling"
                if p1
                else "s4_r7_token_preserving"
            ),
        },
        "model": {
            "evidence_rank": 32,
            "route_mode": "dense",
        },
        "data": {
            "future_feature_cache_mode": "shared_float32_projected_next_view"
        },
        "training": {
            "seed": 707,
            "budget_mode": "fast_selection_30k",
            "updates": 30000,
            "flow_unfreeze_update": 6400,
            "agent_window_budget": 1152000,
            "utility_coupling_weight": 0.05 if p1 else 0.0,
            "relation_weight": 0.0,
            "specialization_weight": 0.0,
            "anchor_weight": 0.0,
        },
    }


def _r8(candidate: str) -> dict[str, object]:
    p1 = candidate == "P1"
    return {
        "round": {
            "round_id": "s4-r8",
            "candidate_id": candidate,
            "model_kind": (
                "s4_r8_causal_prefix_attention"
                if p1
                else "s4_r8_horizon_prefix_mean"
            ),
        },
        "model": {
            "future_horizons": [1, 25, 50, 100],
            "action_prefix_aggregator": (
                "causal_prefix_attention" if p1 else "prefix_mean"
            ),
            "action_prefix_rank": 32,
        },
        "training": {
            "seed": 808,
            "updates": 125000,
            "utility_coupling_weight": 0.05,
            "relation_weight": 0.0,
            "specialization_weight": 0.0,
            "anchor_weight": 0.0,
        },
    }


def test_s4_model_kind_allowlists_are_exact_and_fail_closed() -> None:
    assert S4_R7_MODEL_KINDS == {
        "s4_r7_token_preserving": ("P0", 0.0),
        "s4_r7_world_utility_coupling": ("P1", 0.05),
    }
    assert S4_R7_BUDGET_MODES == {
        "fast_selection_30k": (30_000, 6_400, 1_152_000)
    }
    assert S4_R8_MODEL_KINDS == {
        "s4_r8_horizon_prefix_mean": ("P0", "prefix_mean"),
        "s4_r8_causal_prefix_attention": (
            "P1",
            "causal_prefix_attention",
        ),
    }
    assert len(S4_MODEL_KINDS) == 4
    assert validate_s4_r7_candidate(_r7("P0")) == (
        "P0",
        "s4_r7_token_preserving",
        0.0,
    )
    assert validate_s4_r8_candidate(_r8("P1")) == (
        "P1",
        "s4_r8_causal_prefix_attention",
        "causal_prefix_attention",
    )
    assert validate_s4_candidate(_r8("P0"))[0] == "P0"

    invalid = _r7("P0")
    invalid["round"]["model_kind"] = "s4_r7_unregistered"  # type: ignore[index]
    with pytest.raises(ValueError, match="allowlist"):
        validate_s4_r7_candidate(invalid)


@pytest.mark.parametrize(
    "field", ["relation_weight", "specialization_weight", "anchor_weight"]
)
def test_s4_registry_rejects_disabled_auxiliary_weights(field: str) -> None:
    value = _r7("P1")
    value["training"][field] = 1.0e-6  # type: ignore[index]
    with pytest.raises(ValueError, match="must be exactly zero"):
        validate_s4_r7_candidate(value)


def test_s4_registry_rejects_kind_axis_disagreement() -> None:
    r7 = _r7("P1")
    r7["training"]["utility_coupling_weight"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="identity"):
        validate_s4_r7_candidate(r7)

    r8 = _r8("P1")
    r8["model"]["action_prefix_aggregator"] = "prefix_mean"  # type: ignore[index]
    with pytest.raises(ValueError, match="identity"):
        validate_s4_r8_candidate(r8)

    unknown_auxiliary = _r7("P0")
    unknown_auxiliary["training"]["entropy_balance_weight"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown.*weight"):
        validate_s4_r7_candidate(unknown_auxiliary)


def test_r7_pair_allows_only_registered_utility_axis() -> None:
    p0 = _r7("P0")
    p1 = _r7("P1")
    assert validate_s4_r7_pair(p0, p1) == "s4-r7"

    drifted = copy.deepcopy(p1)
    drifted["training"]["seed"] = 708  # type: ignore[index]
    with pytest.raises(ValueError, match="pre-registered axis"):
        validate_s4_r7_pair(p0, drifted)


def test_r7_registry_rejects_unregistered_or_drifted_budget() -> None:
    value = _r7("P0")
    value["training"]["budget_mode"] = "full_125k"  # type: ignore[index]
    with pytest.raises(ValueError, match="budget mode.*allowlist"):
        validate_s4_r7_candidate(value)

    value = _r7("P0")
    value["training"]["flow_unfreeze_update"] = 26_668  # type: ignore[index]
    with pytest.raises(ValueError, match="budget fields"):
        validate_s4_r7_candidate(value)


def test_r8_pair_allows_only_registered_action_aggregator_axis() -> None:
    p0 = _r8("P0")
    p1 = _r8("P1")
    assert validate_s4_r8_pair(p0, p1) == "s4-r8"

    drifted = copy.deepcopy(p1)
    drifted["model"]["action_prefix_rank"] = 64  # type: ignore[index]
    with pytest.raises(ValueError, match="pre-registered axis"):
        validate_s4_r8_pair(p0, drifted)
