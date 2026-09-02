"""Collectors must be able to pick a candidate family without mixing corpora."""
from __future__ import annotations

import numpy as np
import pytest

from before_we_act.care_behavior_candidates import BehaviorCandidateConfig
from before_we_act.care_candidate_family import (
    BEHAVIOR_FAMILY,
    CANDIDATE_FAMILIES,
    FIXED_FAMILY,
    build_candidate,
    build_candidate_set,
    candidate_count,
    candidate_names,
    family_manifest,
    validate_candidate_for_family,
)


CONFIG = BehaviorCandidateConfig(action_horizon=100, action_dim=8)


class _Space:
    low = np.full(8, -10.0, dtype=np.float32)
    high = np.full(8, 10.0, dtype=np.float32)


def _reference() -> np.ndarray:
    steps = np.arange(100, dtype=np.float32)[:, None]
    plan = np.zeros((100, 8), dtype=np.float32)
    plan[:, :7] = 0.01 * steps
    plan[:, 7] = (steps[:, 0] >= 5).astype(np.float32)
    return plan


def test_both_families_expose_the_same_arity() -> None:
    assert CANDIDATE_FAMILIES == (FIXED_FAMILY, BEHAVIOR_FAMILY)
    assert candidate_count(FIXED_FAMILY) == candidate_count(BEHAVIOR_FAMILY) == 6
    assert candidate_names(BEHAVIOR_FAMILY)[0] == "reference"


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_candidate_zero_is_the_nominal_chunk(family: str) -> None:
    reference = _reference()
    plan = build_candidate(
        family,
        0,
        reference=reference,
        base=reference * 0.5,
        current_qpos=np.zeros(8, dtype=np.float32),
        current_grip=0.0,
        config=CONFIG,
    )
    np.testing.assert_array_equal(plan, reference)


def test_behavior_family_needs_its_config() -> None:
    with pytest.raises(ValueError, match="needs a BehaviorCandidateConfig"):
        build_candidate(
            BEHAVIOR_FAMILY,
            1,
            reference=_reference(),
            base=_reference(),
            current_qpos=np.zeros(8, dtype=np.float32),
            current_grip=0.0,
        )


def test_behavior_set_is_legal_under_its_own_envelope() -> None:
    reference = _reference()
    pose = np.zeros(8, dtype=np.float32)
    family = build_candidate_set(
        BEHAVIOR_FAMILY,
        reference=reference,
        base=reference,
        current_qpos=pose,
        current_grip=0.0,
        config=CONFIG,
    )
    assert family.shape == (6, 100, 8)
    for candidate in range(6):
        valid, failures = validate_candidate_for_family(
            BEHAVIOR_FAMILY,
            candidate,
            family[candidate],
            reference=reference,
            base=reference,
            current_qpos=pose,
            current_grip=0.0,
            action_space=_Space(),
            config=CONFIG,
        )
        assert valid, (candidate, failures)


def test_manifest_records_which_family_produced_a_corpus() -> None:
    fixed = family_manifest(FIXED_FAMILY, None)
    behavior = family_manifest(BEHAVIOR_FAMILY, CONFIG)

    assert fixed["candidate_family"] == FIXED_FAMILY
    assert "candidate_family_config" not in fixed
    assert behavior["candidate_family"] == BEHAVIOR_FAMILY
    assert behavior["candidate_family_config"]["intervention_steps"] == 8
    assert behavior["candidate_names"][1] == "wait"


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown CARE candidate family"):
        candidate_names("learned")
