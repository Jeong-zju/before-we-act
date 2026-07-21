from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eval.m1_legacy_regression import (
    AuditedLegacyDirectPolicy,
    EXPECTED_LEGACY_ACTION_SOURCE,
    EXPECTED_M1_ACTION_SOURCE,
    LEGACY_POLICY,
    LegacyRegressionEpisode,
    LegacyRegressionObserver,
    M1_POLICY,
    legacy_regression_report,
    rotating_train_seed,
)


TRAIN_SEEDS = (101, 202, 303)
SUITE_SEEDS = {
    "standard": tuple(range(120_000, 120_500)),
    "challenge": tuple(range(220_000, 220_500)),
}
TREE_HASH = "a" * 64


def _checkpoint_evidence() -> dict[str, dict]:
    return {
        str(seed): {
            "checkpoint": f"checkpoints/phase_m1/state_vision_future/seed_{seed}",
            "tree_sha256": chr(98 + index) * 64,
            "train_seed": seed,
            "model_variant": "state_vision_future",
            "strict_reload_passed": True,
            "embedded_legacy_matches_source": True,
        }
        for index, seed in enumerate(TRAIN_SEEDS)
    }


def _episode(
    *, suite: str, seed: int, index: int, m1: bool
) -> LegacyRegressionEpisode:
    steps = 8
    if m1:
        return LegacyRegressionEpisode(
            suite=suite,
            seed=seed,
            policy=M1_POLICY,
            train_seed=rotating_train_seed(index, TRAIN_SEEDS),
            success=True,
            failure=False,
            failure_reason="none",
            steps=steps,
            total_reward=50.0,
            action_sources=(EXPECTED_M1_ACTION_SOURCE,),
            presented_observation_paths=(
                (
                    "image_frame_indices.fixed",
                    "image_timestamps.fixed",
                    "images.fixed",
                    "past_executed_actions",
                    "proprioception",
                    "task.id",
                    "task.text",
                ),
            ),
            consumed_observation_paths=(
                (
                    "image_frame_indices.fixed",
                    "images.fixed",
                    "proprioception",
                    "task.id",
                ),
            ),
            direct_execution_steps=steps,
            fallback_steps=0,
            fallback_reasons=(),
            privileged_observation_seen=False,
            actions_finite_and_bounded=True,
            fixed_rgb_presented_steps=steps,
            fixed_rgb_consumed_steps=steps,
            visual_frame_indices=(0, 0, 1, 1, 2, 2, 3, 3),
            visual_10hz_pattern_valid=True,
        )
    return LegacyRegressionEpisode(
        suite=suite,
        seed=seed,
        policy=LEGACY_POLICY,
        train_seed=None,
        success=True,
        failure=False,
        failure_reason="none",
        steps=steps,
        total_reward=50.0,
        action_sources=(EXPECTED_LEGACY_ACTION_SOURCE,),
        presented_observation_paths=(("proprioception",),),
        consumed_observation_paths=(("proprioception",),),
        direct_execution_steps=steps,
        fallback_steps=0,
        fallback_reasons=(),
        privileged_observation_seen=False,
        actions_finite_and_bounded=True,
        fixed_rgb_presented_steps=0,
        fixed_rgb_consumed_steps=0,
        visual_frame_indices=(),
        visual_10hz_pattern_valid=True,
    )


def _records() -> tuple[list[LegacyRegressionEpisode], list[LegacyRegressionEpisode]]:
    m1: list[LegacyRegressionEpisode] = []
    legacy: list[LegacyRegressionEpisode] = []
    for suite, seeds in SUITE_SEEDS.items():
        for index, seed in enumerate(seeds):
            m1.append(_episode(suite=suite, seed=seed, index=index, m1=True))
            legacy.append(_episode(suite=suite, seed=seed, index=index, m1=False))
    return m1, legacy


def _report(
    m1: list[LegacyRegressionEpisode],
    legacy: list[LegacyRegressionEpisode],
    *,
    formal: bool = True,
) -> dict:
    return legacy_regression_report(
        m1,
        legacy,
        suite_seeds=SUITE_SEEDS,
        train_seeds=TRAIN_SEEDS,
        formal_protocol=formal,
        source_checkpoint_sha256_before=TREE_HASH,
        source_checkpoint_sha256_after=TREE_HASH,
        expected_source_checkpoint_sha256=TREE_HASH,
        checkpoint_evidence=_checkpoint_evidence(),
    )


def test_formal_legacy_regression_is_exactly_paired_and_acceptance_ready() -> None:
    m1, legacy = _records()
    report = _report(m1, legacy)

    assert report["passed"]
    assert all(report["checks"].values())
    assert report["suites"]["standard"]["m1_episodes"] == 500
    assert report["suites"]["challenge"]["legacy_episodes"] == 500
    assert report["suites"]["standard"]["train_seed_counts"] == {
        "101": 167,
        "202": 167,
        "303": 166,
    }
    assert report["suites"]["standard"]["paired_outcomes"] == {
        "m1_1_legacy_1": 500
    }


def test_diagnostic_evidence_is_never_promoted_to_formal_acceptance() -> None:
    m1, legacy = _records()
    report = _report(m1, legacy, formal=False)

    assert not report["passed"]
    assert report["diagnostic_criteria_met"]
    assert not report["checks"]["formal_protocol"]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "rotation", "fallback", "rgb", "rgb_pattern_claim"]
)
def test_protocol_mutations_fail_closed(mutation: str) -> None:
    m1, legacy = _records()
    if mutation == "duplicate":
        m1.append(m1[0])
    elif mutation == "rotation":
        m1[0] = replace(m1[0], train_seed=TRAIN_SEEDS[1])
    elif mutation == "fallback":
        m1[0] = replace(
            m1[0], fallback_steps=1, fallback_reasons=("flow_error",)
        )
    elif mutation == "rgb":
        m1[0] = replace(m1[0], fixed_rgb_consumed_steps=m1[0].steps - 1)
    elif mutation == "rgb_pattern_claim":
        m1[0] = replace(
            m1[0], visual_frame_indices=(), visual_10hz_pattern_valid=True
        )

    report = _report(m1, legacy)
    assert not report["passed"]


def test_regression_above_five_percentage_points_fails() -> None:
    m1, legacy = _records()
    # 31/500 failures is a 6.2 percentage-point regression in standard.
    standard_indices = [
        index for index, item in enumerate(m1) if item.suite == "standard"
    ][:31]
    for index in standard_indices:
        m1[index] = replace(
            m1[index], success=False, failure=True, failure_reason="timeout"
        )
    report = _report(m1, legacy)

    assert not report["passed"]
    assert not report["suites"]["standard"]["regression_passed"]
    assert report["suites"]["standard"]["legacy_minus_m1"] == pytest.approx(
        0.062
    )


class _DirectDelegate:
    def __init__(self) -> None:
        self.last_diagnostics: dict = {}

    def reset(self) -> None:
        self.last_diagnostics = {}

    def act(self, observation: dict) -> np.ndarray:
        assert set(observation) == {"proprioception"}
        self.last_diagnostics = {
            "direct_flow_executed": True,
            "fallback_enabled": False,
            "fallback_reason": "none",
        }
        return np.zeros(8, dtype=np.float32)


def test_audited_legacy_wrapper_records_direct_source_and_rejects_leakage() -> None:
    wrapped = AuditedLegacyDirectPolicy(_DirectDelegate())
    action = wrapped.act({"proprioception": np.zeros(22, dtype=np.float32)})

    assert action.shape == (8,)
    assert wrapped.last_diagnostics["action_source"] == EXPECTED_LEGACY_ACTION_SOURCE
    assert wrapped.last_diagnostics["fallback_used"] is False
    assert wrapped.last_diagnostics["presented_observation_paths"] == (
        "proprioception",
    )
    with pytest.raises(RuntimeError, match="forbidden observation"):
        wrapped.act(
            {
                "proprioception": np.zeros(22, dtype=np.float32),
                "privileged_state": np.zeros(1, dtype=np.float32),
            }
        )


def test_m1_legacy_observer_cross_checks_runner_and_policy_frame_indices() -> None:
    policy = SimpleNamespace(
        last_diagnostics={
            "action_source": EXPECTED_M1_ACTION_SOURCE,
            "direct_flow_executed": True,
            "fallback_used": False,
            "fallback_enabled": False,
            "fallback_reason": "none",
            "presented_observation_paths": (
                "image_frame_indices.fixed",
                "images.fixed",
                "past_executed_actions",
                "proprioception",
                "task.id",
                "task.text",
            ),
            "consumed_observation_paths": (
                "image_frame_indices.fixed",
                "images.fixed",
                "proprioception",
                "task.id",
            ),
            "privileged_state_seen": False,
            "visual_frame_index": 0,
        }
    )
    observer = LegacyRegressionObserver(
        suite="standard",
        policy_name=M1_POLICY,
        policy=policy,
        train_seed=101,
    )
    observer.on_episode_start(seed=120_000)
    transition = SimpleNamespace(
        action=np.zeros(8, dtype=np.float32),
        image_frame_indices={"fixed": 0},
    )

    observer.on_transition(transition)
    observer.on_transition(transition)
    assert observer.visual_frame_indices == [0, 0]

    policy.last_diagnostics["visual_frame_index"] = 1
    observer.on_transition(transition)
    assert observer.visual_frame_indices[-1] == -1


def test_round_robin_assignment_requires_exactly_three_unique_seeds() -> None:
    assert [rotating_train_seed(index, TRAIN_SEEDS) for index in range(7)] == [
        101,
        202,
        303,
        101,
        202,
        303,
        101,
    ]
    with pytest.raises(ValueError, match="three unique"):
        rotating_train_seed(0, (101, 202))
