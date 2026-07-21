"""Fail-closed Phase M1 acceptance over auditable in-memory evidence.

This module intentionally has no simulator, model, filesystem, or SciPy
dependency.  Formal runners serialize their episode/probe/runtime evidence and
call :func:`m1_acceptance_report`; unit tests exercise the exact same gates.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from eval.m1_statistics import (
    EpisodeLike,
    aggregate_episode_records,
    coerce_episode_record,
    paired_balanced_accuracy_comparison,
    paired_episode_success,
    paired_rmse_comparison,
    wilson_interval,
)


FORMAT_VERSION = "wam.multimodal.m1.acceptance/2"
TRAIN_SEED_CONSISTENCY_RULE = "strict_majority_positive_mean_difference"
ALL_TRAIN_SEEDS_POSITIVE_CI_RULE = "all_train_seeds_positive_ci"

STATE_ONLY = "state_only"
VISION_ONLY = "vision_only"
STATE_VISION_NO_FUTURE = "state_vision_no_future"
STATE_VISION_FUTURE = "state_vision_future"
PARAMETER_MATCHED_MLP = "parameter_matched_mlp"
PRIMARY_VARIANT = STATE_VISION_FUTURE
REQUIRED_VARIANTS = (
    STATE_ONLY,
    VISION_ONLY,
    STATE_VISION_NO_FUTURE,
    STATE_VISION_FUTURE,
    PARAMETER_MATCHED_MLP,
)

CLEAN = "clean"
SHUFFLE_RGB = "shuffle_rgb"
FREEZE_FIRST_RGB = "freeze_first_rgb"
OCCLUDE_CUE = "occlude_cue"
SHUFFLE_STATE = "shuffle_state"
VISUAL_INTERVENTIONS = (SHUFFLE_RGB, FREEZE_FIRST_RGB, OCCLUDE_CUE)
PRIMARY_INTERVENTIONS = (*VISUAL_INTERVENTIONS, SHUFFLE_STATE)

REQUIRED_LEGACY_SUITES = ("standard", "challenge")
EXPECTED_ACTION_SOURCE = {
    STATE_ONLY: "m1_state_only",
    VISION_ONLY: "m1_vision_only",
    STATE_VISION_NO_FUTURE: "m1_state_vision_no_future",
    STATE_VISION_FUTURE: "m1_state_vision_future",
    PARAMETER_MATCHED_MLP: "m1_parameter_matched_mlp",
}

MINIMUM_TRAINABLE_PARAMETERS = 20_000_000
MAXIMUM_TRAINABLE_PARAMETERS = 60_000_000
MINIMUM_FORMAL_EVALUATION_SEEDS = 100
MAXIMUM_FORMAL_EVALUATION_SEEDS = 500
REQUIRED_TRAIN_SEEDS = 3
MINIMUM_VISION_GAIN = 0.10
MAXIMUM_LEGACY_REGRESSION = 0.05
MINIMUM_VISUAL_INTERVENTION_DROP = 0.15
MINIMUM_STATE_SHUFFLE_DROP = 0.05
MAXIMUM_DIRECT_P95_MS = 50.0
MAXIMUM_DECIMATED_ACTION_AGE_P95_MS = 100.0

_REQUIRED_MULTIMODAL_INPUTS = {
    "images.fixed",
    "past_executed_actions",
    "proprioception",
    "task.id",
    "task.text",
}
_REQUIRED_INPUTS = {
    STATE_ONLY: {
        "past_executed_actions",
        "proprioception",
        "task.id",
        "task.text",
    },
    VISION_ONLY: {
        "images.fixed",
        "past_executed_actions",
        "task.id",
        "task.text",
    },
    STATE_VISION_NO_FUTURE: _REQUIRED_MULTIMODAL_INPUTS,
    STATE_VISION_FUTURE: _REQUIRED_MULTIMODAL_INPUTS,
    PARAMETER_MATCHED_MLP: _REQUIRED_MULTIMODAL_INPUTS,
}
_FORBIDDEN_PATH_PARTS = {
    "cue_id",
    "cue_value",
    "cue_variant",
    "event_truth",
    "goal_truth",
    "obstacle_truth",
    "privileged_state",
    "rendered_cue_variant",
    "target_truth",
    "task_truth",
}


def m1_acceptance_report(
    records: Sequence[EpisodeLike],
    *,
    tasks: Sequence[str],
    evaluation_seeds: Sequence[int],
    cue_variants: Sequence[int],
    train_seeds: Sequence[int],
    architecture: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    parameter_counts: Mapping[str, Any],
    legacy_suites: Mapping[str, Any],
    future_probe: Mapping[str, Any],
    runtime: Mapping[str, Any],
    evidence: Mapping[str, Any],
    formal_protocol: bool,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Apply every Phase M1 gate and return auditable diagnostics.

    ``passed`` can only be true for a formal run.  A diagnostic run may meet
    all technical criteria, which is reported separately as
    ``diagnostic_criteria_met`` and never promoted to acceptance.
    """

    required_tasks = _unique_nonempty_strings(tasks, "tasks")
    required_eval_seeds = _unique_nonnegative_ints(
        evaluation_seeds, "evaluation_seeds"
    )
    required_cues = _unique_nonnegative_ints(cue_variants, "cue_variants")
    required_train_seeds = _unique_nonnegative_ints(train_seeds, "train_seeds")
    if len(required_cues) < 2:
        raise ValueError("cue_variants must contain at least two values")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")

    normalized = tuple(coerce_episode_record(record) for record in records)
    aggregate = aggregate_episode_records(normalized, confidence=confidence)
    matrix = _formal_matrix_check(
        normalized,
        tasks=required_tasks,
        evaluation_seeds=required_eval_seeds,
        cue_variants=required_cues,
        train_seeds=required_train_seeds,
    )
    observation_contract = _observation_and_action_contract(normalized)

    vision_gain = paired_episode_success(
        normalized,
        first_variant=PRIMARY_VARIANT,
        first_intervention=CLEAN,
        second_variant=STATE_ONLY,
        second_intervention=CLEAN,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    vision_gain["train_seed_consistency"] = _train_seed_consistency(
        vision_gain, rule=TRAIN_SEED_CONSISTENCY_RULE
    )
    vision_gain_passed = _paired_difference_passes(
        vision_gain,
        minimum=MINIMUM_VISION_GAIN,
        train_seed_consistency_rule=TRAIN_SEED_CONSISTENCY_RULE,
    )

    visual_family_size = len(VISUAL_INTERVENTIONS)
    visual_comparison_confidence = 1.0 - (1.0 - confidence) / visual_family_size
    visual_multiplicity_adjustment = {
        "method": "bonferroni_simultaneous_confidence_intervals",
        "selection": "best_of_three_post_selected_visual_interventions",
        "family_size": visual_family_size,
        "family_wise_confidence": confidence,
        "per_comparison_confidence": visual_comparison_confidence,
    }
    visual_perturbations: dict[str, Any] = {}
    for offset, intervention in enumerate(VISUAL_INTERVENTIONS, start=1):
        comparison = paired_episode_success(
            normalized,
            first_variant=PRIMARY_VARIANT,
            first_intervention=CLEAN,
            second_variant=PRIMARY_VARIANT,
            second_intervention=intervention,
            confidence=visual_comparison_confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        comparison["multiplicity_adjustment"] = dict(
            visual_multiplicity_adjustment
        )
        comparison["train_seed_consistency"] = _train_seed_consistency(
            comparison, rule=ALL_TRAIN_SEEDS_POSITIVE_CI_RULE
        )
        comparison["passes_15pp_with_positive_ci"] = _paired_difference_passes(
            comparison,
            minimum=MINIMUM_VISUAL_INTERVENTION_DROP,
            train_seed_consistency_rule=ALL_TRAIN_SEEDS_POSITIVE_CI_RULE,
        )
        visual_perturbations[intervention] = comparison
    best_visual_intervention = _best_comparison(visual_perturbations)
    visual_use_passed = bool(
        best_visual_intervention is not None
        and visual_perturbations[best_visual_intervention][
            "passes_15pp_with_positive_ci"
        ]
    )

    state_shuffle = paired_episode_success(
        normalized,
        first_variant=PRIMARY_VARIANT,
        first_intervention=CLEAN,
        second_variant=PRIMARY_VARIANT,
        second_intervention=SHUFFLE_STATE,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + 4,
    )
    state_shuffle["train_seed_consistency"] = _train_seed_consistency(
        state_shuffle, rule=TRAIN_SEED_CONSISTENCY_RULE
    )
    state_shuffle_passed = _paired_difference_passes(
        state_shuffle,
        minimum=MINIMUM_STATE_SHUFFLE_DROP,
        train_seed_consistency_rule=TRAIN_SEED_CONSISTENCY_RULE,
    )

    architecture_check = _architecture_check(architecture)
    training_check = _training_contract_check(training_contract)
    parameter_check = _parameter_budget_check(parameter_counts)
    legacy_check = _legacy_regression_check(
        legacy_suites, confidence=confidence
    )
    future_check = _future_probe_check(
        future_probe,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 5,
    )
    runtime_check = _runtime_check(runtime)
    evidence_check = _evidence_check(
        evidence,
        train_seeds=required_train_seeds,
    )

    checks = {
        "formal_protocol": _check(formal_protocol is True),
        "formal_episode_matrix": _check(matrix["passed"], **_without_passed(matrix)),
        "episode_record_schema": _check(
            aggregate["passed"],
            records=aggregate["records"],
            unique_identities=aggregate["unique_identities"],
            duplicate_identities=aggregate["duplicate_identities"],
            invalid_record_count=aggregate["invalid_record_count"],
        ),
        "architecture_contract": architecture_check,
        "training_contract": training_check,
        "trainable_parameter_budget": parameter_check,
        "policy_inputs_and_action_source": observation_contract,
        "vision_gain_10pp_positive_ci": _check(
            vision_gain_passed,
            threshold=MINIMUM_VISION_GAIN,
            comparison=vision_gain,
        ),
        "legacy_regression_at_most_5pp_each_suite": legacy_check,
        "visual_intervention_drop_15pp_positive_ci": _check(
            visual_use_passed,
            threshold=MINIMUM_VISUAL_INTERVENTION_DROP,
            selected_intervention=best_visual_intervention,
            multiplicity_adjustment=visual_multiplicity_adjustment,
            comparisons=visual_perturbations,
        ),
        "state_shuffle_drop_5pp_positive_ci": _check(
            state_shuffle_passed,
            threshold=MINIMUM_STATE_SHUFFLE_DROP,
            comparison=state_shuffle,
        ),
        "future_probe_beats_current_frame": future_check,
        "runtime_deadline_contract": runtime_check,
        "hashes_and_strict_reload": evidence_check,
    }
    technical_checks = {
        name: value for name, value in checks.items() if name != "formal_protocol"
    }
    technical_checks_passed = all(value["passed"] for value in technical_checks.values())
    passed = bool(formal_protocol is True and technical_checks_passed)
    return {
        "format_version": FORMAT_VERSION,
        "phase": "M1",
        "passed": passed,
        "claim_allowed": passed,
        "formal_protocol": formal_protocol is True,
        "technical_checks_passed": technical_checks_passed,
        "diagnostic_criteria_met": technical_checks_passed,
        "checks": checks,
        "thresholds": {
            "trainable_parameters": [
                MINIMUM_TRAINABLE_PARAMETERS,
                MAXIMUM_TRAINABLE_PARAMETERS,
            ],
            "formal_evaluation_seed_count": [
                MINIMUM_FORMAL_EVALUATION_SEEDS,
                MAXIMUM_FORMAL_EVALUATION_SEEDS,
            ],
            "required_train_seed_count": REQUIRED_TRAIN_SEEDS,
            "primary_value_and_state_train_seed_consistency": {
                "rule": TRAIN_SEED_CONSISTENCY_RULE,
                "minimum_positive_train_seeds": (
                    len(required_train_seeds) // 2 + 1
                ),
                "positive_seed_statistic": "mean_difference_gt_0",
                "per_seed_ci": "reported_not_gated",
            },
            "minimum_vision_gain": MINIMUM_VISION_GAIN,
            "maximum_legacy_regression": MAXIMUM_LEGACY_REGRESSION,
            "minimum_visual_intervention_drop": MINIMUM_VISUAL_INTERVENTION_DROP,
            "minimum_state_shuffle_drop": MINIMUM_STATE_SHUFFLE_DROP,
            "maximum_direct_p95_ms": MAXIMUM_DIRECT_P95_MS,
            "maximum_decimated_action_age_p95_ms": (
                MAXIMUM_DECIMATED_ACTION_AGE_P95_MS
            ),
        },
        "protocol": {
            "tasks": list(required_tasks),
            "evaluation_seeds": list(required_eval_seeds),
            "cue_variants": list(required_cues),
            "train_seeds": list(required_train_seeds),
            "required_variants": list(REQUIRED_VARIANTS),
            "primary_variant": PRIMARY_VARIANT,
            "confidence": confidence,
            "visual_intervention_multiplicity_adjustment": (
                visual_multiplicity_adjustment
            ),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "primary_value_and_state_train_seed_consistency_rule": (
                TRAIN_SEED_CONSISTENCY_RULE
            ),
            "visual_intervention_train_seed_consistency_rule": (
                ALL_TRAIN_SEEDS_POSITIVE_CI_RULE
            ),
        },
        "episode_aggregation": aggregate,
        "comparisons": {
            "vision_gain": vision_gain,
            "visual_interventions": visual_perturbations,
            "state_shuffle": state_shuffle,
        },
    }


def phase_m1_acceptance_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias with the phase-first naming used by older gates."""

    return m1_acceptance_report(*args, **kwargs)


def _formal_matrix_check(
    records: Sequence[Any],
    *,
    tasks: Sequence[str],
    evaluation_seeds: Sequence[int],
    cue_variants: Sequence[int],
    train_seeds: Sequence[int],
) -> dict[str, Any]:
    expected: set[tuple[str, str, int, str, int, int]] = set()
    conditions = [(variant, CLEAN) for variant in REQUIRED_VARIANTS]
    conditions.extend((PRIMARY_VARIANT, value) for value in PRIMARY_INTERVENTIONS)
    for variant, intervention in conditions:
        for train_seed in train_seeds:
            for task in tasks:
                for evaluation_seed in evaluation_seeds:
                    for cue in cue_variants:
                        expected.add(
                            (
                                variant,
                                intervention,
                                train_seed,
                                task,
                                evaluation_seed,
                                cue,
                            )
                        )
    identities = [record.identity for record in records]
    counts = Counter(identities)
    observed = set(identities)
    duplicate = sorted(identity for identity, count in counts.items() if count > 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    seed_count_valid = (
        len(train_seeds) == REQUIRED_TRAIN_SEEDS
        and MINIMUM_FORMAL_EVALUATION_SEEDS
        <= len(evaluation_seeds)
        <= MAXIMUM_FORMAL_EVALUATION_SEEDS
    )
    passed = bool(
        seed_count_valid
        and len(records) == len(expected)
        and not duplicate
        and not missing
        and not unexpected
    )
    return {
        "passed": passed,
        "expected_records": len(expected),
        "observed_records": len(records),
        "required_train_seed_count": REQUIRED_TRAIN_SEEDS,
        "observed_train_seed_count": len(train_seeds),
        "minimum_evaluation_seed_count": MINIMUM_FORMAL_EVALUATION_SEEDS,
        "maximum_evaluation_seed_count": MAXIMUM_FORMAL_EVALUATION_SEEDS,
        "observed_evaluation_seed_count": len(evaluation_seeds),
        "seed_counts_valid": seed_count_valid,
        "duplicates": [list(value) for value in duplicate[:100]],
        "duplicate_count": len(duplicate),
        "missing": [list(value) for value in missing[:100]],
        "missing_count": len(missing),
        "unexpected": [list(value) for value in unexpected[:100]],
        "unexpected_count": len(unexpected),
    }


def _observation_and_action_contract(records: Sequence[Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for record in records:
        consumed = set(record.consumed_observation_paths)
        presented = set(record.presented_observation_paths)
        required = _REQUIRED_INPUTS.get(record.model_variant, set())
        reasons: list[str] = []
        if not presented or not consumed:
            reasons.append("observation_paths_empty")
        if not consumed.issubset(presented):
            reasons.append("consumed_not_presented")
        if not required.issubset(consumed):
            reasons.append("required_inputs_missing")
        if record.model_variant == STATE_ONLY and any(
            path == "images" or path.startswith("images.") for path in consumed
        ):
            reasons.append("state_only_consumed_images")
        if record.model_variant == VISION_ONLY and any(
            path == "proprioception" or path.startswith("proprioception.")
            for path in consumed
        ):
            reasons.append("vision_only_consumed_proprioception")
        if _contains_privileged_path(presented | consumed):
            reasons.append("privileged_path")
        expected_source = EXPECTED_ACTION_SOURCE.get(record.model_variant)
        if record.action_source != expected_source:
            reasons.append("wrong_action_source")
        if record.privileged_observation_seen:
            reasons.append("privileged_observation_seen")
        if record.fallback_used:
            reasons.append("fallback_used")
        if not record.actions_finite_and_bounded:
            reasons.append("actions_not_finite_and_bounded")
        if reasons:
            violations.append(
                {
                    "identity": list(record.identity),
                    "reasons": reasons,
                    "expected_action_source": expected_source,
                    "actual_action_source": record.action_source,
                    "required_inputs": sorted(required),
                    "consumed_inputs": sorted(consumed),
                }
            )
    return _check(
        bool(records) and not violations,
        expected_action_source=dict(EXPECTED_ACTION_SOURCE),
        violation_count=len(violations),
        violations=violations[:100],
    )


def _architecture_check(value: Mapping[str, Any]) -> dict[str, Any]:
    required_true = (
        "state_feature_encoder_preserved",
        "world_heads_preserved",
        "stateful_action_flow",
        "prior_anchor_frozen",
        "prior_anchor_immutable",
        "visual_backbone_pretrained",
        "visual_backbone_frozen",
        "planning_feature_fuses_state_and_visual",
        "future_visual_latent_head",
        "action_residual",
    )
    true_fields = {name: value.get(name) is True for name in required_true}
    resampler_layers = _finite_number(value.get("resampler_layers"))
    visual_tokens = _finite_number(value.get("visual_tokens"))
    expected_horizons = value.get("future_horizons") == [1, 2, 4, 8] or value.get(
        "future_horizons"
    ) == (1, 2, 4, 8)
    exact_control_semantics = bool(
        value.get("action_flow_experts") == 1
        and value.get("action_chunk_steps") == 8
        and value.get("execution_steps") == 2
    )
    checks = {
        **true_fields,
        "resampler_layers_2_to_4": bool(
            resampler_layers is not None
            and 2 <= resampler_layers <= 4
            and resampler_layers.is_integer()
        ),
        "visual_tokens_8_to_32": bool(
            visual_tokens is not None
            and 8 <= visual_tokens <= 32
            and visual_tokens.is_integer()
        ),
        "future_horizons_1_2_4_8": expected_horizons,
        "single_expert_chunk8_execute2": exact_control_semantics,
    }
    return _check(all(checks.values()), checks=checks)


def _training_contract_check(value: Mapping[str, Any]) -> dict[str, Any]:
    stage_order = value.get("stage_order")
    first_three = (
        "adapter_fusion",
        "fusion_future_action",
        "legacy_low_lr",
    )
    valid_order = bool(
        isinstance(stage_order, Sequence)
        and not isinstance(stage_order, (str, bytes))
        and tuple(stage_order[:3]) == first_three
        and len(stage_order) in {3, 4}
        and (len(stage_order) == 3 or stage_order[3] == "vision_last_blocks")
    )
    ratio = _finite_number(value.get("legacy_learning_rate_ratio"))
    blocks = value.get("vision_unfrozen_blocks")
    valid_blocks = bool(
        isinstance(blocks, int)
        and not isinstance(blocks, bool)
        and (blocks == 0 or 2 <= blocks <= 4)
        and (
            blocks == 0
            or value.get("m1_stable_before_visual_unfreeze") is True
        )
    )
    checks = {
        "stage_order": valid_order,
        "stage1_old_world_action_frozen": value.get(
            "stage1_old_world_action_frozen"
        )
        is True,
        "stage1_visual_backbone_frozen": value.get(
            "stage1_visual_backbone_frozen"
        )
        is True,
        "stage2_visual_backbone_frozen": value.get(
            "stage2_visual_backbone_frozen"
        )
        is True,
        "legacy_lr_10_to_20_times_lower": bool(
            ratio is not None and 0.05 <= ratio <= 0.10
        ),
        "visual_unfreeze_only_2_to_4_after_stable": valid_blocks,
        "cold_replan_observation_regrounding": bool(
            value.get("replan_warm_start_enabled") is False
            and _finite_number(value.get("training_warm_start_probability")) == 0.0
            and value.get("observation_regrounding")
            == "cold_start_every_execute_2_replan"
            and value.get("cold_replan_scope")
            == "m1_latent_flow_visual_required_only"
        ),
    }
    return _check(all(checks.values()), checks=checks)


def _parameter_budget_check(value: Mapping[str, Any]) -> dict[str, Any]:
    trainable = _strict_nonnegative_integer(value.get("trainable"))
    frozen_visual = _strict_nonnegative_integer(
        value.get("frozen_visual_backbone")
    )
    passed = bool(
        trainable is not None
        and MINIMUM_TRAINABLE_PARAMETERS
        <= trainable
        <= MAXIMUM_TRAINABLE_PARAMETERS
        and frozen_visual is not None
        and frozen_visual > 0
    )
    return _check(
        passed,
        trainable=trainable,
        minimum_trainable=MINIMUM_TRAINABLE_PARAMETERS,
        maximum_trainable=MAXIMUM_TRAINABLE_PARAMETERS,
        frozen_visual_backbone=frozen_visual,
    )


def _legacy_regression_check(
    suites: Mapping[str, Any], *, confidence: float
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for suite in REQUIRED_LEGACY_SUITES:
        value = suites.get(suite)
        if not isinstance(value, Mapping):
            details[suite] = {"passed": False, "reason": "missing_suite"}
            continue
        m1_successes = _strict_nonnegative_integer(value.get("m1_successes"))
        m1_episodes = _strict_positive_integer(value.get("m1_episodes"))
        legacy_successes = _strict_nonnegative_integer(
            value.get("legacy_successes")
        )
        legacy_episodes = _strict_positive_integer(value.get("legacy_episodes"))
        valid = bool(
            m1_successes is not None
            and m1_episodes is not None
            and m1_successes <= m1_episodes
            and legacy_successes is not None
            and legacy_episodes is not None
            and legacy_successes <= legacy_episodes
        )
        if not valid:
            details[suite] = {"passed": False, "reason": "invalid_counts"}
            continue
        assert m1_successes is not None and m1_episodes is not None
        assert legacy_successes is not None and legacy_episodes is not None
        m1_rate = m1_successes / m1_episodes
        legacy_rate = legacy_successes / legacy_episodes
        regression = legacy_rate - m1_rate
        details[suite] = {
            "passed": regression <= MAXIMUM_LEGACY_REGRESSION + 1e-12,
            "m1": wilson_interval(m1_successes, m1_episodes, confidence=confidence),
            "legacy": wilson_interval(
                legacy_successes, legacy_episodes, confidence=confidence
            ),
            "legacy_minus_m1": regression,
            "maximum_regression": MAXIMUM_LEGACY_REGRESSION,
        }
    unexpected = sorted(set(suites) - set(REQUIRED_LEGACY_SUITES))
    passed = not unexpected and all(
        details.get(suite, {}).get("passed") is True
        for suite in REQUIRED_LEGACY_SUITES
    )
    return _check(passed, suites=details, unexpected_suites=unexpected)


def _future_probe_check(
    value: Mapping[str, Any],
    *,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if value.get("baseline") != "current_frame_only":
        return _check(False, reason="baseline_must_be_current_frame_only")
    try:
        object_comparison = paired_rmse_comparison(
            value["object_model_errors"],
            value["object_baseline_errors"],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        event_comparison = paired_balanced_accuracy_comparison(
            value["event_model_predictions"],
            value["event_baseline_predictions"],
            value["event_labels"],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
    except (KeyError, TypeError, ValueError) as error:
        return _check(False, reason=f"invalid_probe_evidence:{error}")
    object_passed = bool(
        object_comparison["baseline_minus_model_rmse"] > 0.0
        and object_comparison["ci_lower"] > 0.0
    )
    event_passed = bool(
        event_comparison["model_minus_baseline_balanced_accuracy"] > 0.0
        and event_comparison["ci_lower"] > 0.0
        and event_comparison["mcnemar"]["p_value_two_sided"] < 0.05
    )
    return _check(
        object_passed and event_passed,
        horizon=8,
        object_rmse_significantly_better=object_passed,
        event_balanced_accuracy_significantly_better=event_passed,
        object_comparison=object_comparison,
        event_comparison=event_comparison,
    )


def _runtime_check(value: Mapping[str, Any]) -> dict[str, Any]:
    latency = _finite_1d(value.get("sensor_to_action_ms"))
    action_age = _finite_1d(value.get("action_age_ms"))
    direct_p95 = _p95(latency)
    action_age_p95 = _p95(action_age)
    deadline_misses = _strict_nonnegative_integer(value.get("deadline_misses"))
    direct_passed = bool(
        direct_p95 is not None and direct_p95 < MAXIMUM_DIRECT_P95_MS
    )
    decimated_passed = bool(
        value.get("decimated") is True
        and action_age_p95 is not None
        and action_age_p95 <= MAXIMUM_DECIMATED_ACTION_AGE_P95_MS
        and deadline_misses is not None
    )
    return _check(
        direct_passed or decimated_passed,
        mode=(
            "direct_under_50ms"
            if direct_passed
            else "decimated_action_age_p95_under_100ms"
            if decimated_passed
            else None
        ),
        sensor_to_action_p95_ms=direct_p95,
        action_age_p95_ms=action_age_p95,
        deadline_misses=deadline_misses,
        direct_path_passed=direct_passed,
        decimated_path_passed=decimated_passed,
    )


def _evidence_check(
    value: Mapping[str, Any], *, train_seeds: Sequence[int]
) -> dict[str, Any]:
    artifacts = value.get("artifact_sha256")
    artifact_names = (
        "dataset_manifest",
        "episode_records",
        "config",
        "visual_backbone",
    )
    artifact_hashes_valid = bool(
        isinstance(artifacts, Mapping)
        and all(_is_sha256(artifacts.get(name)) for name in artifact_names)
    )
    checkpoint_hashes = value.get("checkpoint_sha256")
    strict_reload = value.get("strict_reload")
    checkpoint_matrix_valid = isinstance(checkpoint_hashes, Mapping)
    reload_matrix_valid = isinstance(strict_reload, Mapping)
    checkpoint_details: dict[str, Any] = {}
    for variant in REQUIRED_VARIANTS:
        variant_hashes = (
            checkpoint_hashes.get(variant)
            if isinstance(checkpoint_hashes, Mapping)
            else None
        )
        variant_reload = (
            strict_reload.get(variant) if isinstance(strict_reload, Mapping) else None
        )
        seed_details: dict[str, Any] = {}
        for train_seed in train_seeds:
            checkpoint_hash = _seed_lookup(variant_hashes, train_seed)
            reload_value = _seed_lookup(variant_reload, train_seed)
            hash_valid = _is_sha256(checkpoint_hash)
            reload_valid = bool(
                isinstance(reload_value, Mapping)
                and reload_value.get("passed") is True
                and _exact_zero(reload_value.get("max_abs_diff"))
            )
            checkpoint_matrix_valid = checkpoint_matrix_valid and hash_valid
            reload_matrix_valid = reload_matrix_valid and reload_valid
            seed_details[str(train_seed)] = {
                "checkpoint_sha256_valid": hash_valid,
                "strict_reload_exact": reload_valid,
            }
        checkpoint_details[variant] = seed_details
    source_immutable = value.get("source_checkpoint_immutable") is True
    visual_hash_matches = bool(
        isinstance(artifacts, Mapping)
        and _is_sha256(value.get("visual_backbone_weights_sha256"))
        and value.get("visual_backbone_weights_sha256")
        == artifacts.get("visual_backbone")
    )
    checks = {
        "artifact_hashes_valid": artifact_hashes_valid,
        "all_checkpoint_hashes_valid": checkpoint_matrix_valid,
        "all_strict_reloads_exact": reload_matrix_valid,
        "source_checkpoint_immutable": source_immutable,
        "visual_backbone_hash_matches": visual_hash_matches,
    }
    return _check(
        all(checks.values()), checks=checks, checkpoint_matrix=checkpoint_details
    )


def _paired_difference_passes(
    value: Mapping[str, Any],
    *,
    minimum: float,
    train_seed_consistency_rule: str = ALL_TRAIN_SEEDS_POSITIVE_CI_RULE,
) -> bool:
    difference = value.get("difference")
    return bool(
        value.get("exact_pairs") is True
        and not value.get("duplicate_first_keys")
        and not value.get("duplicate_second_keys")
        and isinstance(difference, Mapping)
        and _finite_number(difference.get("mean_difference")) is not None
        and difference["mean_difference"] + 1e-12 >= minimum
        and _finite_number(difference.get("ci_lower")) is not None
        and difference["ci_lower"] > 0.0
        and _cluster_evidence_valid(
            value, train_seed_consistency_rule=train_seed_consistency_rule
        )
    )


def _cluster_evidence_valid(
    value: Mapping[str, Any], *, train_seed_consistency_rule: str
) -> bool:
    """Require complete cluster evidence and the selected seed policy."""

    difference = value.get("difference")
    if not isinstance(difference, Mapping) or not _cluster_difference_valid(
        difference
    ):
        return False
    return _train_seed_consistency(
        value, rule=train_seed_consistency_rule
    )["passed"]


def _train_seed_consistency(
    value: Mapping[str, Any], *, rule: str
) -> dict[str, Any]:
    """Summarize an auditable cross-training-seed rule.

    The aggregate comparison still has to clear its fixed effect threshold and
    positive clustered confidence interval.  The primary visual-value and state
    dependence checks use a strict majority of seed point-estimate directions;
    visual interventions retain the original all-seed positive-CI rule.
    """

    if rule not in {
        TRAIN_SEED_CONSISTENCY_RULE,
        ALL_TRAIN_SEEDS_POSITIVE_CI_RULE,
    }:
        raise ValueError(f"unsupported train seed consistency rule: {rule}")

    per_train_seed = value.get("per_train_seed")
    if not isinstance(per_train_seed, Mapping) or not per_train_seed:
        return {
            "rule": rule,
            "train_seed_count": 0,
            "required_positive_train_seeds": 0,
            "positive_train_seed_count": 0,
            "positive_ci_train_seed_count": 0,
            "per_seed_ci_gated": rule == ALL_TRAIN_SEEDS_POSITIVE_CI_RULE,
            "all_seed_evidence_valid": False,
            "passed": False,
            "per_train_seed": {},
        }

    seed_details: dict[str, Any] = {}
    positive_count = 0
    positive_ci_count = 0
    all_seed_evidence_valid = True
    for train_seed, comparison in per_train_seed.items():
        difference = (
            comparison.get("difference")
            if isinstance(comparison, Mapping)
            else None
        )
        mean_difference = (
            _finite_number(difference.get("mean_difference"))
            if isinstance(difference, Mapping)
            else None
        )
        ci_lower = (
            _finite_number(difference.get("ci_lower"))
            if isinstance(difference, Mapping)
            else None
        )
        evidence_valid = bool(
            isinstance(comparison, Mapping)
            and comparison.get("exact_pairs") is True
            and not comparison.get("duplicate_first_keys")
            and not comparison.get("duplicate_second_keys")
            and isinstance(difference, Mapping)
            and _cluster_difference_valid(difference)
            and mean_difference is not None
            and ci_lower is not None
        )
        direction_positive = bool(
            evidence_valid and mean_difference is not None and mean_difference > 0.0
        )
        ci_positive = bool(evidence_valid and ci_lower is not None and ci_lower > 0.0)
        positive_count += int(direction_positive)
        positive_ci_count += int(ci_positive)
        all_seed_evidence_valid = all_seed_evidence_valid and evidence_valid
        seed_details[str(train_seed)] = {
            "evidence_valid": evidence_valid,
            "mean_difference": mean_difference,
            "ci_lower": ci_lower,
            "direction_positive": direction_positive,
            "ci_positive": ci_positive,
        }

    train_seed_count = len(per_train_seed)
    required_positive = train_seed_count // 2 + 1
    if rule == TRAIN_SEED_CONSISTENCY_RULE:
        direction_requirement_met = positive_count >= required_positive
    else:
        direction_requirement_met = positive_ci_count == train_seed_count
    passed = bool(
        train_seed_count >= REQUIRED_TRAIN_SEEDS
        and all_seed_evidence_valid
        and direction_requirement_met
    )
    return {
        "rule": rule,
        "train_seed_count": train_seed_count,
        "required_positive_train_seeds": required_positive,
        "positive_train_seed_count": positive_count,
        "positive_ci_train_seed_count": positive_ci_count,
        "per_seed_ci_gated": rule == ALL_TRAIN_SEEDS_POSITIVE_CI_RULE,
        "all_seed_evidence_valid": all_seed_evidence_valid,
        "passed": passed,
        "per_train_seed": seed_details,
    }


def _cluster_difference_valid(value: Mapping[str, Any]) -> bool:
    clusters = _strict_positive_integer(value.get("clusters"))
    pairs = _strict_positive_integer(value.get("pairs"))
    cluster_ids = value.get("cluster_ids")
    records_per_cluster = value.get("records_per_cluster")
    if (
        value.get("cluster_bootstrap") is not True
        or value.get("cluster_key") != "evaluation_seed"
        or clusters is None
        or clusters < 2
        or pairs is None
        or not isinstance(cluster_ids, Sequence)
        or isinstance(cluster_ids, (str, bytes))
        or len(cluster_ids) != clusters
        or len(set(cluster_ids)) != clusters
        or not isinstance(records_per_cluster, Mapping)
        or len(records_per_cluster) != clusters
    ):
        return False
    cluster_sizes = [
        _strict_positive_integer(records_per_cluster.get(str(cluster_id)))
        for cluster_id in cluster_ids
    ]
    return bool(
        all(size is not None for size in cluster_sizes)
        and sum(int(size) for size in cluster_sizes if size is not None) == pairs
    )


def _best_comparison(comparisons: Mapping[str, Any]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for name, comparison in comparisons.items():
        difference = comparison.get("difference")
        if isinstance(difference, Mapping):
            estimate = _finite_number(difference.get("mean_difference"))
            if estimate is not None:
                candidates.append((estimate, name))
    return max(candidates)[1] if candidates else None


def _contains_privileged_path(paths: Iterable[str]) -> bool:
    for path in paths:
        parts = {part.lower() for part in str(path).split(".")}
        if parts & _FORBIDDEN_PATH_PARTS:
            return True
    return False


def _unique_nonempty_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _unique_nonnegative_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain integers")
        if int(value) < 0:
            raise ValueError(f"{name} must contain non-negative integers")
        normalized.append(int(value))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(normalized)


def _finite_1d(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        return None
    return array


def _p95(value: np.ndarray | None) -> float | None:
    return None if value is None else float(np.quantile(value, 0.95))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strict_nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value) if int(value) >= 0 else None


def _strict_positive_integer(value: Any) -> int | None:
    result = _strict_nonnegative_integer(value)
    return result if result is not None and result > 0 else None


def _exact_zero(value: Any) -> bool:
    number = _finite_number(value)
    return number == 0.0 if number is not None else False


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _seed_lookup(value: Any, seed: int) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get(seed, value.get(str(seed)))


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    details.pop("passed", None)
    return {"passed": bool(passed), **details}


def _without_passed(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "passed"}


__all__ = [
    "CLEAN",
    "ALL_TRAIN_SEEDS_POSITIVE_CI_RULE",
    "EXPECTED_ACTION_SOURCE",
    "FORMAT_VERSION",
    "FREEZE_FIRST_RGB",
    "OCCLUDE_CUE",
    "PARAMETER_MATCHED_MLP",
    "PRIMARY_VARIANT",
    "REQUIRED_VARIANTS",
    "SHUFFLE_RGB",
    "SHUFFLE_STATE",
    "STATE_ONLY",
    "STATE_VISION_FUTURE",
    "STATE_VISION_NO_FUTURE",
    "TRAIN_SEED_CONSISTENCY_RULE",
    "VISION_ONLY",
    "VISUAL_INTERVENTIONS",
    "m1_acceptance_report",
    "phase_m1_acceptance_report",
]
