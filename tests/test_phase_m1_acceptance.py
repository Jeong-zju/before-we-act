from __future__ import annotations

from copy import deepcopy

import pytest

from eval.m1_acceptance import (
    CLEAN,
    EXPECTED_ACTION_SOURCE,
    FREEZE_FIRST_RGB,
    OCCLUDE_CUE,
    PARAMETER_MATCHED_MLP,
    PRIMARY_VARIANT,
    REQUIRED_VARIANTS,
    SHUFFLE_RGB,
    SHUFFLE_STATE,
    STATE_ONLY,
    STATE_VISION_NO_FUTURE,
    VISION_ONLY,
    m1_acceptance_report,
)


TRAIN_SEEDS = (11, 22, 33)
EVALUATION_SEEDS = tuple(range(100))
CUES = (0, 1)
TASKS = ("visual_required",)


def _consumed_paths(variant: str) -> tuple[str, ...]:
    common = ("past_executed_actions", "task.id", "task.text")
    if variant == STATE_ONLY:
        return (*common, "proprioception")
    if variant == VISION_ONLY:
        return (*common, "images.fixed")
    return (*common, "images.fixed", "proprioception")


def _success_threshold(variant: str, intervention: str) -> int:
    if intervention == SHUFFLE_RGB:
        return 50
    if intervention == FREEZE_FIRST_RGB:
        return 60
    if intervention == OCCLUDE_CUE:
        return 55
    if intervention == SHUFFLE_STATE:
        return 65
    return {
        STATE_ONLY: 50,
        VISION_ONLY: 70,
        STATE_VISION_NO_FUTURE: 75,
        PRIMARY_VARIANT: 80,
        PARAMETER_MATCHED_MLP: 60,
    }[variant]


def _record(
    *,
    variant: str,
    intervention: str,
    train_seed: int,
    evaluation_seed: int,
    cue: int,
) -> dict:
    presented = (
        "past_executed_actions",
        "task.id",
        "task.text",
        "images.fixed",
        "proprioception",
    )
    return {
        "task_id": TASKS[0],
        "evaluation_seed": evaluation_seed,
        "cue_id": cue,
        "model_variant": variant,
        "train_seed": train_seed,
        "intervention": intervention,
        "success": evaluation_seed < _success_threshold(variant, intervention),
        "steps": 40,
        "total_reward": float(evaluation_seed < _success_threshold(variant, intervention)),
        "action_source": EXPECTED_ACTION_SOURCE[variant],
        "presented_observation_paths": presented,
        "consumed_observation_paths": _consumed_paths(variant),
        "privileged_observation_seen": False,
        "fallback_used": False,
        "actions_finite_and_bounded": True,
    }


def _records() -> list[dict]:
    result: list[dict] = []
    conditions = [(variant, CLEAN) for variant in REQUIRED_VARIANTS]
    conditions.extend(
        (PRIMARY_VARIANT, intervention)
        for intervention in (
            SHUFFLE_RGB,
            FREEZE_FIRST_RGB,
            OCCLUDE_CUE,
            SHUFFLE_STATE,
        )
    )
    for variant, intervention in conditions:
        for train_seed in TRAIN_SEEDS:
            for evaluation_seed in EVALUATION_SEEDS:
                for cue in CUES:
                    result.append(
                        _record(
                            variant=variant,
                            intervention=intervention,
                            train_seed=train_seed,
                            evaluation_seed=evaluation_seed,
                            cue=cue,
                        )
                    )
    return result


def _future_probe() -> dict:
    labels = [index % 2 for index in range(200)]
    model = labels.copy()
    baseline = labels.copy()
    for pair in range(100):
        indices = slice(2 * pair, 2 * pair + 2)
        if pair % 10 == 0:
            model[indices] = [1 - value for value in labels[indices]]
        if pair % 10 < 4:
            baseline[indices] = [1 - value for value in labels[indices]]
    return {
        "baseline": "current_frame_only",
        "object_model_errors": [0.1] * 200,
        "object_baseline_errors": [0.3] * 200,
        "event_labels": labels,
        "event_model_predictions": model,
        "event_baseline_predictions": baseline,
    }


def _evidence() -> dict:
    visual_hash = "d" * 64
    return {
        "artifact_sha256": {
            "dataset_manifest": "a" * 64,
            "episode_records": "b" * 64,
            "config": "c" * 64,
            "visual_backbone": visual_hash,
        },
        "visual_backbone_weights_sha256": visual_hash,
        "checkpoint_sha256": {
            variant: {str(seed): "e" * 64 for seed in TRAIN_SEEDS}
            for variant in REQUIRED_VARIANTS
        },
        "strict_reload": {
            variant: {
                str(seed): {"passed": True, "max_abs_diff": 0.0}
                for seed in TRAIN_SEEDS
            }
            for variant in REQUIRED_VARIANTS
        },
        "source_checkpoint_immutable": True,
    }


def _payload() -> dict:
    return {
        "records": _records(),
        "tasks": TASKS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "cue_variants": CUES,
        "train_seeds": TRAIN_SEEDS,
        "architecture": {
            "state_feature_encoder_preserved": True,
            "world_heads_preserved": True,
            "stateful_action_flow": True,
            "prior_anchor_frozen": True,
            "prior_anchor_immutable": True,
            "visual_backbone_pretrained": True,
            "visual_backbone_frozen": True,
            "resampler_layers": 3,
            "visual_tokens": 16,
            "planning_feature_fuses_state_and_visual": True,
            "future_visual_latent_head": True,
            "future_horizons": [1, 2, 4, 8],
            "action_residual": True,
            "action_flow_experts": 1,
            "action_chunk_steps": 8,
            "execution_steps": 2,
        },
        "training_contract": {
            "stage_order": [
                "adapter_fusion",
                "fusion_future_action",
                "legacy_low_lr",
            ],
            "stage1_old_world_action_frozen": True,
            "stage1_visual_backbone_frozen": True,
            "stage2_visual_backbone_frozen": True,
            "legacy_learning_rate_ratio": 0.075,
            "vision_unfrozen_blocks": 0,
            "m1_stable_before_visual_unfreeze": False,
            "replan_warm_start_enabled": False,
            "training_warm_start_probability": 0.0,
            "observation_regrounding": "cold_start_every_execute_2_replan",
            "cold_replan_scope": "m1_latent_flow_visual_required_only",
        },
        "parameter_counts": {
            "trainable": 35_000_000,
            "frozen_visual_backbone": 11_000_000,
        },
        "legacy_suites": {
            "standard": {
                "m1_successes": 97,
                "m1_episodes": 100,
                "legacy_successes": 100,
                "legacy_episodes": 100,
            },
            "challenge": {
                "m1_successes": 92,
                "m1_episodes": 100,
                "legacy_successes": 95,
                "legacy_episodes": 100,
            },
        },
        "future_probe": _future_probe(),
        "runtime": {
            "sensor_to_action_ms": [10.0, 20.0, 30.0, 40.0],
            "decimated": False,
            "action_age_ms": [0.0],
            "deadline_misses": 0,
        },
        "evidence": _evidence(),
        "formal_protocol": True,
        "bootstrap_samples": 256,
        "bootstrap_seed": 12345,
    }


def _report(payload: dict) -> dict:
    return m1_acceptance_report(**payload)


def test_phase_m1_full_formal_evidence_passes_every_gate() -> None:
    report = _report(_payload())
    assert report["passed"]
    assert report["claim_allowed"]
    assert report["technical_checks_passed"]
    assert report["checks"]["formal_episode_matrix"]["expected_records"] == 5_400
    assert report["comparisons"]["vision_gain"]["difference"][
        "mean_difference"
    ] == pytest.approx(0.30)
    assert report["checks"]["visual_intervention_drop_15pp_positive_ci"][
        "selected_intervention"
    ] == SHUFFLE_RGB


def test_visual_intervention_selection_uses_bonferroni_confidence() -> None:
    report = _report(_payload())
    check = report["checks"]["visual_intervention_drop_15pp_positive_ci"]
    adjustment = check["multiplicity_adjustment"]
    expected_confidence = 1.0 - (1.0 - 0.95) / 3

    assert adjustment == {
        "method": "bonferroni_simultaneous_confidence_intervals",
        "selection": "best_of_three_post_selected_visual_interventions",
        "family_size": 3,
        "family_wise_confidence": 0.95,
        "per_comparison_confidence": pytest.approx(expected_confidence),
    }
    for comparison in report["comparisons"]["visual_interventions"].values():
        assert comparison["difference"]["confidence"] == pytest.approx(
            expected_confidence
        )
        assert comparison["multiplicity_adjustment"] == adjustment


def test_closed_loop_acceptance_fails_without_cluster_evidence(monkeypatch) -> None:
    import eval.m1_acceptance as acceptance

    original = acceptance.paired_episode_success

    def without_cluster_evidence(*args, **kwargs):
        comparison = original(*args, **kwargs)
        comparison["difference"].pop("cluster_bootstrap")
        return comparison

    monkeypatch.setattr(
        acceptance, "paired_episode_success", without_cluster_evidence
    )
    report = _report(_payload())

    assert not report["passed"]
    assert not report["checks"]["vision_gain_10pp_positive_ci"]["passed"]
    assert not report["checks"][
        "visual_intervention_drop_15pp_positive_ci"
    ]["passed"]
    assert not report["checks"]["state_shuffle_drop_5pp_positive_ci"]["passed"]


def test_closed_loop_gate_allows_one_reversed_training_seed() -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["model_variant"] != PRIMARY_VARIANT:
            continue
        train_seed = record["train_seed"]
        evaluation_seed = record["evaluation_seed"]
        if record["intervention"] == CLEAN:
            record["success"] = evaluation_seed < (100 if train_seed != 33 else 40)
        elif train_seed == 33 and record["intervention"] == SHUFFLE_STATE:
            # This seed reverses the causal effects, while the two perfect
            # seeds keep the pooled point estimates above their thresholds.
            record["success"] = evaluation_seed < 60
        elif train_seed == 33:
            # Visual interventions keep their stricter all-seed positive-CI
            # rule; isolate this test to the relaxed value/state checks.
            record["success"] = False

    report = _report(payload)

    assert report["comparisons"]["vision_gain"]["difference"][
        "mean_difference"
    ] >= 0.10
    assert report["comparisons"]["vision_gain"]["difference"]["ci_lower"] > 0.0
    assert report["comparisons"]["vision_gain"]["per_train_seed"]["33"][
        "difference"
    ]["mean_difference"] < 0.0
    consistency = report["comparisons"]["vision_gain"][
        "train_seed_consistency"
    ]
    assert consistency["positive_train_seed_count"] == 2
    assert consistency["required_positive_train_seeds"] == 2
    assert consistency["positive_ci_train_seed_count"] == 2
    assert not consistency["per_seed_ci_gated"]
    assert consistency["passed"]
    assert report["checks"]["vision_gain_10pp_positive_ci"]["passed"]
    assert report["comparisons"]["visual_interventions"][SHUFFLE_RGB][
        "train_seed_consistency"
    ]["rule"] == "all_train_seeds_positive_ci"
    assert report["comparisons"]["visual_interventions"][SHUFFLE_RGB][
        "train_seed_consistency"
    ]["per_seed_ci_gated"]
    assert report["passed"]


def test_majority_seed_rule_reports_but_does_not_gate_per_seed_ci() -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["model_variant"] != PRIMARY_VARIANT:
            continue
        train_seed = record["train_seed"]
        evaluation_seed = record["evaluation_seed"]
        if record["intervention"] == CLEAN:
            threshold = 100 if train_seed == TRAIN_SEEDS[0] else 51
            record["success"] = evaluation_seed < threshold
        elif record["intervention"] == SHUFFLE_STATE:
            threshold = 0 if train_seed == TRAIN_SEEDS[0] else 50
            record["success"] = evaluation_seed < threshold
        else:
            record["success"] = False

    report = _report(payload)

    comparison = report["comparisons"]["vision_gain"]
    assert comparison["difference"]["mean_difference"] >= 0.10
    assert comparison["difference"]["ci_lower"] > 0.0
    consistency = comparison["train_seed_consistency"]
    assert consistency["positive_train_seed_count"] == 3
    assert consistency["positive_ci_train_seed_count"] < 3
    assert consistency["passed"]
    assert report["checks"]["vision_gain_10pp_positive_ci"]["passed"]
    assert report["passed"]


def test_closed_loop_gate_rejects_when_only_one_training_seed_is_positive() -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["model_variant"] != STATE_ONLY:
            continue
        train_seed = record["train_seed"]
        threshold = 0 if train_seed == TRAIN_SEEDS[0] else 81
        record["success"] = record["evaluation_seed"] < threshold

    report = _report(payload)

    comparison = report["comparisons"]["vision_gain"]
    assert comparison["difference"]["mean_difference"] >= 0.10
    assert comparison["difference"]["ci_lower"] > 0.0
    consistency = comparison["train_seed_consistency"]
    assert consistency["positive_train_seed_count"] == 1
    assert consistency["required_positive_train_seeds"] == 2
    assert not consistency["passed"]
    assert not report["checks"]["vision_gain_10pp_positive_ci"]["passed"]
    assert not report["passed"]


def test_diagnostic_evidence_can_never_be_accepted() -> None:
    payload = _payload()
    payload["formal_protocol"] = False
    report = _report(payload)
    assert not report["passed"]
    assert not report["claim_allowed"]
    assert report["diagnostic_criteria_met"]
    assert not report["checks"]["formal_protocol"]["passed"]


def test_decimated_runtime_path_uses_action_age_p95_and_reports_misses() -> None:
    payload = _payload()
    payload["runtime"] = {
        "sensor_to_action_ms": [70.0, 80.0, 90.0],
        "decimated": True,
        "action_age_ms": [80.0, 90.0, 100.0],
        "deadline_misses": 0,
    }
    report = _report(payload)
    check = report["checks"]["runtime_deadline_contract"]
    assert report["passed"]
    assert check["mode"] == "decimated_action_age_p95_under_100ms"

    payload["runtime"]["deadline_misses"] = 1
    report = _report(payload)
    assert report["passed"]
    assert report["checks"]["runtime_deadline_contract"]["deadline_misses"] == 1

    payload["runtime"]["deadline_misses"] = None
    assert not _report(payload)["passed"]


def _mutate_missing_variant(payload: dict) -> None:
    payload["records"] = [
        record
        for record in payload["records"]
        if record["model_variant"] != PARAMETER_MATCHED_MLP
    ]


def _mutate_train_seed_count(payload: dict) -> None:
    payload["train_seeds"] = TRAIN_SEEDS[:2]
    payload["records"] = [
        record
        for record in payload["records"]
        if record["train_seed"] in TRAIN_SEEDS[:2]
    ]


def _mutate_parameter_budget(payload: dict) -> None:
    payload["parameter_counts"]["trainable"] = 19_999_999


def _mutate_architecture(payload: dict) -> None:
    payload["architecture"]["future_horizons"] = [1, 2, 4]


def _mutate_training_contract(payload: dict) -> None:
    payload["training_contract"]["legacy_learning_rate_ratio"] = 0.5


def _mutate_cold_replan_contract(payload: dict) -> None:
    payload["training_contract"]["replan_warm_start_enabled"] = True


def _mutate_vision_gain(payload: dict) -> None:
    for record in payload["records"]:
        if (
            record["model_variant"] == PRIMARY_VARIANT
            and record["intervention"] == CLEAN
        ):
            record["success"] = record["evaluation_seed"] < 50


def _mutate_legacy_regression(payload: dict) -> None:
    payload["legacy_suites"]["standard"]["m1_successes"] = 90


def _mutate_visual_perturbations(payload: dict) -> None:
    for record in payload["records"]:
        if record["intervention"] in {SHUFFLE_RGB, FREEZE_FIRST_RGB, OCCLUDE_CUE}:
            record["success"] = record["evaluation_seed"] < 80


def _mutate_state_shuffle(payload: dict) -> None:
    for record in payload["records"]:
        if record["intervention"] == SHUFFLE_STATE:
            record["success"] = record["evaluation_seed"] < 80


def _mutate_future_probe(payload: dict) -> None:
    payload["future_probe"]["object_model_errors"] = [0.3] * 200


def _mutate_runtime(payload: dict) -> None:
    payload["runtime"] = {
        "sensor_to_action_ms": [80.0, 90.0],
        "decimated": True,
        "action_age_ms": [101.0, 120.0],
        "deadline_misses": 0,
    }


def _mutate_action_source(payload: dict) -> None:
    payload["records"][0]["action_source"] = "fallback"


def _mutate_privileged_input(payload: dict) -> None:
    paths = payload["records"][0]["consumed_observation_paths"]
    payload["records"][0]["consumed_observation_paths"] = (
        *paths,
        "privileged_state.cue_id",
    )
    payload["records"][0]["presented_observation_paths"] = (
        *payload["records"][0]["presented_observation_paths"],
        "privileged_state.cue_id",
    )


def _mutate_hash(payload: dict) -> None:
    payload["evidence"]["artifact_sha256"]["dataset_manifest"] = "not-a-hash"


def _mutate_reload(payload: dict) -> None:
    payload["evidence"]["strict_reload"][STATE_ONLY][str(TRAIN_SEEDS[0])][
        "max_abs_diff"
    ] = 1e-8


@pytest.mark.parametrize(
    "mutator",
    [
        _mutate_missing_variant,
        _mutate_train_seed_count,
        _mutate_parameter_budget,
        _mutate_architecture,
        _mutate_training_contract,
        _mutate_cold_replan_contract,
        _mutate_vision_gain,
        _mutate_legacy_regression,
        _mutate_visual_perturbations,
        _mutate_state_shuffle,
        _mutate_future_probe,
        _mutate_runtime,
        _mutate_action_source,
        _mutate_privileged_input,
        _mutate_hash,
        _mutate_reload,
    ],
)
def test_every_required_m1_gate_fails_closed(mutator) -> None:
    payload = deepcopy(_payload())
    mutator(payload)
    report = _report(payload)
    assert not report["passed"]
    assert not report["technical_checks_passed"]
