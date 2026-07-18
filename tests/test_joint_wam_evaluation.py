from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from eval.closed_loop import ClosedLoopEpisode
from eval.joint_wam import (
    joint_wam_acceptance_report,
    select_video_episodes,
    validate_video_evidence,
)
from scripts.evaluate_joint_wam import (
    _prepare_output_directory,
    _settings as evaluation_settings,
    _training_seeds,
    build_parser as build_evaluation_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/wam/joint_wam.yaml"
POLICIES = (
    "joint_wam_direct",
    "action_prior",
    "stationary",
    "scripted_oracle",
)


def test_config_locks_formal_evaluation_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["name"] == "wam.cooperative_stop/joint-wam"
    assert config["evaluation"]["episodes_per_suite"] == 500
    assert config["evaluation"]["standard_seed_start"] == 120_000
    assert config["evaluation"]["challenge_seed_start"] == 220_000
    assert tuple(config["evaluation"]["policies"]) == POLICIES
    assert config["evaluation"]["same_seeds_for_all_policies"] is True
    assert config["acceptance"]["minimum_episodes_per_suite"] == 500
    assert config["acceptance"]["minimum_direct_success_rate"] == 0.90
    assert config["acceptance"]["maximum_prior_success_regression"] == 0.10
    assert config["acceptance"]["required_direct_execution_rate"] == 1.0
    assert config["acceptance"]["required_fallback_trigger_rate"] == 0.0
    assert config["acceptance"]["require_strict_joint_evidence"] is True
    assert config["acceptance"]["require_source_checkpoint_immutability"] is True
    assert config["acceptance"]["require_strict_checkpoint_reload"] is True
    assert config["video"]["selection"] == "sorted_smallest_seed"
    assert config["video"]["success_per_suite"] == 3
    assert config["video"]["failure_global_max"] == 3
    assert config["video"]["sidecar_required"] is True
    assert config["video"]["fallback_enabled"] is False


def test_default_cli_settings_are_the_only_formal_protocol() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    args = build_evaluation_parser().parse_args(["--config", str(CONFIG_PATH)])

    settings = evaluation_settings(config, args)

    assert settings["formal_protocol"] is True
    assert settings["episodes"] == 500
    assert settings["minimum_episodes"] == 500
    assert settings["suite_seeds"]["standard"] == tuple(
        range(120_000, 120_500)
    )
    assert settings["suite_seeds"]["challenge"] == tuple(
        range(220_000, 220_500)
    )
    assert settings["policies"] == (*POLICIES, "joint_wam_with_fallback")
    assert settings["max_steps"] is None
    assert settings["render_videos"] is True


@pytest.mark.parametrize(
    "override",
    (
        ("--episodes", "10"),
        ("--standard-seed-start", "130000"),
        ("--challenge-seed-start", "230000"),
        ("--policies", "joint_wam_direct", "action_prior"),
        ("--max-steps", "10"),
        ("--skip-videos",),
        ("--world-model-checkpoint-dir", "/tmp/world_model"),
        ("--action-prior-checkpoint-dir", "/tmp/prior"),
        ("--checkpoint-dir", "/tmp/joint_wam"),
    ),
)
def test_cli_overrides_require_separate_output_and_remain_diagnostic(
    tmp_path: Path,
    override: tuple[str, ...],
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = build_evaluation_parser()
    base = ["--config", str(CONFIG_PATH)]

    with pytest.raises(ValueError, match="separate --output-dir"):
        evaluation_settings(config, parser.parse_args([*base, *override]))

    settings = evaluation_settings(
        config,
        parser.parse_args(
            [
                *base,
                *override,
                "--output-dir",
                str(tmp_path / "diagnostic"),
            ]
        ),
    )
    assert settings["formal_protocol"] is False
    assert settings["minimum_episodes"] == 500


def test_output_override_is_diagnostic_even_without_other_overrides(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    args = build_evaluation_parser().parse_args(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "diagnostic"),
        ]
    )

    settings = evaluation_settings(config, args)

    assert settings["formal_protocol"] is False
    assert settings["episodes"] == 500
    assert settings["minimum_episodes"] == 500


def test_acceptance_passes_exact_500_seed_90_percent_boundary() -> None:
    report = _acceptance_report(
        _formal_metrics(flow_success_rate=0.90),
        records_by_suite=_formal_records(flow_success_rate=0.90),
    )

    assert report["passed"]
    assert report["policy_acceptable"] is True
    assert report["held_out_seed_overlap"] == 0
    for suite in ("standard", "challenge"):
        assert report["suites"][suite]["passed"]
        assert report["suites"][suite]["direct_success_rate"] == 0.90
        assert report["suites"][suite]["prior_regression"] == pytest.approx(0.10)
        assert all(report["suites"][suite]["checks"].values())


def test_fallback_metrics_are_report_only_and_cannot_change_acceptance() -> None:
    metrics = _formal_metrics(flow_success_rate=0.95)
    for suite, seed_start in (("standard", 120_000), ("challenge", 220_000)):
        metrics[suite]["joint_wam_with_fallback"] = _closed_loop_metrics(
            seeds=list(range(seed_start, seed_start + 500)),
            success_rate=0.0,
            mode="joint_wam_with_fallback",
            residual=True,
        )

    failed_fallback = _acceptance_report(
        metrics,
        records_by_suite=_formal_records(fallback_success_rate=0.0),
    )
    assert failed_fallback["passed"]
    assert failed_fallback["report_only"]["fallback_present_in_all_suites"]
    assert not failed_fallback["report_only"]["fallback_affects_policy_acceptance"]

    for suite in ("standard", "challenge"):
        metrics[suite]["joint_wam_with_fallback"]["success_rate"] = 1.0
        metrics[suite]["joint_wam_direct"]["success_rate"] = 0.0
    rescued_only_by_fallback = _acceptance_report(
        metrics,
        records_by_suite=_formal_records(
            flow_success_rate=0.0,
            fallback_success_rate=1.0,
        ),
    )
    assert not rescued_only_by_fallback["passed"]
    assert rescued_only_by_fallback["policy_acceptable"] is False


def test_missing_direct_policy_fails_closed() -> None:
    metrics = _formal_metrics()
    del metrics["standard"]["joint_wam_direct"]

    report = _acceptance_report(metrics, records_by_suite=_formal_records())

    assert not report["passed"]
    assert report["suites"]["standard"]["checks"] == {
        "required_policies_present": False
    }


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("too_few_episodes", "minimum_episodes"),
        ("success_below_90", "minimum_direct_success_rate"),
        ("regression_over_10pp", "prior_success_regression"),
        ("fallback_used", "fallback_disabled"),
        ("invalid_action", "all_actions_finite_and_bounded"),
        ("privileged_leakage", "no_privileged_state_leakage"),
        ("premature_stationary", "no_premature_stationary_success"),
    ),
)
def test_acceptance_fails_closed_on_formal_policy_violations(
    mutation: str,
    failed_check: str,
) -> None:
    metrics = _formal_metrics()
    if mutation == "too_few_episodes":
        metrics["standard"]["joint_wam_direct"]["episodes"] = 499
        metrics["standard"]["joint_wam_direct"]["seeds"] = list(
            range(120_000, 120_499)
        )
    elif mutation == "success_below_90":
        metrics["standard"]["joint_wam_direct"]["success_rate"] = 0.898
    elif mutation == "regression_over_10pp":
        metrics["standard"]["joint_wam_direct"]["success_rate"] = 0.899
        metrics["standard"]["action_prior"]["success_rate"] = 1.0
    elif mutation == "fallback_used":
        metrics["standard"]["joint_wam_direct"]["fallback_trigger_rate"] = (
            1.0 / 500.0
        )
        metrics["standard"]["joint_wam_direct"]["planner_modes"] = {
            "joint_wam_direct": 4_999,
            "action_prior_risk_fallback": 1,
        }
    elif mutation == "invalid_action":
        metrics["standard"]["joint_wam_direct"][
            "all_actions_finite_and_bounded"
        ] = False
    elif mutation == "privileged_leakage":
        metrics["standard"]["joint_wam_direct"][
            "privileged_state_leakage"
        ] = True
    elif mutation == "premature_stationary":
        metrics["standard"]["joint_wam_direct"][
            "premature_stationary_successes"
        ] = 1
    else:  # pragma: no cover - keeps additions to the parameter table honest.
        raise AssertionError(mutation)

    report = _acceptance_report(metrics, records_by_suite=_formal_records())

    assert not report["passed"]
    assert report["policy_acceptable"] is False
    assert not report["suites"]["standard"]["checks"][failed_check]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("different_seeds", "paired_seed_sets_per_suite"),
        ("duplicate_seed", "unique_seeds_per_suite_policy"),
        ("wrong_policy_label", "episode_record_labels_match_policy"),
    ),
)
def test_formal_schema_rejects_unpaired_duplicate_or_mislabeled_records(
    mutation: str,
    failed_check: str,
) -> None:
    records = _formal_records()
    prior = records["standard"]["action_prior"]
    if mutation == "different_seeds":
        prior[-1] = replace(prior[-1], seed=999_999)
    elif mutation == "duplicate_seed":
        prior[-1] = replace(prior[-1], seed=prior[0].seed)
    elif mutation == "wrong_policy_label":
        prior[-1] = replace(prior[-1], policy="stationary")
    else:  # pragma: no cover
        raise AssertionError(mutation)

    report = _acceptance_report(
        _formal_metrics(),
        records_by_suite=records,
    )

    assert not report["passed"]
    assert not report["schema"]["checks"][failed_check]


def test_formal_schema_rejects_metrics_that_disagree_with_raw_records() -> None:
    metrics = _formal_metrics()
    metrics["standard"]["joint_wam_direct"]["success_rate"] = 0.90

    report = _acceptance_report(metrics, records_by_suite=_formal_records())

    assert not report["passed"]
    assert not report["schema"]["checks"][
        "aggregate_metrics_match_episode_records"
    ]


def test_direct_coverage_requires_one_action_source_per_rollout_step() -> None:
    metrics = _formal_metrics()
    direct = metrics["standard"]["joint_wam_direct"]
    direct["planner_modes"] = {"joint_wam_direct": direct["total_steps"] - 1}

    report = _acceptance_report(metrics, records_by_suite=_formal_records())

    assert not report["passed"]
    assert report["suites"]["standard"]["direct_execution_rate"] == 0.0
    assert not report["suites"]["standard"]["checks"][
        "direct_execution_rate"
    ]


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("held_out_seed_overlap", 1),
        ("strict_joint_evidence", False),
        ("source_checkpoints_immutable", False),
        ("strict_reload_max_abs_diff", 1e-7),
        ("required_videos_complete", False),
        ("formal_protocol", False),
    ),
)
def test_acceptance_requires_strict_joint_and_formal_evidence(
    override: str,
    value: object,
) -> None:
    arguments = _acceptance_arguments()
    arguments[override] = value

    report = joint_wam_acceptance_report(
        _formal_metrics(),
        records_by_suite=_formal_records(),
        **arguments,
    )

    assert not report["passed"]
    assert report["policy_acceptable"] is False
    assert not report["checks"][
        {
            "held_out_seed_overlap": "held_out_training_seed_overlap_zero",
            "strict_joint_evidence": "strict_joint_evidence",
            "source_checkpoints_immutable": "source_checkpoints_immutable",
            "strict_reload_max_abs_diff": "strict_checkpoint_reload_exact",
            "required_videos_complete": "required_videos_complete",
            "formal_protocol": "formal_protocol",
        }[override]
    ]


def test_strict_joint_evidence_requires_every_coupling_path() -> None:
    required = {
        name: True
        for name in (
            "world_model_parameter_delta_nonzero",
            "shared_history_parameter_delta_nonzero",
            "world_parameter_delta_nonzero",
            "action_flow_parameter_delta_nonzero",
            "action_to_flow_gradient_nonzero",
            "action_to_backbone_gradient_nonzero",
            "world_to_backbone_gradient_nonzero",
            "consistency_to_flow_gradient_nonzero",
            "consistency_to_backbone_gradient_nonzero",
            "anchor_prior_immutable",
            "frozen_teacher_immutable",
            "source_checkpoints_immutable",
            "strict_checkpoint_reload_exact",
            "formal_run",
            "offline_audit_passed",
        )
    }
    arguments = _acceptance_arguments()
    arguments["strict_joint_evidence"] = {"passed": True, "checks": required}
    passed = joint_wam_acceptance_report(
        _formal_metrics(),
        records_by_suite=_formal_records(),
        **arguments,
    )
    assert passed["passed"]

    required["consistency_to_backbone_gradient_nonzero"] = False
    failed = joint_wam_acceptance_report(
        _formal_metrics(),
        records_by_suite=_formal_records(),
        **arguments,
    )
    assert not failed["passed"]
    assert not failed["checks"]["strict_joint_evidence"]


def test_video_validation_rejects_empty_selection_and_evidence() -> None:
    report = validate_video_evidence(
        {
            "format_version": "wam.joint_wam.video_selection/1",
            "policy": "joint_wam_direct",
            "selected": (),
        },
        (),
    )

    assert not report["passed"]
    assert not report["checks"]["selection_schema_valid"]


def test_video_selection_uses_smallest_success_and_failure_seeds() -> None:
    records = {
        "standard": [
            _episode(seed, success=success)
            for seed, success in (
                (120_009, True),
                (120_004, False),
                (120_001, True),
                (120_006, True),
                (120_003, False),
                (120_002, True),
            )
        ],
        "challenge": [
            _episode(seed, success=success)
            for seed, success in (
                (220_008, False),
                (220_004, True),
                (220_001, False),
                (220_003, True),
                (220_002, False),
                (220_005, True),
            )
        ],
    }

    selection = select_video_episodes(
        records,
        success_per_suite=3,
        failure_global_max=3,
    )

    assert [(item.suite, item.seed) for item in selection["success"]] == [
        ("standard", 120_001),
        ("standard", 120_002),
        ("standard", 120_006),
        ("challenge", 220_003),
        ("challenge", 220_004),
        ("challenge", 220_005),
    ]
    assert [(item.suite, item.seed) for item in selection["failure"]] == [
        ("standard", 120_003),
        ("standard", 120_004),
        ("challenge", 220_001),
    ]
    assert all(item.policy == "joint_wam_direct" for item in selection["selected"])


def test_video_selection_keeps_all_failures_when_fewer_than_three() -> None:
    records = {
        "standard": [_episode(seed, success=True) for seed in range(120_000, 120_003)],
        "challenge": [
            _episode(220_000, success=False),
            _episode(220_001, success=True),
            _episode(220_002, success=True),
            _episode(220_003, success=True),
        ],
    }

    selection = select_video_episodes(
        records,
        success_per_suite=3,
        failure_global_max=3,
    )

    assert [(item.suite, item.seed) for item in selection["failure"]] == [
        ("challenge", 220_000)
    ]


def test_output_directory_rejects_stale_evidence(tmp_path: Path) -> None:
    output = tmp_path / "gate"
    output.mkdir()
    (output / "stale.mp4").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="stale evidence"):
        _prepare_output_directory(output)


def test_training_seed_manifest_is_required_and_cross_checked() -> None:
    partitions = {
        "train": [
            "/dataset/episode_000000.hdf5",
            "/dataset/episode_000001.hdf5",
        ],
        "validation": ["/dataset/episode_000002.hdf5"],
        "test": ["/dataset/episode_000003.hdf5"],
    }
    parent_evidence = {
        "world_model_manifest": {"partitions": partitions},
        "action_flow_metrics": {
            "on_policy_distillation": {"seeds": [300_000, 300_001]}
        },
        "action_prior_manifest": {
            "partitions": partitions,
            "smoke_subset": False,
        },
    }
    joint_manifest = {
        "partitions": partitions,
        "partition_seeds": {
            "train": [0, 1],
            "validation": [2],
            "test": [3],
        },
        "action_flow_on_policy_seeds": [300_000, 300_001],
        "generated_or_relabel_seeds": [],
        "smoke_subset": False,
    }

    assert _training_seeds(
        joint_manifest,
        **parent_evidence,
    ) == {0, 1, 2, 3, 300_000, 300_001}

    for missing in (
        "partitions",
        "partition_seeds",
        "action_flow_on_policy_seeds",
        "generated_or_relabel_seeds",
        "smoke_subset",
    ):
        malformed = dict(joint_manifest)
        malformed.pop(missing)
        with pytest.raises(ValueError):
            _training_seeds(
                malformed,
                **parent_evidence,
            )

    mismatched = dict(joint_manifest)
    mismatched["action_flow_on_policy_seeds"] = [300_000]
    with pytest.raises(ValueError, match="does not match"):
        _training_seeds(
            mismatched,
            **parent_evidence,
        )

    truncated = {
        **joint_manifest,
        "partition_seeds": {
            **joint_manifest["partition_seeds"],
            "train": [0],
        },
    }
    with pytest.raises(ValueError, match="do not match its paths"):
        _training_seeds(truncated, **parent_evidence)

    contaminated_prior_partitions = {
        name: list(paths) for name, paths in partitions.items()
    }
    contaminated_prior_partitions["train"].append(
        "/dataset/episode_120000.hdf5"
    )
    contaminated_evidence = {
        **parent_evidence,
        "action_prior_manifest": {
            "partitions": contaminated_prior_partitions,
            "smoke_subset": False,
        },
    }
    with pytest.raises(ValueError, match="action prior"):
        _training_seeds(joint_manifest, **contaminated_evidence)

    mismatched_metrics = {
        **parent_evidence,
        "action_flow_metrics": {
            "on_policy_distillation": {"seeds": [300_000]}
        },
    }
    with pytest.raises(ValueError, match="embedded warm-up metrics"):
        _training_seeds(joint_manifest, **mismatched_metrics)


def _acceptance_report(
    metrics: dict[str, dict[str, dict[str, object]]],
    *,
    records_by_suite: dict[str, dict[str, list[ClosedLoopEpisode]]] | None = None,
) -> dict[str, object]:
    return joint_wam_acceptance_report(
        metrics,
        records_by_suite=records_by_suite,
        **_acceptance_arguments(),
    )


def _acceptance_arguments() -> dict[str, object]:
    return {
        "held_out_seed_overlap": 0,
        "minimum_episodes": 500,
        "minimum_success_rate": 0.90,
        "maximum_prior_regression": 0.10,
        "strict_joint_evidence": _strict_joint_evidence(),
        "source_checkpoints_immutable": True,
        "strict_reload_max_abs_diff": 0.0,
        "required_videos_complete": True,
        "formal_protocol": True,
    }


def _formal_metrics(
    *, flow_success_rate: float = 0.95,
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for suite, seed_start in (("standard", 120_000), ("challenge", 220_000)):
        seeds = list(range(seed_start, seed_start + 500))
        result[suite] = {
            "joint_wam_direct": _closed_loop_metrics(
                seeds=seeds,
                success_rate=flow_success_rate,
                mode="joint_wam_direct",
                residual=True,
            ),
            "action_prior": _closed_loop_metrics(
                seeds=seeds,
                success_rate=1.0,
                mode="action_prior",
            ),
            "stationary": _closed_loop_metrics(
                seeds=seeds,
                success_rate=0.0,
                mode="stationary",
            ),
            "scripted_oracle": _closed_loop_metrics(
                seeds=seeds,
                success_rate=1.0,
                mode="scripted_oracle",
            ),
            "joint_wam_with_fallback": _closed_loop_metrics(
                seeds=seeds,
                success_rate=1.0,
                mode="joint_wam_with_fallback",
                residual=True,
            ),
        }
    return result


def _formal_records(
    *,
    flow_success_rate: float = 0.95,
    fallback_success_rate: float = 1.0,
) -> dict[str, dict[str, list[ClosedLoopEpisode]]]:
    result: dict[str, dict[str, list[ClosedLoopEpisode]]] = {}
    for suite, seed_start in (("standard", 120_000), ("challenge", 220_000)):
        seeds = list(range(seed_start, seed_start + 500))
        direct_successes = round(flow_success_rate * len(seeds))
        fallback_successes = round(fallback_success_rate * len(seeds))
        result[suite] = {}
        for policy in (*POLICIES, "joint_wam_with_fallback"):
            result[suite][policy] = [
                _episode(
                    seed,
                    success=(
                        index < direct_successes
                        if policy == "joint_wam_direct"
                        else (
                            index < fallback_successes
                            if policy == "joint_wam_with_fallback"
                            else policy != "stationary"
                        )
                    ),
                    policy=policy,
                )
                for index, seed in enumerate(seeds)
            ]
    return result


def _closed_loop_metrics(
    *,
    seeds: list[int],
    success_rate: float,
    mode: str,
    residual: bool = False,
) -> dict[str, object]:
    total_steps = len(seeds) * 100
    return {
        "episodes": len(seeds),
        "seeds": list(seeds),
        "success_rate": success_rate,
        "all_actions_finite_and_bounded": True,
        "privileged_state_leakage": False,
        "planner_modes": {mode: total_steps},
        "total_steps": total_steps,
        "action_source_diagnostic_steps": total_steps,
        "action_source_coverage": 1.0,
        "applied_flow_residual": {
            "samples": total_steps if residual else 0,
            "mean": 0.01 if residual else None,
            "max": 0.01 if residual else None,
        },
        "fallback_trigger_rate": 0.0,
        "premature_stationary_successes": 0,
    }


def _episode(
    seed: int,
    *,
    success: bool,
    policy: str = "joint_wam_direct",
) -> ClosedLoopEpisode:
    return ClosedLoopEpisode(
        policy=policy,
        seed=seed,
        steps=100,
        success=success,
        failure=not success,
        failure_reason="none" if success else "timeout",
        total_reward=1.0 if success else -1.0,
        response_delay_seconds=0.2,
        mean_coordination_error=0.01,
        gradual_brake_steps=10,
        stop_hold_steps=8,
        pre_brake_motion_valid=True,
        planner_latency_ms=(10.0,),
        planner_modes=(policy,) * 100,
        planner_attempted_modes=(policy,) * 100,
        deadline_misses=0,
        discarded_plans=0,
        fallback_reasons=(),
        predicted_returns=(1.0,),
        applied_flow_residuals=(
            (0.01,) * 100
            if policy in {"joint_wam_direct", "joint_wam_with_fallback"}
            else ()
        ),
        observation_residual_nrmse=(0.01,),
        predicted_robot_distances=(0.5,),
        actual_robot_distances=(0.5,),
        actions_finite_and_bounded=True,
        privileged_state_seen=False,
    )


def _strict_joint_evidence() -> dict[str, object]:
    return {
        "passed": True,
        "checks": {
            name: True
            for name in (
                "world_model_parameter_delta_nonzero",
                "shared_history_parameter_delta_nonzero",
                "world_parameter_delta_nonzero",
                "action_flow_parameter_delta_nonzero",
                "action_to_flow_gradient_nonzero",
                "action_to_backbone_gradient_nonzero",
                "world_to_backbone_gradient_nonzero",
                "consistency_to_flow_gradient_nonzero",
                "consistency_to_backbone_gradient_nonzero",
                "anchor_prior_immutable",
                "frozen_teacher_immutable",
                "source_checkpoints_immutable",
                "strict_checkpoint_reload_exact",
                "formal_run",
                "offline_audit_passed",
            )
        },
    }
