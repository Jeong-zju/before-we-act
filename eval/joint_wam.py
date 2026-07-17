"""Pure acceptance and evidence helpers for final Joint WAM validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence


FORMAT_VERSION = "wam.joint_wam.acceptance/1"
VIDEO_SELECTION_FORMAT_VERSION = "wam.joint_wam.video_selection/1"
VIDEO_SIDECAR_FORMAT_VERSION = "wam.joint_wam.video/1"

SUITES = ("standard", "challenge")
DIRECT_POLICY = "joint_wam_direct"
ACTION_PRIOR_POLICY = "action_prior"
STATIONARY_POLICY = "stationary"
SCRIPTED_ORACLE_POLICY = "scripted_oracle"
FALLBACK_POLICY = "joint_wam_with_fallback"
REQUIRED_POLICIES = (
    DIRECT_POLICY,
    ACTION_PRIOR_POLICY,
    STATIONARY_POLICY,
    SCRIPTED_ORACLE_POLICY,
)
DIRECT_ACTION_MODES = (DIRECT_POLICY, "joint_wam_flow")


class EpisodeRecord(Protocol):
    """Structural type shared by ``ClosedLoopEpisode`` and test doubles."""

    policy: str
    seed: int
    success: bool
    failure_reason: str


EpisodeLike = EpisodeRecord | Mapping[str, Any]


@dataclass(frozen=True, order=True)
class VideoEpisodeSelection:
    """One deterministic direct-policy episode selected for video replay."""

    seed: int
    suite: str
    success: bool
    failure_reason: str
    policy: str = DIRECT_POLICY

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "seed": self.seed,
            "policy": self.policy,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "fallback_enabled": False,
        }


def select_video_episodes(
    records_by_suite: Mapping[str, Any],
    *,
    success_per_suite: int = 3,
    failure_global_max: int = 3,
) -> dict[str, Any]:
    """Select videos without cherry-picking.

    Successes are selected independently in each required suite, in ascending
    seed order.  This is stronger than the project-wide minimum of three and
    ensures that both environments receive qualitative evidence.  Failures are
    selected globally, capped at ``failure_global_max``, in ascending seed order
    with the fixed suite order as a deterministic tie-breaker.

    ``records_by_suite`` may map each suite directly to direct-policy episodes,
    or use the rollout shape ``suite -> policy -> episodes``.  Duplicate seeds,
    missing suites, non-direct records, or too few successful episodes fail
    closed with ``ValueError``.
    """

    if success_per_suite <= 0:
        raise ValueError("success_per_suite must be positive")
    if failure_global_max < 0:
        raise ValueError("failure_global_max must be non-negative")
    if set(records_by_suite) != set(SUITES):
        raise ValueError("video selection requires exactly standard and challenge")

    suite_rank = {name: index for index, name in enumerate(SUITES)}
    successes: list[VideoEpisodeSelection] = []
    failures: list[VideoEpisodeSelection] = []
    available: dict[str, dict[str, int]] = {}

    for suite in SUITES:
        raw_records = _direct_records(records_by_suite[suite])
        selections = [_video_selection(suite, record) for record in raw_records]
        seeds = [item.seed for item in selections]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{suite} direct video candidates contain duplicate seeds")
        successful = sorted(
            (item for item in selections if item.success), key=_seed_key
        )
        failed = sorted(
            (item for item in selections if not item.success), key=_seed_key
        )
        if len(successful) < success_per_suite:
            raise ValueError(
                f"{suite} has {len(successful)} successful direct episodes; "
                f"{success_per_suite} are required"
            )
        successes.extend(successful[:success_per_suite])
        failures.extend(failed)
        available[suite] = {
            "success": len(successful),
            "failure": len(failed),
        }

    failures.sort(key=lambda item: (item.seed, suite_rank[item.suite]))
    selected_failures = failures[:failure_global_max]
    return {
        "format_version": VIDEO_SELECTION_FORMAT_VERSION,
        "policy": DIRECT_POLICY,
        "selection_rule": {
            "success": "lowest_seed_per_suite",
            "failure": "lowest_seed_global_then_fixed_suite_order",
            "success_per_suite": success_per_suite,
            "failure_global_max": failure_global_max,
        },
        "available": available,
        "success": tuple(successes),
        "failure": tuple(selected_failures),
        "selected": tuple(successes) + tuple(selected_failures),
    }


def select_joint_wam_video_seeds(
    records_by_suite: Mapping[str, Any],
    *,
    success_per_suite: int = 3,
    failure_global_max: int = 3,
) -> dict[str, Any]:
    """Explicitly named alias for :func:`select_video_episodes`."""

    return select_video_episodes(
        records_by_suite,
        success_per_suite=success_per_suite,
        failure_global_max=failure_global_max,
    )


def validate_video_evidence(
    selection: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate encoded-video sidecars against deterministic selections.

    The caller must already have encoded and hashed each video.  This function
    only validates the supplied schema, exact selected episode set, direct
    action-source evidence, disabled fallback, and replay/evaluation parity.
    """

    expected_items = tuple(selection.get("selected", ()))
    expected: dict[tuple[str, int], VideoEpisodeSelection] = {}
    rule = selection.get("selection_rule")
    selection_schema_valid = bool(
        selection.get("format_version") == VIDEO_SELECTION_FORMAT_VERSION
        and selection.get("policy") == DIRECT_POLICY
        and isinstance(rule, Mapping)
        and rule.get("success") == "lowest_seed_per_suite"
        and rule.get("failure")
        == "lowest_seed_global_then_fixed_suite_order"
        and _positive_integer(rule.get("success_per_suite"))
        and _non_negative_integer(rule.get("failure_global_max"))
        and expected_items
    )
    for raw in expected_items:
        try:
            item = _coerce_selection(raw)
        except (KeyError, TypeError, ValueError):
            selection_schema_valid = False
            continue
        key = (item.suite, item.seed)
        if key in expected:
            selection_schema_valid = False
        expected[key] = item

    if isinstance(rule, Mapping):
        success_per_suite = int(rule.get("success_per_suite", -1))
        failure_global_max = int(rule.get("failure_global_max", -1))
        for suite in SUITES:
            selection_schema_valid = selection_schema_valid and sum(
                item.success and item.suite == suite for item in expected.values()
            ) == success_per_suite
        selected_failures = sum(not item.success for item in expected.values())
        available = selection.get("available")
        if isinstance(available, Mapping) and all(
            isinstance(available.get(suite), Mapping) for suite in SUITES
        ):
            available_failures = sum(
                int(available[suite].get("failure", -1)) for suite in SUITES
            )
            selection_schema_valid = selection_schema_valid and (
                selected_failures == min(available_failures, failure_global_max)
            )
        else:
            selection_schema_valid = False
        selection_schema_valid = (
            selection_schema_valid and selected_failures <= failure_global_max
        )

    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    sidecar_schema_valid = True
    direct_sources_valid = True
    fallback_disabled = True
    replay_matches = True
    video_artifacts_valid = True
    outcomes_match = True
    for sidecar in evidence:
        try:
            suite = str(sidecar["suite"])
            seed = int(sidecar["seed"])
            success = sidecar["success"]
            failure = sidecar["failure"]
            failure_reason = sidecar["failure_reason"]
        except (KeyError, TypeError, ValueError):
            sidecar_schema_valid = False
            continue
        key = (suite, seed)
        if (
            sidecar.get("format_version") != VIDEO_SIDECAR_FORMAT_VERSION
            or suite not in SUITES
            or type(success) is not bool
            or type(failure) is not bool
            or failure is success
            or not isinstance(failure_reason, str)
            or sidecar.get("policy") != DIRECT_POLICY
            or key in observed
        ):
            sidecar_schema_valid = False
        observed[key] = sidecar

        fallback_disabled = (
            fallback_disabled and sidecar.get("fallback_enabled") is False
        )
        replay_matches = (
            replay_matches and sidecar.get("replay_matches_evaluation") is True
        )
        direct_sources_valid = direct_sources_valid and _direct_video_sources(sidecar)
        video_artifacts_valid = video_artifacts_valid and bool(
            _valid_sha256(sidecar.get("video_sha256"))
            and _positive_integer(sidecar.get("frames_written"))
            and _positive_integer(sidecar.get("steps"))
            and int(sidecar["frames_written"]) == int(sidecar["steps"])
        )
        selected = expected.get(key)
        outcomes_match = outcomes_match and bool(
            selected is not None
            and success is selected.success
            and failure_reason == selected.failure_reason
        )

    exact_episode_set = set(observed) == set(expected)
    checks = {
        "selection_schema_valid": selection_schema_valid,
        "sidecar_schema_valid": sidecar_schema_valid,
        "exact_selected_episode_set": exact_episode_set,
        "outcomes_match_evaluation": outcomes_match and exact_episode_set,
        "direct_action_sources_recorded": direct_sources_valid,
        "fallback_disabled": fallback_disabled,
        "replay_matches_evaluation": replay_matches,
        "video_artifacts_nonempty_and_hashed": video_artifacts_valid,
    }
    return {
        "format_version": VIDEO_SIDECAR_FORMAT_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_videos": len(expected),
        "observed_videos": len(observed),
    }


def validate_joint_wam_schema(
    metrics: Mapping[str, Any],
    records_by_suite: Mapping[str, Any] | None = None,
    *,
    minimum_episodes: int = 500,
) -> dict[str, Any]:
    """Validate the two-suite/four-policy formal evaluation shape.

    When episode records are supplied, this also proves exact paired seed sets
    across all required policies and rejects duplicates or mislabeled records.
    """

    suite_coverage = set(metrics) == set(SUITES)
    policy_coverage = suite_coverage and all(
        isinstance(metrics.get(suite), Mapping)
        and all(policy in metrics[suite] for policy in REQUIRED_POLICIES)
        for suite in SUITES
    )
    metric_episode_counts = policy_coverage and all(
        _integer_at_least(metrics[suite][policy].get("episodes"), minimum_episodes)
        for suite in SUITES
        for policy in REQUIRED_POLICIES
        if isinstance(metrics[suite][policy], Mapping)
    )

    records_available = records_by_suite is not None
    paired_seed_sets = True
    unique_seeds = True
    record_labels = True
    record_episode_counts = True
    aggregate_metrics_match_records = True
    if records_by_suite is not None:
        if set(records_by_suite) != set(SUITES):
            paired_seed_sets = unique_seeds = record_labels = record_episode_counts = (
                False
            )
            aggregate_metrics_match_records = False
        else:
            for suite in SUITES:
                policies = records_by_suite[suite]
                if not isinstance(policies, Mapping) or any(
                    policy not in policies for policy in REQUIRED_POLICIES
                ):
                    paired_seed_sets = unique_seeds = False
                    record_labels = record_episode_counts = False
                    aggregate_metrics_match_records = False
                    continue
                seed_sets: list[set[int]] = []
                for policy in REQUIRED_POLICIES:
                    records = tuple(policies[policy])
                    seeds: list[int] = []
                    for record in records:
                        try:
                            seed = _strict_seed(_record_value(record, "seed"))
                            label = str(_record_value(record, "policy"))
                        except (AttributeError, KeyError, TypeError, ValueError):
                            unique_seeds = False
                            record_labels = False
                            continue
                        seeds.append(seed)
                        record_labels = record_labels and label == policy
                    unique_seeds = unique_seeds and len(seeds) == len(set(seeds))
                    record_episode_counts = (
                        record_episode_counts and len(seeds) >= minimum_episodes
                    )
                    seed_sets.append(set(seeds))
                    suite_metrics = metrics.get(suite)
                    policy_metrics = (
                        suite_metrics.get(policy)
                        if isinstance(suite_metrics, Mapping)
                        else None
                    )
                    aggregate_metrics_match_records = bool(
                        aggregate_metrics_match_records
                        and isinstance(policy_metrics, Mapping)
                        and _aggregate_matches_records(policy_metrics, records)
                    )
                paired_seed_sets = (
                    paired_seed_sets
                    and bool(seed_sets)
                    and all(values == seed_sets[0] for values in seed_sets[1:])
                )

    checks = {
        "required_suite_coverage": suite_coverage,
        "required_policy_coverage": policy_coverage,
        "metric_minimum_episode_counts": metric_episode_counts,
        "episode_records_available": records_available,
        "episode_record_minimum_counts": records_available and record_episode_counts,
        "episode_record_labels_match_policy": records_available and record_labels,
        "unique_seeds_per_suite_policy": records_available and unique_seeds,
        "paired_seed_sets_per_suite": records_available and paired_seed_sets,
        "aggregate_metrics_match_episode_records": records_available
        and aggregate_metrics_match_records,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "required_suites": list(SUITES),
        "required_policies": list(REQUIRED_POLICIES),
        "minimum_episodes": minimum_episodes,
    }


def joint_wam_acceptance_report(
    metrics: Mapping[str, Any],
    *,
    records_by_suite: Mapping[str, Any] | None = None,
    held_out_seed_overlap: int,
    strict_joint_evidence: bool | Mapping[str, Any],
    source_checkpoints_immutable: bool,
    strict_reload_max_abs_diff: float | int | None,
    required_videos_complete: bool,
    minimum_episodes: int = 500,
    minimum_success_rate: float = 0.90,
    maximum_prior_regression: float = 0.10,
    formal_protocol: bool = True,
) -> dict[str, Any]:
    """Apply the Joint WAM policy-acceptance contract.

    Only ``joint_wam_direct`` is gated.  ``joint_wam_with_fallback`` is copied
    into the report when present, but its success rate can neither rescue nor
    fail the direct policy.  ``joint_benefit`` remains explicitly unevaluated.
    """

    if minimum_episodes <= 0:
        raise ValueError("minimum_episodes must be positive")
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in [0,1]")
    if not 0.0 <= maximum_prior_regression <= 1.0:
        raise ValueError("maximum_prior_regression must be in [0,1]")
    if held_out_seed_overlap < 0:
        raise ValueError("held_out_seed_overlap must be non-negative")

    schema = validate_joint_wam_schema(
        metrics,
        records_by_suite,
        minimum_episodes=minimum_episodes,
    )
    suites: dict[str, Any] = {}
    suites_passed = True
    for suite in SUITES:
        policies = metrics.get(suite, {})
        direct = policies.get(DIRECT_POLICY) if isinstance(policies, Mapping) else None
        prior = (
            policies.get(ACTION_PRIOR_POLICY) if isinstance(policies, Mapping) else None
        )
        stationary = (
            policies.get(STATIONARY_POLICY) if isinstance(policies, Mapping) else None
        )
        oracle = (
            policies.get(SCRIPTED_ORACLE_POLICY)
            if isinstance(policies, Mapping)
            else None
        )
        if not all(
            isinstance(item, Mapping) for item in (direct, prior, stationary, oracle)
        ):
            suites[suite] = {
                "passed": False,
                "checks": {"required_policies_present": False},
            }
            suites_passed = False
            continue
        assert isinstance(direct, Mapping)
        assert isinstance(prior, Mapping)
        assert isinstance(stationary, Mapping)
        assert isinstance(oracle, Mapping)
        direct_success = _finite_float(direct.get("success_rate"))
        prior_success = _finite_float(prior.get("success_rate"))
        prior_regression = (
            prior_success - direct_success
            if direct_success is not None and prior_success is not None
            else None
        )
        direct_rate = _direct_execution_rate(direct)
        residual = direct.get("applied_flow_residual", {})
        direct_steps = direct.get("total_steps")
        residual_applied = bool(
            isinstance(residual, Mapping)
            and _positive_integer(direct_steps)
            and residual.get("samples") == direct_steps
            and (_finite_float(residual.get("max")) or 0.0) > 1e-8
        )
        checks = {
            "required_policies_present": True,
            "minimum_episodes": all(
                _integer_at_least(item.get("episodes"), minimum_episodes)
                for item in (direct, prior, stationary, oracle)
            ),
            "minimum_direct_success_rate": bool(
                direct_success is not None
                and direct_success + 1e-12 >= minimum_success_rate
            ),
            "prior_success_regression": bool(
                prior_regression is not None
                and prior_regression <= maximum_prior_regression + 1e-12
            ),
            "all_actions_finite_and_bounded": direct.get(
                "all_actions_finite_and_bounded"
            )
            is True,
            "no_privileged_state_leakage": direct.get("privileged_state_leakage")
            is False,
            "direct_execution_rate": direct_rate >= 1.0 - 1e-12,
            "flow_residual_applied": residual_applied,
            "fallback_disabled": _exact_zero(direct.get("fallback_trigger_rate")),
            "no_premature_stationary_success": _exact_zero(
                direct.get("premature_stationary_successes")
            ),
            "stationary_baseline_does_not_succeed": _exact_zero(
                stationary.get("success_rate")
            ),
        }
        passed = all(checks.values())
        suites_passed = suites_passed and passed
        suites[suite] = {
            "passed": passed,
            "checks": checks,
            "direct_success_rate": direct_success,
            "prior_success_rate": prior_success,
            "prior_regression": prior_regression,
            "direct_execution_rate": direct_rate,
        }

    top_checks = {
        "formal_protocol": formal_protocol is True,
        "formal_schema": schema["passed"]
        if records_by_suite is not None
        else bool(
            schema["checks"]["required_suite_coverage"]
            and schema["checks"]["required_policy_coverage"]
            and schema["checks"]["metric_minimum_episode_counts"]
        ),
        "held_out_training_seed_overlap_zero": held_out_seed_overlap == 0,
        "strict_joint_evidence": _strict_joint_evidence_ok(strict_joint_evidence),
        "source_checkpoints_immutable": source_checkpoints_immutable is True,
        "strict_checkpoint_reload_exact": _exact_zero(strict_reload_max_abs_diff),
        "required_videos_complete": required_videos_complete is True,
        "fallback_deployment_report_complete": _fallback_report_complete(
            metrics,
            records_by_suite,
            minimum_episodes=minimum_episodes,
        ),
        "all_suites_passed": suites_passed,
    }
    policy_acceptable = all(top_checks.values())
    fallback_metrics = {
        suite: metrics.get(suite, {}).get(FALLBACK_POLICY)
        for suite in SUITES
        if isinstance(metrics.get(suite), Mapping) and FALLBACK_POLICY in metrics[suite]
    }
    return {
        "format_version": FORMAT_VERSION,
        "model": "joint_wam",
        "protocol": "paired_500_seed",
        "passed": policy_acceptable,
        "policy_acceptable": policy_acceptable,
        "checks": top_checks,
        "suites": suites,
        "schema": schema,
        "thresholds": {
            "minimum_episodes_per_suite_policy": minimum_episodes,
            "minimum_direct_success_rate": minimum_success_rate,
            "maximum_prior_success_regression": maximum_prior_regression,
            "direct_execution_rate": 1.0,
            "fallback_trigger_rate": 0.0,
        },
        "held_out_seed_overlap": held_out_seed_overlap,
        "report_only": {
            "fallback_policy": FALLBACK_POLICY,
            "fallback_present_in_all_suites": len(fallback_metrics) == len(SUITES),
            "fallback_metrics": fallback_metrics,
            "fallback_affects_policy_acceptance": False,
        },
        "joint_benefit": {
            "evaluated": False,
            "value": None,
            "claim_allowed": False,
            "affects_policy_acceptance": False,
        },
    }


def _fallback_report_complete(
    metrics: Mapping[str, Any],
    records_by_suite: Mapping[str, Any] | None,
    *,
    minimum_episodes: int,
) -> bool:
    for suite in SUITES:
        policies = metrics.get(suite)
        if not isinstance(policies, Mapping):
            return False
        fallback = policies.get(FALLBACK_POLICY)
        if not isinstance(fallback, Mapping) or not _integer_at_least(
            fallback.get("episodes"), minimum_episodes
        ):
            return False
        if records_by_suite is None:
            continue
        suite_records = records_by_suite.get(suite)
        if not isinstance(suite_records, Mapping):
            return False
        direct_records = tuple(suite_records.get(DIRECT_POLICY, ()))
        fallback_records = tuple(suite_records.get(FALLBACK_POLICY, ()))
        try:
            direct_seeds = {
                _strict_seed(_record_value(record, "seed"))
                for record in direct_records
            }
            fallback_seeds = [
                _strict_seed(_record_value(record, "seed"))
                for record in fallback_records
            ]
            fallback_labels = all(
                str(_record_value(record, "policy")) == FALLBACK_POLICY
                for record in fallback_records
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        if (
            len(fallback_seeds) < minimum_episodes
            or len(fallback_seeds) != len(set(fallback_seeds))
            or set(fallback_seeds) != direct_seeds
            or not fallback_labels
            or not _aggregate_matches_records(fallback, fallback_records)
        ):
            return False
    return True


def _direct_records(value: Any) -> tuple[EpisodeLike, ...]:
    if isinstance(value, Mapping):
        if DIRECT_POLICY not in value:
            raise ValueError(f"video candidates are missing policy {DIRECT_POLICY!r}")
        value = value[DIRECT_POLICY]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("video candidates must be an episode sequence")
    return tuple(value)


def _video_selection(suite: str, record: EpisodeLike) -> VideoEpisodeSelection:
    policy = str(_record_value(record, "policy"))
    if policy != DIRECT_POLICY:
        raise ValueError(f"video evidence must come from {DIRECT_POLICY!r}")
    success = _record_value(record, "success")
    if type(success) is not bool:
        raise TypeError("episode success must be a bool")
    reason = str(_record_value(record, "failure_reason"))
    if success and reason not in {"none", ""}:
        raise ValueError("a successful episode may not carry a failure reason")
    if not success and reason in {"none", ""}:
        raise ValueError("a failed episode must carry a failure reason")
    return VideoEpisodeSelection(
        suite=suite,
        seed=_strict_seed(_record_value(record, "seed")),
        success=success,
        failure_reason=reason,
    )


def _coerce_selection(value: Any) -> VideoEpisodeSelection:
    if isinstance(value, VideoEpisodeSelection):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("video selection entries must be mappings or selections")
    result = VideoEpisodeSelection(
        suite=str(value["suite"]),
        seed=_strict_seed(value["seed"]),
        policy=str(value.get("policy", DIRECT_POLICY)),
        success=_strict_bool(value["success"]),
        failure_reason=str(value["failure_reason"]),
    )
    if result.suite not in SUITES or result.policy != DIRECT_POLICY:
        raise ValueError("video selection has an unsupported suite or policy")
    return result


def _record_value(record: EpisodeLike, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _seed_key(item: VideoEpisodeSelection) -> tuple[int, str]:
    return item.seed, item.suite


def _direct_video_sources(sidecar: Mapping[str, Any]) -> bool:
    sources: Any = sidecar.get("action_source_counts")
    if sources is None:
        sources = sidecar.get("planner_modes")
    if isinstance(sources, Mapping):
        counts: dict[str, int] = {}
        for name, value in sources.items():
            if not _non_negative_integer(value):
                return False
            counts[str(name)] = int(value)
        total = sum(counts.values())
        direct = sum(counts.get(name, 0) for name in DIRECT_ACTION_MODES)
        steps = sidecar.get("steps")
        return bool(
            _positive_integer(steps)
            and total == int(steps)
            and direct == total
        )
    source = sidecar.get("action_source")
    if isinstance(source, str):
        return source in DIRECT_ACTION_MODES
    sources = sidecar.get("action_sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        return bool(sources) and all(
            str(item) in DIRECT_ACTION_MODES for item in sources
        )
    return False


def _direct_execution_rate(metrics: Mapping[str, Any]) -> float:
    modes = metrics.get("planner_modes")
    if not isinstance(modes, Mapping):
        return 0.0
    total = 0
    direct = 0
    for name, raw_count in modes.items():
        if not _non_negative_integer(raw_count):
            return 0.0
        count = int(raw_count)
        total += count
        if str(name) in DIRECT_ACTION_MODES:
            direct += count
    rollout_steps = metrics.get("total_steps")
    if not _positive_integer(rollout_steps) or total != int(rollout_steps):
        return 0.0
    return direct / int(rollout_steps)


def _aggregate_matches_records(
    metrics: Mapping[str, Any], records: Sequence[EpisodeLike]
) -> bool:
    if not records:
        return False
    try:
        successes = [_record_value(record, "success") for record in records]
        if any(type(value) is not bool for value in successes):
            return False
        total_steps = sum(int(_record_value(record, "steps")) for record in records)
        modes = Counter(
            str(mode)
            for record in records
            for mode in _record_value(record, "planner_modes")
        )
        residuals = [
            float(value)
            for record in records
            for value in _record_value(record, "applied_flow_residuals")
        ]
        actions_valid = all(
            _record_value(record, "actions_finite_and_bounded") is True
            for record in records
        )
        privileged = any(
            _record_value(record, "privileged_state_seen") is True
            for record in records
        )
        premature = sum(
            bool(_record_value(record, "success"))
            and not bool(_record_value(record, "pre_brake_motion_valid"))
            for record in records
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if total_steps <= 0 or any(not math.isfinite(value) for value in residuals):
        return False
    residual_metrics = metrics.get("applied_flow_residual")
    if not isinstance(residual_metrics, Mapping):
        return False
    expected_max = max(residuals) if residuals else None
    reported_max = residual_metrics.get("max")
    expected_mean = sum(residuals) / len(residuals) if residuals else None
    reported_mean = residual_metrics.get("mean")
    residual_max_matches = (
        expected_max is None and reported_max is None
    ) or bool(
        expected_max is not None
        and _finite_float(reported_max) is not None
        and math.isclose(expected_max, float(reported_max), abs_tol=1e-12)
    )
    residual_mean_matches = (
        expected_mean is None and reported_mean is None
    ) or bool(
        expected_mean is not None
        and _finite_float(reported_mean) is not None
        and math.isclose(expected_mean, float(reported_mean), abs_tol=1e-12)
    )
    diagnostic_steps = sum(modes.values())
    return bool(
        metrics.get("episodes") == len(records)
        and _finite_float(metrics.get("success_rate")) is not None
        and math.isclose(
            float(metrics["success_rate"]),
            sum(successes) / len(successes),
            abs_tol=1e-12,
        )
        and metrics.get("total_steps") == total_steps
        and metrics.get("planner_modes") == dict(sorted(modes.items()))
        and metrics.get("action_source_diagnostic_steps") == diagnostic_steps
        and _finite_float(metrics.get("action_source_coverage")) is not None
        and math.isclose(
            float(metrics["action_source_coverage"]),
            diagnostic_steps / total_steps,
            abs_tol=1e-12,
        )
        and metrics.get("all_actions_finite_and_bounded") is actions_valid
        and metrics.get("privileged_state_leakage") is privileged
        and metrics.get("premature_stationary_successes") == premature
        and residual_metrics.get("samples") == len(residuals)
        and residual_max_matches
        and residual_mean_matches
    )


def _strict_joint_evidence_ok(value: bool | Mapping[str, Any]) -> bool:
    if type(value) is bool:
        return False
    checks = value.get("checks")
    if isinstance(checks, Mapping):
        required = (
            "member_0_parameter_delta_nonzero",
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
        return bool(
            value.get("passed") is True
            and all(checks.get(name) is True for name in required)
        )

    positive = (
        "member_0_parameter_delta",
        "shared_history_parameter_delta",
        "world_parameter_delta",
        "action_flow_parameter_delta",
    )
    gradients = value.get("branch_gradient_maxima")
    required_gradients = (
        "action_to_flow_gradient_norm",
        "action_to_backbone_gradient_norm",
        "world_to_backbone_gradient_norm",
        "consistency_to_flow_gradient_norm",
        "consistency_to_backbone_gradient_norm",
    )
    return bool(
        all(_positive_finite(value.get(name)) for name in positive)
        and isinstance(gradients, Mapping)
        and all(_positive_finite(gradients.get(name)) for name in required_gradients)
        and _exact_zero(value.get("anchor_prior_parameter_delta"))
        and _exact_zero(value.get("frozen_teacher_parameter_delta"))
        and value.get("source_checkpoints_immutable") is True
        and value.get("formal_run") is True
        and value.get("passed") is True
    )


def _strict_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("episode seeds must be non-negative integers")
    return value


def _strict_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError("value must be a bool")
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positive_finite(value: Any) -> bool:
    result = _finite_float(value)
    return result is not None and result > 0.0


def _exact_zero(value: Any) -> bool:
    result = _finite_float(value)
    return result is not None and result == 0.0


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _integer_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ACTION_PRIOR_POLICY",
    "DIRECT_POLICY",
    "FALLBACK_POLICY",
    "FORMAT_VERSION",
    "REQUIRED_POLICIES",
    "SCRIPTED_ORACLE_POLICY",
    "STATIONARY_POLICY",
    "SUITES",
    "VIDEO_SIDECAR_FORMAT_VERSION",
    "VIDEO_SELECTION_FORMAT_VERSION",
    "VideoEpisodeSelection",
    "joint_wam_acceptance_report",
    "select_joint_wam_video_seeds",
    "select_video_episodes",
    "validate_joint_wam_schema",
    "validate_video_evidence",
]
