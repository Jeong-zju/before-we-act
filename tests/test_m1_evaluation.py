from __future__ import annotations

import numpy as np
import pytest

from eval.m1_acceptance import PRIMARY_VARIANT, REQUIRED_VARIANTS
from eval.m1_statistics import (
    M1EpisodeRecord,
    aggregate_episode_records,
    exact_binomial_two_sided,
    exact_mcnemar,
    paired_balanced_accuracy_comparison,
    paired_bootstrap_difference,
    paired_episode_success,
    paired_rmse_comparison,
    wilson_interval,
)
from scripts.evaluate_multimodal_wam import (
    _evaluation_jobs,
    _expected_episode_count,
    _merge_evaluation_results,
    _warm_policy_runtime,
)


def _record(
    *,
    variant: str,
    success: bool,
    evaluation_seed: int,
    intervention: str = "clean",
) -> M1EpisodeRecord:
    return M1EpisodeRecord(
        task_id="visual_required",
        evaluation_seed=evaluation_seed,
        cue_id=0,
        model_variant=variant,
        train_seed=11,
        intervention=intervention,
        success=success,
        steps=20,
        total_reward=float(success),
        action_source=f"m1_{variant}",
    )


def test_wilson_interval_matches_known_value_and_handles_boundaries() -> None:
    result = wilson_interval(5, 10)
    assert result["rate"] == 0.5
    assert result["lower"] == pytest.approx(0.236593, abs=1e-6)
    assert result["upper"] == pytest.approx(0.763407, abs=1e-6)

    zero = wilson_interval(0, 10)
    full = wilson_interval(10, 10)
    assert zero["lower"] == 0.0
    assert zero["upper"] == pytest.approx(0.277533, abs=1e-6)
    assert full["lower"] == pytest.approx(0.722467, abs=1e-6)
    assert full["upper"] == 1.0


@pytest.mark.parametrize(
    ("successes", "total"),
    [(-1, 10), (11, 10), (0, 0)],
)
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_paired_bootstrap_is_deterministic_and_does_not_touch_global_rng() -> None:
    first = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.float64)
    second = np.asarray([0, 0, 0, 1, 1, 0], dtype=np.float64)
    np.random.seed(123)
    expected_global_draw = np.random.random()
    np.random.seed(123)
    left = paired_bootstrap_difference(
        first, second, bootstrap_samples=1_000, seed=91
    )
    actual_global_draw = np.random.random()
    right = paired_bootstrap_difference(
        first, second, bootstrap_samples=1_000, seed=91
    )
    different_seed = paired_bootstrap_difference(
        first, second, bootstrap_samples=1_000, seed=92
    )

    assert left == right
    assert left["mean_difference"] == pytest.approx(1 / 3)
    assert actual_global_draw == expected_global_draw
    assert left["bootstrap_seed"] == 91
    assert different_seed["bootstrap_seed"] == 92


def test_exact_binomial_and_mcnemar_need_no_scipy() -> None:
    assert exact_binomial_two_sided(10, 10) == pytest.approx(0.001953125)
    assert exact_binomial_two_sided(0, 10) == pytest.approx(0.001953125)
    assert exact_binomial_two_sided(5, 10) == 1.0
    assert exact_binomial_two_sided(0, 0) == 1.0

    result = exact_mcnemar([1] * 10, [0] * 10)
    assert result["first_only_success"] == 10
    assert result["second_only_success"] == 0
    assert result["p_value_two_sided"] == pytest.approx(0.001953125)
    symmetric = exact_mcnemar([0, 1, 1, 0], [1, 0, 1, 0])
    assert symmetric["discordant"] == 2
    assert symmetric["p_value_two_sided"] == 1.0


def test_aggregate_episode_records_reports_groups_duplicates_and_invalid_rows() -> None:
    records = [
        _record(variant="state_only", success=True, evaluation_seed=1),
        _record(variant="state_only", success=False, evaluation_seed=2),
    ]
    result = aggregate_episode_records(records)
    key = "state_only/clean/train-11/visual_required"
    assert result["passed"]
    assert result["groups"][key]["successes"] == 1
    assert result["groups"][key]["total"] == 2
    assert result["groups"][key]["rate"] == 0.5

    duplicate = aggregate_episode_records([records[0], records[0]])
    assert not duplicate["passed"]
    assert duplicate["duplicate_identities"]

    invalid = records[0].as_dict()
    invalid["steps"] = 0
    invalid_result = aggregate_episode_records([invalid])
    assert not invalid_result["passed"]
    assert invalid_result["invalid_record_count"] == 1


def test_paired_episode_success_requires_the_exact_rollout_identity_set() -> None:
    records = []
    for seed in range(40):
        records.extend(
            [
                _record(
                    variant="state_vision_future",
                    success=seed < 30,
                    evaluation_seed=seed,
                ),
                _record(
                    variant="state_only",
                    success=seed < 15,
                    evaluation_seed=seed,
                ),
            ]
        )
    result = paired_episode_success(
        records,
        first_variant="state_vision_future",
        first_intervention="clean",
        second_variant="state_only",
        second_intervention="clean",
        bootstrap_samples=500,
        seed=7,
    )
    assert result["exact_pairs"]
    assert result["paired_records"] == 40
    assert result["difference"]["mean_difference"] == pytest.approx(0.375)
    assert result["mcnemar"]["first_only_success"] == 15

    missing = paired_episode_success(
        records[:-1],
        first_variant="state_vision_future",
        first_intervention="clean",
        second_variant="state_only",
        second_intervention="clean",
        bootstrap_samples=100,
    )
    assert not missing["exact_pairs"]
    assert missing["missing_from_second"]


def test_episode_success_bootstrap_keeps_correlated_physical_seed_blocks() -> None:
    records: list[M1EpisodeRecord] = []
    for evaluation_seed in range(10):
        first_success = evaluation_seed < 6
        second_success = not first_success
        for train_seed in (11, 22, 33):
            for task_id in ("visual_event_stop", "visual_target_select"):
                for cue_id in (0, 1):
                    for variant, success in (
                        ("state_vision_future", first_success),
                        ("state_only", second_success),
                    ):
                        records.append(
                            M1EpisodeRecord(
                                task_id=task_id,
                                evaluation_seed=evaluation_seed,
                                cue_id=cue_id,
                                model_variant=variant,
                                train_seed=train_seed,
                                intervention="clean",
                                success=success,
                                steps=20,
                                total_reward=float(success),
                                action_source=f"m1_{variant}",
                            )
                        )

    result = paired_episode_success(
        records,
        first_variant="state_vision_future",
        first_intervention="clean",
        second_variant="state_only",
        second_intervention="clean",
        bootstrap_samples=5_000,
        seed=19,
    )
    difference = result["difference"]
    assert difference["mean_difference"] == pytest.approx(0.2)
    assert difference["cluster_bootstrap"] is True
    assert difference["cluster_key"] == "evaluation_seed"
    assert difference["clusters"] == 10
    assert difference["balanced_cluster_sizes"] is True
    assert set(difference["records_per_cluster"].values()) == {12}
    assert difference["ci_lower"] <= 0.0
    assert set(result["per_train_seed"]) == {"11", "22", "33"}
    assert all(
        comparison["difference"]["clusters"] == 10
        for comparison in result["per_train_seed"].values()
    )

    # Treating the 12 exact copies per physical seed as independent would give
    # a spuriously positive interval, which is the pseudoreplication regression.
    row_level = paired_bootstrap_difference(
        [1.0] * 72 + [0.0] * 48,
        [0.0] * 72 + [1.0] * 48,
        bootstrap_samples=5_000,
        seed=19,
    )
    assert row_level["ci_lower"] > 0.0


def test_future_probe_statistics_use_paired_rmse_and_balanced_accuracy() -> None:
    object_result = paired_rmse_comparison(
        [0.1] * 100,
        [0.3] * 100,
        bootstrap_samples=300,
        seed=4,
    )
    assert object_result["baseline_minus_model_rmse"] == pytest.approx(0.2)
    assert object_result["ci_lower"] > 0.0

    labels = np.tile([0, 1], 100)
    model = labels.copy()
    baseline = labels.copy()
    for pair in range(100):
        if pair % 10 == 0:
            model[2 * pair : 2 * pair + 2] = 1 - labels[
                2 * pair : 2 * pair + 2
            ]
        if pair % 10 < 4:
            baseline[2 * pair : 2 * pair + 2] = 1 - labels[
                2 * pair : 2 * pair + 2
            ]
    event_result = paired_balanced_accuracy_comparison(
        model,
        baseline,
        labels,
        bootstrap_samples=300,
        seed=8,
    )
    assert event_result["model_balanced_accuracy"] == pytest.approx(0.9)
    assert event_result["baseline_balanced_accuracy"] == pytest.approx(0.6)
    assert event_result["ci_lower"] > 0.0
    assert event_result["mcnemar"]["p_value_two_sided"] < 0.05


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (paired_bootstrap_difference, ([1.0], [1.0, 2.0])),
        (paired_rmse_comparison, ([float("nan")], [1.0])),
        (exact_mcnemar, ([2], [1])),
    ],
)
def test_pairwise_statistics_reject_malformed_inputs(function, args) -> None:
    with pytest.raises(ValueError):
        function(*args)


def test_formal_visual_matrix_is_exactly_16200_and_jobs_are_contiguous() -> None:
    train_seeds = (101, 202, 303)
    tasks = ("visual_event_stop", "visual_obstacle_avoid", "visual_target_select")
    physical_seeds = tuple(range(710_000, 710_100))
    cues = (0, 1)
    checkpoints = {
        variant: {
            seed: f"/checkpoint/{variant}/seed_{seed}" for seed in train_seeds
        }
        for variant in REQUIRED_VARIANTS
    }
    count = _expected_episode_count(
        variants=REQUIRED_VARIANTS,
        train_seeds=train_seeds,
        tasks=tasks,
        physical_seeds=physical_seeds,
        cue_variants=cues,
    )
    jobs = _evaluation_jobs(
        checkpoints=checkpoints,
        variants=REQUIRED_VARIANTS,
        train_seeds=train_seeds,
        tasks=tasks,
        physical_seeds=physical_seeds,
        cue_variants=cues,
        completed_start=37,
    )

    assert count == 9 * 3 * 3 * 100 * 2 == 16_200
    assert len(jobs) == 15
    assert sum(job["records"] for job in jobs) == count
    assert jobs[0]["completed_start"] == 37
    assert jobs[-1]["completed_start"] + jobs[-1]["records"] == 37 + count
    primary_jobs = [job for job in jobs if job["variant"] == PRIMARY_VARIANT]
    assert len(primary_jobs) == 3
    assert all(len(job["conditions"]) == 5 for job in primary_jobs)
    assert all(job["records"] == 5 * 3 * 100 * 2 for job in primary_jobs)

    diagnostic_conditions = ("clean", "shuffle_state")
    diagnostic_count = _expected_episode_count(
        variants=(PRIMARY_VARIANT,),
        train_seeds=(101,),
        tasks=tasks,
        physical_seeds=physical_seeds[:20],
        cue_variants=cues,
        condition_override=diagnostic_conditions,
    )
    diagnostic_jobs = _evaluation_jobs(
        checkpoints={PRIMARY_VARIANT: {101: "/checkpoint/diagnostic"}},
        variants=(PRIMARY_VARIANT,),
        train_seeds=(101,),
        tasks=tasks,
        physical_seeds=physical_seeds[:20],
        cue_variants=cues,
        completed_start=0,
        condition_override=diagnostic_conditions,
    )
    assert diagnostic_count == 2 * 3 * 20 * 2 == 240
    assert diagnostic_jobs[0]["conditions"] == diagnostic_conditions
    assert diagnostic_jobs[0]["records"] == diagnostic_count


def test_parallel_results_merge_in_preregistered_order_not_completion_order() -> None:
    first = _record(variant="state_only", success=True, evaluation_seed=1)
    second = _record(variant="vision_only", success=False, evaluation_seed=1)
    results = {
        1: (
            [second],
            {
                "latencies_ms": [2.0],
                "action_ages_ms": [50.0],
                "deadline_misses": 1,
                "replan_events": 4,
                "cold_replan_events": 3,
                "warm_replan_events": 1,
            },
        ),
        0: (
            [first],
            {
                "latencies_ms": [1.0],
                "action_ages_ms": [0.0],
                "deadline_misses": 0,
                "replan_events": 2,
                "cold_replan_events": 2,
                "warm_replan_events": 0,
            },
        ),
    }
    records, runtime = _merge_evaluation_results(results, expected_jobs=2)

    assert records == [first, second]
    assert runtime == {
        "latencies_ms": [1.0, 2.0],
        "action_ages_ms": [0.0, 50.0],
        "deadline_misses": 1,
        "replan_events": 6,
        "cold_replan_events": 5,
        "warm_replan_events": 1,
    }
    with pytest.raises(RuntimeError, match="missing"):
        _merge_evaluation_results({1: results[1]}, expected_jobs=2)


def test_runtime_warmup_uses_one_and_two_frames_then_resets_evidence() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, seed: int, randomize: bool):
            assert seed == 14
            assert randomize
            return {"proprioception": np.zeros(22, dtype=np.float32)}, {}

        def render(self, *, camera: str, width: int, height: int) -> np.ndarray:
            assert camera == "fixed"
            return np.zeros((height, width, 3), dtype=np.uint8)

        def step(self, action: np.ndarray):
            self.steps += 1
            return (
                {"proprioception": np.zeros(22, dtype=np.float32)},
                0.0,
                False,
                False,
                {},
            )

    class FakePolicy:
        def __init__(self) -> None:
            self.views: list[dict] = []
            self.reset_count = 0
            self.evidence: list[int] = []

        def reset(self) -> None:
            self.reset_count += 1
            self.evidence.clear()

        def act(self, observation: dict) -> np.ndarray:
            self.views.append(observation)
            self.evidence.append(len(self.views))
            return np.zeros(8, dtype=np.float32)

    env = FakeEnv()
    policy = FakePolicy()
    _warm_policy_runtime(
        env,  # type: ignore[arg-type]
        policy,  # type: ignore[arg-type]
        task_id="visual_event_stop",
        consumes_state=True,
        consumes_vision=True,
        seed=14,
        image_size=16,
    )

    assert env.steps == 2
    assert policy.reset_count == 2
    assert policy.evidence == []
    assert policy.views[0]["past_executed_actions"].shape == (0, 8)
    assert policy.views[1]["past_executed_actions"].shape == (1, 8)
    assert policy.views[0]["image_frame_indices"] == {"fixed": 0}
    assert policy.views[1]["image_frame_indices"] == {"fixed": 1}
