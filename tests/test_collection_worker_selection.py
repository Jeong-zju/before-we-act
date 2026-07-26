from __future__ import annotations

import pytest

from scripts.select_robofactory_collection_workers import choose_worker_plan


def test_worker_selection_reserves_memory_for_640x480_lift_rgb() -> None:
    plan = choose_worker_plan(
        task="lift_barrier",
        trajectories=150,
        effective_cpus=32,
        available_memory_bytes=64 * 1024**3,
        requested="auto",
        max_workers=16,
        cpu_threads_per_worker=2,
        memory_fraction=0.8,
    )

    assert plan.workers == 12
    assert plan.cpu_limit == 15
    assert plan.memory_limit == 12
    assert plan.threads_per_worker == 2


def test_worker_selection_reserves_memory_for_five_long_rgb_streams() -> None:
    plan = choose_worker_plan(
        task="long_pipeline_delivery",
        trajectories=150,
        effective_cpus=32,
        available_memory_bytes=52 * 1024**3,
        requested="auto",
        max_workers=16,
        cpu_threads_per_worker=2,
        memory_fraction=0.8,
    )

    assert plan.workers == 6
    assert plan.memory_limit == 6
    assert plan.threads_per_worker == 5


def test_manual_override_is_preserved_and_marked() -> None:
    plan = choose_worker_plan(
        task="long_pipeline_delivery",
        trajectories=150,
        effective_cpus=8,
        available_memory_bytes=16 * 1024**3,
        requested="7",
        max_workers=16,
        cpu_threads_per_worker=2,
        memory_fraction=0.8,
    )

    assert plan.workers == 7
    assert plan.manually_overridden
    assert plan.workers > plan.automatic_limit


@pytest.mark.parametrize("requested", ("0", "151", "invalid"))
def test_invalid_manual_override_is_rejected(requested: str) -> None:
    with pytest.raises(ValueError):
        choose_worker_plan(
            task="lift_barrier",
            trajectories=150,
            effective_cpus=8,
            available_memory_bytes=16 * 1024**3,
            requested=requested,
            max_workers=16,
            cpu_threads_per_worker=2,
            memory_fraction=0.8,
        )
