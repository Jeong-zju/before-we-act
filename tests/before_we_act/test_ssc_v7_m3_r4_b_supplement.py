from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.before_we_act import run_ssc_v7_m3_r4_b as r4b
from scripts.before_we_act import run_ssc_v7_m3_r4_b_supplement as supplement


def grouped_data() -> SimpleNamespace:
    tasks: list[str] = []
    agent_slots: list[int] = []
    ratios: list[float] = []
    for task in r4b.TASKS:
        for agent_slot in (0, 1):
            for ratio in (0.10, 0.20, 0.30, 0.40, 0.55, 0.60, 0.80, 0.90):
                tasks.append(task)
                agent_slots.append(agent_slot)
                ratios.append(ratio)
    time = np.zeros((len(tasks), 192), dtype=np.float32)
    time[:, 0] = np.asarray(ratios, dtype=np.float32)
    return SimpleNamespace(
        tasks=np.asarray(tasks, dtype="U32"),
        agent_slots=np.asarray(agent_slots, dtype=np.int16),
        time=time,
    )


def test_phase_matched_shuffle_deranges_only_inside_frozen_groups() -> None:
    data = grouped_data()
    values = np.arange(len(data.tasks), dtype=np.int64).reshape(-1, 1)
    shuffled = supplement.phase_matched_shuffle(values, data, seed=19)
    bins = supplement.phase_bins(data)
    assert not np.any(shuffled[:, 0] == values[:, 0])
    for task in r4b.TASKS:
        for phase in range(4):
            mask = (data.tasks == task) & (bins == phase)
            assert set(shuffled[mask, 0]) == set(values[mask, 0])


def test_phase_matched_shuffle_merges_a_singleton_phase() -> None:
    data = grouped_data()
    bins = supplement.phase_bins(data)
    keep = np.ones(len(data.tasks), dtype=bool)
    first_task = r4b.TASKS[0]
    last_phase = (data.tasks == first_task) & (bins == 3)
    keep[np.flatnonzero(last_phase)[1:]] = False
    sparse = SimpleNamespace(
        tasks=data.tasks[keep],
        agent_slots=data.agent_slots[keep],
        time=data.time[keep],
    )
    values = np.arange(len(sparse.tasks), dtype=np.int64).reshape(-1, 1)
    shuffled = supplement.phase_matched_shuffle(values, sparse, seed=23)
    assert not np.any(shuffled[:, 0] == values[:, 0])


def core_source_receipt() -> dict[str, object]:
    return {
        "r4_b_completed": True,
        "test_paths_opened": 0,
        "strict_checks": {
            "arb_hat_ci_lower_positive": True,
            "at_least_two_positive_tasks": True,
            "at_least_two_of_three_seeds_positive": True,
            "no_stable_task_harm_at_3pct": True,
            "no_seed_stably_harmed_at_3pct": True,
            "retains_at_least_half_oracle_direct_gain": True,
            "all_active_heads_beat_constant_brier": True,
            "reliability_bins_directional": True,
            "gate_off_exactly_returns_hc": True,
            "all_action_branches_converged": True,
            # These source diagnostics intentionally stay outside the amendment.
            "arb_hat_beats_row_shuffle": False,
            "arb_hat_beats_time_only": False,
            "stale_degradation_monotonic": False,
        },
    }


def test_signal_first_decision_ignores_diagnostic_ordering() -> None:
    checks = supplement.decision_checks(core_source_receipt(), True, 0)
    assert all(checks.values())


def test_signal_first_decision_still_protects_test_boundary() -> None:
    checks = supplement.decision_checks(core_source_receipt(), True, 1)
    assert checks["sealed_test_untouched"] is False


def test_attribution_codes_are_honest() -> None:
    assert supplement.attribution_code({"ci95": [0.01, 0.03], "macro_gain": 0.02}) == (
        "SUPPORTED_ARB_HAT_INCREMENT_BEYOND_CONTROL"
    )
    assert supplement.attribution_code({"ci95": [-0.01, 0.03], "macro_gain": 0.01}) == (
        "DIRECTIONAL_ARB_HAT_INCREMENT_NOT_CI_CONFIRMED"
    )
    assert supplement.attribution_code({"ci95": [-0.03, 0.01], "macro_gain": -0.01}) == (
        "ARB_HAT_SEMANTIC_INCREMENT_NOT_ISOLATED_FROM_CONTROL"
    )
