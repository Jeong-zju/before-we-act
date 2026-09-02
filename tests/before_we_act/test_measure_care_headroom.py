"""The headroom gate must fail a candidate family the selector cannot use.

This is the cheap check that replaces "train, calibrate, evaluate, discover the
override rate is zero". It has to be decisive in both directions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.before_we_act.measure_care_headroom import (
    build_report,
    family_advantages,
    horizon_summary,
    load_families,
)


HORIZON = 16
WEIGHT = 0.3409090909090909  # first entry of ORDINARY_WEIGHTS


def _outcome(progress: float) -> dict[str, Any]:
    vector = [0.0] * 8
    vector[0] = progress
    return {"bounded_utility_vector": vector, "hard_safety_violation": False}


def _family(
    snapshot: str,
    advantages: dict[int, float],
    *,
    task: str = "lift_barrier",
    repeats: tuple[int, ...] = (0, 1),
    response: float = 0.0,
) -> dict[str, Any]:
    """A family whose reactive candidate k beats the nominal action by advantages[k]."""
    branches = []
    for repeat in repeats:
        for regime in ("reactive", "replay"):
            branches.append(
                {
                    "candidate_id": 0,
                    "regime": regime,
                    "repeat_id": repeat,
                    "outcomes": {str(HORIZON): _outcome(0.0)},
                }
            )
            for candidate, gain in advantages.items():
                direct = gain - response
                value = gain if regime == "reactive" else direct
                branches.append(
                    {
                        "candidate_id": candidate,
                        "regime": regime,
                        "repeat_id": repeat,
                        "outcomes": {str(HORIZON): _outcome(value / WEIGHT)},
                    }
                )
    return {"snapshot_id": snapshot, "task": task, "branches": branches}


def test_advantages_decompose_into_direct_and_response() -> None:
    family = _family("s0", {1: 0.5}, repeats=(0,), response=0.2)
    rows = family_advantages(family, HORIZON)[0][1]

    assert rows["total"] == pytest.approx(0.5)
    assert rows["direct"] == pytest.approx(0.3)
    assert rows["response"] == pytest.approx(0.2)


def test_a_family_with_no_better_candidate_is_blocked() -> None:
    families = [_family(f"s{i}", {1: -0.01, 2: 0.0}) for i in range(12)]
    report = build_report(
        families,
        horizons=(HORIZON,),
        coverage=0.9,
        reference_radius=0.02,
        primary_horizon=HORIZON,
    )

    assert report["verdict"] == "BLOCKED"
    assert "never reaches the calibration radius" in report["reason"]


def test_a_family_that_clears_the_radius_passes() -> None:
    families = [_family(f"s{i}", {1: 0.2, 2: 0.1}) for i in range(12)]
    report = build_report(
        families,
        horizons=(HORIZON,),
        coverage=0.9,
        reference_radius=0.02,
        primary_horizon=HORIZON,
    )

    assert report["verdict"] == "PASS"
    against = report["by_horizon"][str(HORIZON)]["against_reference_radius"]
    assert against["oracle_override_rate"] == pytest.approx(1.0)
    assert against["signal_to_radius"] > 1.0


def test_a_rare_win_is_marginal_rather_than_a_pass() -> None:
    families = [_family(f"s{i}", {1: -0.01}) for i in range(39)]
    families.append(_family("s39", {1: 0.5}))
    report = build_report(
        families,
        horizons=(HORIZON,),
        coverage=0.9,
        reference_radius=0.02,
        primary_horizon=HORIZON,
    )

    assert report["verdict"] == "MARGINAL"
    assert "too little room" in report["reason"]


def test_summary_reports_the_signal_scale_that_drives_the_verdict() -> None:
    families = [_family(f"s{i}", {1: 0.004, 2: 0.0}) for i in range(11)]
    summary = horizon_summary(
        families, HORIZON, coverage=0.9, reference_radius=0.0239
    )

    assert summary["max_abs_total"] == pytest.approx(0.004)
    assert summary["fraction_positive"] == pytest.approx(0.5)
    # Matched repeats are identical here, so the noise floor is zero and the
    # whole radius is model error rather than something irreducible.
    assert summary["irreducible_radius"] == pytest.approx(0.0, abs=1e-9)
    assert summary["against_reference_radius"]["oracle_override_rate"] == 0.0


def test_loader_skips_files_that_are_not_branch_families(tmp_path: Path) -> None:
    (tmp_path / "note.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "family.json").write_text(
        json.dumps(_family("s0", {1: 0.1})), encoding="utf-8"
    )

    families = load_families([tmp_path])
    assert [row["snapshot_id"] for row in families] == ["s0"]


def test_report_groups_by_task() -> None:
    families = [
        _family("a0", {1: 0.2}, task="lift_barrier"),
        _family("b0", {1: -0.2}, task="pass_shoe"),
    ]
    report = build_report(
        families,
        horizons=(HORIZON,),
        coverage=0.9,
        reference_radius=0.02,
        primary_horizon=HORIZON,
    )

    assert report["tasks"] == ["lift_barrier", "pass_shoe"]
    per_task = report["by_task_primary_horizon"]
    assert per_task["lift_barrier"]["fraction_positive"] == pytest.approx(1.0)
    assert per_task["pass_shoe"]["fraction_positive"] == pytest.approx(0.0)
