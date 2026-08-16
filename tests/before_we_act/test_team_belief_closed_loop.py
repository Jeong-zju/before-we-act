from pathlib import Path

from scripts.before_we_act.prepare_team_belief_closed_loop import (
    OWNER_TOKEN,
    PRIMARY_SERIES,
    primary_plateau_override,
)


def _training_row(*, extra_blocker: bool = False) -> dict:
    series = {
        name: {"all_three_below_one_percent": True}
        for name in PRIMARY_SERIES
    }
    series["teammate_delta"] = {
        "all_three_below_one_percent": False,
        "relative_improvements": [0.02, 0.03, 0.01],
    }
    if extra_blocker:
        series["future_1.6s"]["all_three_below_one_percent"] = False
    return {
        "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
        "training_sufficiency": {
            "minimum_exposure_met": True,
            "learning_rate_drop_completed": True,
            "receipt": {
                "maximum_updates": 120000,
                "points": [115000, 120000],
                "overfit_last_three_intervals": False,
                "series": series,
            },
        },
    }


def test_owner_override_requires_only_teammate_delta_to_block() -> None:
    result = primary_plateau_override(
        {str(seed): _training_row() for seed in (20260815, 20260816, 20260817)}
    )

    assert result["passed"] is True
    assert all(
        row["old_rule_blocking_series"] == ["teammate_delta"]
        for row in result["per_seed"].values()
    )


def test_owner_override_rejects_a_deployment_series_blocker() -> None:
    training = {
        "20260815": _training_row(),
        "20260816": _training_row(extra_blocker=True),
        "20260817": _training_row(),
    }

    assert primary_plateau_override(training)["passed"] is False


def test_owner_runner_preserves_old_conclusion_and_completes_validation20() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/before_we_act/run_team_belief_closed_loop.sh"
    ).read_text(encoding="utf-8")
    validation20 = (
        root / "scripts/before_we_act/run_team_belief_validation.sh"
    ).read_text(encoding="utf-8")

    assert OWNER_TOKEN in launcher
    assert 'OLD_CONCLUSION="${RUN_ROOT}/conclusion.json"' in launcher
    assert 'BWA_TEAM_BELIEF_CONCLUSION="${AUTHORIZATION}"' in launcher
    assert "run_team_belief_validation.sh" in launcher
    assert 'BWA_TEAM_BELIEF_CONCLUSION:-${RUN_ROOT}/conclusion.json' in validation20
