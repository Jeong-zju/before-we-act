"""Evaluation statistics tests."""

from __future__ import annotations

import json

import pytest

from eval.evaluate import (
    DEPLOYABLE_COMMUNICATION_MODES,
    EvaluationContractError,
    compare_communication_modes,
    load_evaluation_records,
)


def test_five_mode_report_has_deterministic_paired_ci_scenarios_and_acceptance():
    records = _acceptance_records()

    report = compare_communication_modes(
        records,
        required_modes=DEPLOYABLE_COMMUNICATION_MODES,
        bootstrap_samples=512,
        bootstrap_seed=19,
    )
    repeated = compare_communication_modes(
        records,
        required_modes=DEPLOYABLE_COMMUNICATION_MODES,
        bootstrap_samples=512,
        bootstrap_seed=19,
    )

    assert report["mode_order"] == list(DEPLOYABLE_COMMUNICATION_MODES)
    assert report["deployable_mode_order"] == list(DEPLOYABLE_COMMUNICATION_MODES)
    assert list(report["deployable_modes"]) == list(DEPLOYABLE_COMMUNICATION_MODES)
    assert "oracle_upper_bound" not in report["modes"]
    assert report["bootstrap"] == {
        "samples": 512,
        "seed": 19,
        "confidence_level": 0.95,
        "method": "paired_episode_percentile",
    }
    assert report["paired_comparisons"] == repeated["paired_comparisons"]
    assert report["modes"]["selective_vpi"][
        "actual_communication_delay_mean"
    ] == 0.0
    assert report["modes"]["selective_vpi"][
        "expected_latency_cost_mean"
    ] == pytest.approx(0.2)
    assert report["latency_semantics"]["real_network_latency_validated"] is False

    success = report["paired_comparisons"]["selective_vpi"]["vs_no_comm"][
        "metrics"
    ]["success_rate"]
    assert success["mean_delta"] == pytest.approx(1.0)
    assert success["ci95"] == pytest.approx([1.0, 1.0])
    assert success["paired_episode_count"] == 4
    assert success["ci_excludes_zero"] is True

    assert set(report["scenario_breakdown"]) == {"hard_comm", "nominal"}
    assert report["scenario_breakdown"]["nominal"]["episode_count"] == 2
    assert report["scenario_breakdown"]["hard_comm"]["episode_count"] == 2
    assert (
        report["scenario_breakdown"]["nominal"]["modes"]["selective_vpi"][
            "success_rate"
        ]
        == 1.0
    )

    acceptance = report["selective_vpi_acceptance"]
    assert acceptance["status"] == "passed"
    assert acceptance["passed"] is True
    assert all(check["passed"] for check in acceptance["checks"].values())
    assert acceptance["checks"][
        "bits_reduced_at_least_50pct_vs_always_reply"
    ]["observed_reduction_fraction"] == pytest.approx(0.6)
    assert acceptance["success_advantage_over_no_comm"][
        "supported_by_paired_ci"
    ] is True
    assert "only when" in acceptance["statistical_claim_policy"]


def test_bootstrap_pairs_by_episode_key_instead_of_record_order():
    records = _acceptance_records(episode_count=2)
    records["no_comm"][0]["success"] = False
    records["no_comm"][1]["success"] = True
    records["selective_vpi"][0]["success"] = True
    records["selective_vpi"][1]["success"] = False
    records["selective_vpi"].reverse()

    report = compare_communication_modes(
        records,
        required_modes=DEPLOYABLE_COMMUNICATION_MODES,
        bootstrap_samples=2_000,
        bootstrap_seed=7,
    )
    stats = report["modes"]["selective_vpi"]["paired_deltas"]["vs_no_comm"][
        "metrics"
    ]["success_rate"]

    assert stats["mean_delta"] == pytest.approx(0.0)
    assert stats["ci95"][0] < 0.0
    assert stats["ci95"][1] > 0.0
    assert stats["ci_excludes_zero"] is False


def test_manifest_metadata_is_not_treated_as_a_mode(tmp_path):
    records = _acceptance_records(episode_count=1)
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps({"metadata": {"split": "test"}, **records}),
        encoding="utf-8",
    )

    loaded = load_evaluation_records(path)

    assert set(loaded) == set(DEPLOYABLE_COMMUNICATION_MODES)


def test_paired_scenario_mismatch_is_rejected():
    records = _acceptance_records(episode_count=1)
    records["periodic"][0]["scenario"] = "different"

    with pytest.raises(EvaluationContractError, match="scenarios differ"):
        compare_communication_modes(
            records,
            required_modes=DEPLOYABLE_COMMUNICATION_MODES,
            bootstrap_samples=10,
        )


def test_truncated_smoke_never_reports_formal_acceptance():
    records = _acceptance_records(episode_count=1)
    for mode_records in records.values():
        mode_records[0]["truncated"] = True

    report = compare_communication_modes(
        records,
        required_modes=DEPLOYABLE_COMMUNICATION_MODES,
        bootstrap_samples=10,
    )

    assert report["truncated_episode_record_count"] == len(
        DEPLOYABLE_COMMUNICATION_MODES
    )
    acceptance = report["selective_vpi_acceptance"]
    assert acceptance["status"] == "not_applicable_truncated"
    assert acceptance["passed"] is None


def _acceptance_records(episode_count: int = 4) -> dict[str, list[dict]]:
    bits = {
        "no_comm": 0.0,
        "always_reply": 100.0,
        "selective_vpi": 40.0,
        "periodic": 50.0,
        "random": 50.0,
    }
    successes = {
        "no_comm": False,
        "always_reply": True,
        "selective_vpi": True,
        "periodic": True,
        "random": False,
    }
    records: dict[str, list[dict]] = {}
    for mode in DEPLOYABLE_COMMUNICATION_MODES:
        records[mode] = []
        for index in range(episode_count):
            scenario = "nominal" if index < 2 else "hard_comm"
            communicated = mode != "no_comm"
            records[mode].append(
                {
                    "seed": 100 + index,
                    "episode_id": f"episode-{index}",
                    "input_digest": f"input-{index}",
                    "scenario": scenario,
                    "success": successes[mode],
                    "safe": True,
                    "return": float(index),
                    "collision_count": 0,
                    "force_violation_rate": 0.0,
                    "max_force": 1.0,
                    "decision_count": 1,
                    "steps": [
                        {
                            "request": communicated,
                            "reply": communicated,
                            "actual_round_trip_bits": bits[mode],
                            "actual_communication_delay": 0.0,
                            "expected_latency_cost": 0.2,
                            "VPI": 0.5,
                            "G_before": 2.0,
                            "G_after": 1.0,
                            "replanned": communicated,
                        }
                    ],
                }
            )
    return records
