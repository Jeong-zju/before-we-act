from __future__ import annotations

import json

from scripts.before_we_act import run_ssc_v7_m3_r4_c as r4c


def candidate_summary() -> dict[str, object]:
    per_task = {
        task: {"gain": 0.1, "ci95": [0.05, 0.15]}
        for task in r4c.TASKS
    }
    return {
        "macro_gain": 0.1,
        "ci95": [0.05, 0.15],
        "positive_tasks": list(r4c.TASKS),
        "per_task": per_task,
    }


def test_signal_first_checks_do_not_include_diagnostic_control_ordering() -> None:
    checks, harms = r4c.signal_first_checks(
        candidate_summary(),
        [
            {"macro_gain": 0.10, "ci95": [0.05, 0.15]},
            {"macro_gain": 0.08, "ci95": [0.03, 0.13]},
            {"macro_gain": 0.09, "ci95": [0.04, 0.14]},
        ],
        0.8,
        {
            "mean_predictor_brier": 0.04,
            "mean_constant_rate_brier": 0.12,
            "error_rises_as_reliability_falls": True,
        },
        {"exact_fallback": True},
        {"episode_count": 72, "test_open_events": 1},
        {
            "positive_tasks_min": 2,
            "stable_task_harm_threshold_abs": 0.03,
            "oracle_gain_retention_min": 0.5,
        },
        True,
        72,
    )
    assert harms == []
    assert all(checks.values())
    assert not any("shuffle" in key or "hidden" in key or "time" in key for key in checks)


def test_signal_first_checks_preserve_one_time_boundary() -> None:
    checks, _ = r4c.signal_first_checks(
        candidate_summary(),
        [{"macro_gain": 0.1, "ci95": [0.05, 0.15]}] * 3,
        0.8,
        {
            "mean_predictor_brier": 0.04,
            "mean_constant_rate_brier": 0.12,
            "error_rises_as_reliability_falls": True,
        },
        {"exact_fallback": True},
        {"episode_count": 72, "test_open_events": 2},
        {
            "positive_tasks_min": 2,
            "stable_task_harm_threshold_abs": 0.03,
            "oracle_gain_retention_min": 0.5,
        },
        True,
        72,
    )
    assert checks["test_opened_exactly_once"] is False


def test_frozen_gate_integrity_rejects_tampering(tmp_path) -> None:
    gate = {
        "stage_id": r4c.STAGE_ID,
        "status": r4c.FROZEN_STATUS,
        "value": 1,
    }
    gate["integrity"] = {
        "payload_sha256": r4c.hashlib.sha256(r4c.canonical_bytes(gate)).hexdigest()
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    assert r4c.load_gate(path)["value"] == 1
    gate["value"] = 2
    path.write_text(json.dumps(gate), encoding="utf-8")
    try:
        r4c.load_gate(path)
    except RuntimeError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered R4-C gate was accepted")


def test_all_frozen_residual_conditions_are_explicit() -> None:
    assert r4c.ALL_CONDITIONS == (
        "arb_hat_direct",
        "row_shuffled_direct",
        "time_only_direct",
        "episode_shuffled_direct",
        "stale_8_direct",
        "stale_16_direct",
        "hc_hidden_only_direct",
        "phase_matched_row_shuffle_direct",
        "oracle_direct",
    )
