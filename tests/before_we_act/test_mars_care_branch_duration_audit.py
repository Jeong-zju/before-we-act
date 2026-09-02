from __future__ import annotations

import json
from pathlib import Path

from scripts.before_we_act.analyze_mars_care_branch_duration import audit_family
from scripts.before_we_act.analyze_mars_care_branch_parity import aggregate_rows


def _outcome(value: float, stage: str = "approach") -> dict[str, object]:
    return {
        "utility_main": value,
        "final_stage_id": stage,
        "hard_safety_violation": False,
    }


def test_duration_audit_counts_effective_direct_response_pairs(tmp_path: Path) -> None:
    branches = []
    for repeat in (0, 1):
        for regime in ("reactive", "replay"):
            for candidate in range(6):
                value = 0.0
                if candidate == 1:
                    value = 0.04 if regime == "reactive" else 0.015
                branches.append(
                    {
                        "repeat_id": repeat,
                        "regime": regime,
                        "candidate_id": candidate,
                        "candidate_valid": True,
                        "outcomes": {str(h): _outcome(value) for h in (8, 16, 32, 64)},
                        "executed_actions": [
                            {"panda-0": [float(candidate)] * 8} for _ in range(4)
                        ],
                        "restore_observation_max_abs_error": 0.0,
                        "replay_teammate_action_max_abs_error": 0.0,
                        "candidate0_reference_action_max_abs_error": 0.0,
                    }
                )
    family = {
        "task": "place_cube_in_cup",
        "snapshot_id": "family",
        "focal_agent": 0,
        "intervention_steps": 4,
        "candidate_legality": [{"valid": True} for _ in range(6)],
        "branches": branches,
    }
    path = tmp_path / "family.json"
    path.write_text(json.dumps(family))
    result = audit_family(path, 4)
    assert result["branch_count"] == 24
    assert result["pair_count"] == 40
    assert result["effective_pair_count"] == 8
    assert result["signal_density"] == 0.2
    assert result["action_exposure_l2"]["max"] > 2.8
    assert result["maximum_restore_error"] == 0.0


def test_duration_supervisor_is_single_shot_and_keeps_main_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = (root / "deployment/mars-care-branch-duration-smoke.supervisor.conf").read_text()
    script = (root / "scripts/before_we_act/run_mars_care_branch_duration_smoke.sh").read_text()
    for value in ("autostart=false", "autorestart=false", "startretries=0"):
        assert value in config
    assert "durations=(1 4 8 16)" in script
    assert "--limit 1 --intervention-steps" in script
    assert "fixed_stratified_main_protocol_unchanged\":True" in script
    assert "validation20_used_for_tuning\":False" in script
    assert "automatic_retry\":False" in script
    assert "globally serial Vulkan on GPU0" in script


def test_h8_parity_promotion_uses_four_task_aggregate_signal_gate() -> None:
    effective_by_task = (10, 0, 20, 4)
    rows = []
    for effective in effective_by_task:
        rows.append(
            {
                "pair_count": 40,
                "effective_pair_count": effective,
                "direct_signal_pair_count": effective,
                "response_signal_pair_count": 0,
                "total_signal_pair_count": effective,
                "execution_parity": True,
                "all_branch_support_complete": True,
                "all_candidates_legal": True,
                "maximum_restore_error": 0.0,
                "maximum_restore_rerender_diagnostic_error": 16.0,
                "maximum_replay_teammate_action_error": 0.0,
                "maximum_candidate0_reference_action_error": 0.0,
                "hard_safety_pair_count": 0,
                "passes_parity_and_safety": True,
            }
        )

    result = aggregate_rows(rows, baseline_h1_density=0.0625)

    assert result["effective_pair_count"] == 34
    assert result["signal_density"] == 0.2125
    assert result["signal_density_gain_over_h1"] == 0.15
    assert result["signal_density_ratio_over_h1"] == 3.4
    # One task deliberately has zero signal.  It must not veto the
    # pre-registered aggregate duration decision.
    assert result["eligible_for_scorer_smoke"] is True


def test_h8_parity_aggregate_still_fails_closed_on_physical_parity() -> None:
    row = {
        "pair_count": 40,
        "effective_pair_count": 10,
        "direct_signal_pair_count": 10,
        "response_signal_pair_count": 0,
        "total_signal_pair_count": 10,
        "execution_parity": True,
        "all_branch_support_complete": True,
        "all_candidates_legal": True,
        "maximum_restore_error": 0.0,
        "maximum_restore_rerender_diagnostic_error": 16.0,
        "maximum_replay_teammate_action_error": 0.0,
        "maximum_candidate0_reference_action_error": 0.0,
        "hard_safety_pair_count": 0,
        "passes_parity_and_safety": True,
    }
    rows = [dict(row) for _ in range(4)]
    rows[2]["maximum_restore_error"] = 1e-3
    rows[2]["passes_parity_and_safety"] = False

    result = aggregate_rows(rows, baseline_h1_density=0.0625)

    assert result["signal_density"] == 0.25
    assert result["eligible_for_scorer_smoke"] is False


def test_h8_formal_collection_is_single_shot_and_preregistered() -> None:
    root = Path(__file__).resolve().parents[2]
    config = (
        root / "deployment/mars-care-h8-formal-collection.supervisor.conf"
    ).read_text()
    script = (
        root / "scripts/before_we_act/run_mars_care_h8_formal_collection.sh"
    ).read_text()
    for value in (
        "autostart=false",
        "autorestart=false",
        "startretries=0",
        "stopasgroup=true",
        "killasgroup=true",
    ):
        assert value in config
    assert "intervention_steps\":8" in script
    assert '"reference_policy":"B-core/TUNE"' in script
    assert '"all_family_training":True' in script
    assert '"validation20_used_for_tuning":False' in script
    assert '"legacy_h1_corpus_unchanged":True' in script
    assert '"automatic_retry":False' in script
    assert '"globally_serial_vulkan":True' in script
    assert "--intervention-steps 8" in script
    assert "--limit" not in script
