from __future__ import annotations

import json
from pathlib import Path

from scripts.before_we_act.analyze_mars_care_rollout_diagnostic import summarize_episode


def test_rollout_audit_counts_events_masks_and_action_signal(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    start = {"type": "episode_start"}
    row = {
        "type": "step",
        "step": 0,
        "privileged_metrics": {"stage_id": "approach", "progress": 0.25, "success": False},
        "events": {"progress_increase": True, "strike_proxy_is_not_contact": True},
        "care_diagnostics": {"gate": 0.2, "reliability": 0.4, "sigma": 0.1, "events": 2},
        "selection_reason": ["reference_below_delta"],
        "best_lower": [0.0],
        "candidate_legality": [[True, False]],
        "illegal_mask": [[False, True]],
        "learned_unsafe_mask": [[False, False]],
        "assembly": {"proposed_rows": [], "applied_rows": []},
        "action_bounds": {"panda-0": {"elements_clipped": 1}},
        "reference_first_action": {"panda-0": [0.0] * 8},
        "candidate_first_actions": [[[0.0] * 8, [1.0] * 8]],
        "action_applied": {"panda-0": [0.0] * 8},
    }
    end = {"type": "episode_end"}
    path.write_text("\n".join(json.dumps(value) for value in (start, row, end)) + "\n")
    result = summarize_episode(path)
    assert result["steps"] == 1
    assert result["first_event_steps"] == {"progress_increase": 0}
    assert result["candidate_legal_fraction"] == 0.5
    assert result["illegal_mask_count"] == 1
    assert result["action_elements_clipped"] == 1
    assert result["candidate_first_action_l2_from_reference"]["max"] > 2.8
    assert result["applied_action_l2_from_reference"]["max"] == 0.0
