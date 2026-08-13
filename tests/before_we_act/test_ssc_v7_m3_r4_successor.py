from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.before_we_act import run_ssc_v7_m3 as m3
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4
from scripts.before_we_act import run_ssc_v7_m3_r4_successor as successor


def label() -> dict:
    return {
        "ambiguity_code": 0,
        "stage_id": "forbidden_progress",
        "within_stage_progress": 0.75,
        "remaining_goal_mask": ["forbidden_goal"],
        "future_teammate_actions": [999],
        "label_validity_mask": {
            "grasp_contact_custody_state": True,
            "causal_automaton_state": True,
            "per_agent_contribution": True,
            "collision_drop_contention_risk": True,
        },
        "grasp_contact_custody_state": {
            "shoe": {
                "contact_agents": [0, 1],
                "grasp_agents": [1],
                "controller_agents": [1],
                "current_custodian": 1,
                "shared_control": True,
            }
        },
        "causal_automaton_state": {
            "completed_handoff_mask": ["forbidden_progress_event"],
            "custody_transfer_count": 17,
        },
        "per_agent_contribution": [
            {
                "agent_slot": 0,
                "active": False,
                "contact_objects": [],
                "grasp_objects": [],
                "roles": [],
            },
            {
                "agent_slot": 1,
                "active": True,
                "contact_objects": ["shoe"],
                "grasp_objects": ["shoe"],
                "roles": ["receiver"],
            },
        ],
        "collision_drop_contention_risk": {
            "robot_collision": False,
            "robot_proximity_risk": True,
            "contested_objects": ["shoe"],
            "dropped_objects": [],
        },
    }


def hc_payload() -> dict:
    model = m3.build_action_model(successor.HC_INPUT_WIDTH, 256, 17)
    return {
        "state_dict": model.state_dict(),
        "input_width": successor.HC_INPUT_WIDTH,
        "hidden_width": 256,
    }


def test_sanitized_legacy_excludes_progress_future_and_fixed_robot_ids() -> None:
    first = label()
    changed = deepcopy(first)
    changed["stage_id"] = "another_stage"
    changed["within_stage_progress"] = 0.01
    changed["remaining_goal_mask"] = []
    changed["future_teammate_actions"] = [-123, -456]
    changed["causal_automaton_state"] = {
        "completed_handoff_mask": [],
        "custody_transfer_count": 0,
    }
    assert np.array_equal(
        successor.sanitized_legacy_tokens(first, own_slot=0),
        successor.sanitized_legacy_tokens(changed, own_slot=0),
    )

    renamed = deepcopy(first)
    mapping = {0: 7, 1: 3}
    for state in renamed["grasp_contact_custody_state"].values():
        for key in ("contact_agents", "grasp_agents", "controller_agents"):
            state[key] = [mapping[value] for value in state[key]]
        state["current_custodian"] = mapping[state["current_custodian"]]
    for item in renamed["per_agent_contribution"]:
        item["agent_slot"] = mapping[item["agent_slot"]]
    assert np.array_equal(
        successor.sanitized_legacy_tokens(first, own_slot=0),
        successor.sanitized_legacy_tokens(renamed, own_slot=7),
    )


def test_standardized_2x2_parameter_counts_are_matched() -> None:
    counts = successor.parameter_counts(101)
    assert counts["query_attention_residual"] == 42753
    assert counts["direct_residual_mlp"] == 42835
    assert counts["within_5pct"] is True
    assert counts["relative_spread"] < 0.003


def test_both_fusions_start_as_exact_hc_fallback() -> None:
    torch = pytest.importorskip("torch")
    payload = hc_payload()
    rng = np.random.default_rng(9)
    hc = rng.normal(size=(5, successor.HC_INPUT_WIDTH)).astype(np.float32)
    side = rng.normal(size=(5, successor.FEATURE_WIDTH)).astype(np.float32)
    reliability = np.ones((5, 1), dtype=np.float32)
    values = torch.from_numpy(np.concatenate((hc, side, reliability), axis=1))
    with torch.no_grad():
        baseline = r4.HCWrapper.create(payload)(torch.from_numpy(hc))
        query = r4.ArbResidualFactory.create(payload, seed=23).eval()(values)
        direct = successor.DirectResidualFactory.create(payload, seed=23).eval()(values)
    assert torch.equal(query, baseline)
    assert torch.equal(direct, baseline)


def probe_for_shuffle() -> m3.ProbeData:
    rows = len(successor.TASKS) * 3
    tasks = np.repeat(np.asarray(successor.TASKS, dtype="U32"), 3)
    return m3.ProbeData(
        legal=np.zeros((rows, 1), dtype=np.float32),
        e0=np.zeros((rows, 1), dtype=np.float32),
        social=np.zeros((rows, 192), dtype=np.float32),
        time=np.zeros((rows, 1), dtype=np.float32),
        target=np.zeros((rows, 1, 1), dtype=np.float32),
        target_mask=np.ones((rows, 1), dtype=np.float32),
        tasks=tasks,
        episode_ids=np.asarray([f"episode_{index}" for index in range(rows)], dtype="U64"),
        frame_indices=np.arange(rows, dtype=np.int32),
        agent_slots=np.zeros(rows, dtype=np.int16),
    )


def test_label_shuffle_is_task_local_distribution_preserving_derangement() -> None:
    data = probe_for_shuffle()
    values = np.arange(len(data) * 2, dtype=np.float32).reshape(len(data), 1, 2)
    shuffled = successor.label_shuffle(values, data, seed=31)
    for task in successor.TASKS:
        indices = np.flatnonzero(data.tasks == task)
        assert all(not np.array_equal(values[index], shuffled[index]) for index in indices)
        assert sorted(shuffled[indices, 0, 0].tolist()) == sorted(
            values[indices, 0, 0].tolist()
        )


def test_sealed_test_split_is_rejected_before_loading(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        successor.load_successor_bundle(manifest, {"read_only_test"})
