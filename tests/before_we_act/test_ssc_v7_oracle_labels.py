from __future__ import annotations

from copy import deepcopy

import pytest

from before_we_act.ssc_v7_oracle_labels import (
    TASK_OBJECTS,
    TASK_STAGE_NAMES,
    build_oracle_label,
    initial_automaton_state,
    permute_agent_slots,
    permute_automaton_state,
    permute_label_slots,
)


AGENTS = {
    "lift_barrier": 2,
    "camera_alignment": 3,
    "long_pipeline_delivery": 4,
    "take_photo": 4,
    "pass_shoe": 2,
    "place_food": 2,
}


ROLES = {
    "lift_barrier": {"support_0": 0, "support_1": 1},
    "camera_alignment": {"camera_0": 0, "camera_1": 1, "meat": 2},
    "long_pipeline_delivery": {
        "pipeline_0": 3,
        "pipeline_1": 2,
        "pipeline_2": 1,
        "pipeline_3": 0,
    },
    "take_photo": {"camera_0": 0, "camera_1": 1, "meat": 2, "button": 3},
    "pass_shoe": {"source": 0, "receiver": 1},
    "place_food": {"meat": 0, "pot": 1},
}


def snapshot(task: str) -> dict:
    count = AGENTS[task]
    objects = TASK_OBJECTS[task]
    return {
        "task": task,
        "agent_count": count,
        "role_agents": deepcopy(ROLES[task]),
        "tcp_positions": [[0.2 * slot, 0.1 * slot, 0.3] for slot in range(count)],
        "base_positions": [[0.3 * slot, 0.2 * slot, 0.0] for slot in range(count)],
        "object_positions": {
            name: [0.1 * index, 0.05 * index, 0.03]
            for index, name in enumerate(objects)
        },
        "object_velocities": {name: [0.0, 0.0, 0.0] for name in objects},
        "goal_positions": {"goal": [1.0, 0.0, 0.0]},
        "reference_heights": {"robot_base": 0.0},
        "grasp": {name: [False] * count for name in objects},
        "contact": {name: [False] * count for name in objects},
        "contact_force": {name: [0.0] * count for name in objects},
        "task_predicates": {
            "button_aligned": False,
            "planar_meat_to_pot_distance": 0.25,
        },
        "robot_collision": False,
        "robot_proximity_risk": False,
        "drop_risk": {name: False for name in objects},
        "contact_threshold_ambiguous": False,
        "environment_success": False,
    }


REQUIRED = {
    "stage_id",
    "within_stage_progress",
    "per_agent_contribution",
    "remaining_goal_mask",
    "agent_object_role_slots",
    "grasp_contact_custody_state",
    "collision_drop_contention_risk",
    "causal_automaton_state",
    "label_validity_mask",
    "ambiguity_code",
}


@pytest.mark.parametrize("task", tuple(AGENTS))
def test_label_is_deterministic_complete_and_finite(task: str) -> None:
    state = snapshot(task)
    memory = initial_automaton_state(task)
    first = build_oracle_label(state, memory)
    second = build_oracle_label(deepcopy(state), deepcopy(memory))
    assert first == second
    assert REQUIRED <= set(first)
    assert 0.0 <= first["within_stage_progress"] <= 1.0
    assert first["task_complete"] is state["environment_success"]
    assert first["ambiguity_code"] == 0
    assert all(first["label_validity_mask"].values())


@pytest.mark.parametrize("task", tuple(AGENTS))
def test_terminal_label_exactly_tracks_environment(task: str) -> None:
    state = snapshot(task)
    state["environment_success"] = True
    result = build_oracle_label(state)
    assert result["task_complete"] is True
    assert result["within_stage_progress"] == 1.0
    state["environment_success"] = False
    assert build_oracle_label(state)["task_complete"] is False


@pytest.mark.parametrize("task", tuple(AGENTS))
def test_agent_slot_permutation_equivariance(task: str) -> None:
    state = snapshot(task)
    count = AGENTS[task]
    first_object = TASK_OBJECTS[task][0]
    state["contact"][first_object][0] = True
    state["contact_force"][first_object][0] = 2.0
    state["grasp"][first_object][0] = task not in {
        "lift_barrier",
        "camera_alignment",
        "take_photo",
    }
    memory = initial_automaton_state(task)
    original = build_oracle_label(state, memory)
    permutation = list(reversed(range(count)))
    renamed = build_oracle_label(
        permute_agent_slots(state, permutation),
        permute_automaton_state(memory, permutation),
    )
    expected = permute_label_slots(original, permutation)
    assert renamed == expected


def test_agent_slot_permutation_equivariance_with_latched_support_history() -> None:
    state = snapshot("lift_barrier")
    state["grasp"]["barrier"] = [True, False]
    memory = initial_automaton_state("lift_barrier")
    memory["achieved_predicate_latches"] = {"agent0_support": True}
    permutation = [1, 0]
    original = build_oracle_label(state, memory)
    renamed = build_oracle_label(
        permute_agent_slots(state, permutation),
        permute_automaton_state(memory, permutation),
    )
    assert renamed == permute_label_slots(original, permutation)


def test_take_photo_has_no_unobserved_preterminal_button_stage() -> None:
    state = snapshot("take_photo")
    state["task_predicates"]["button_aligned"] = True
    state["environment_success"] = True
    result = build_oracle_label(state)
    assert "button_aligned" not in TASK_STAGE_NAMES["take_photo"]
    assert result["stage_id"] == "photo_complete"
    assert result["factorized_predicates"]["button_aligned"] is True


def test_place_food_stages_follow_observed_expert_object_motion() -> None:
    state = snapshot("place_food")
    assert build_oracle_label(state)["stage_id"] == "approach"
    state["grasp"]["meat"] = [True, False]
    controlled = build_oracle_label(state)
    assert controlled["stage_id"] == "meat_controlled"
    assert "pot_grasp" not in controlled["remaining_goal_mask"]
    state["object_positions"]["meat"][2] = 0.15
    lifted = build_oracle_label(state)
    assert lifted["stage_id"] == "meat_lifted"
    state["task_predicates"]["planar_meat_to_pot_distance"] = 0.05
    assert build_oracle_label(state)["stage_id"] == "aligned"
    state["environment_success"] = True
    released = build_oracle_label(state)
    assert released["stage_id"] == "released"
    assert all(
        row["object"] != "pot" or row["roles"] == ["none"]
        for row in released["agent_object_role_slots"]
    )


def test_pass_shoe_causal_memory_records_only_observed_transfer() -> None:
    state = snapshot("pass_shoe")
    memory = initial_automaton_state("pass_shoe")
    state["grasp"]["shoe"] = [True, False]
    first = build_oracle_label(state, memory)
    assert first["stage_id"] == "agent0_custody"
    state["grasp"]["shoe"] = [False, False]
    between = build_oracle_label(state, first["causal_automaton_state"])
    assert between["stage_id"] == "handoff"
    state["grasp"]["shoe"] = [False, True]
    second = build_oracle_label(state, between["causal_automaton_state"])
    assert second["stage_id"] == "agent1_custody"
    assert second["causal_automaton_state"]["custody_transfer_count"] == 1
    assert second["causal_automaton_state"]["completed_handoff_mask"] == [True]


def test_wrong_delivery_order_is_explicitly_ambiguous() -> None:
    state = snapshot("long_pipeline_delivery")
    state["grasp"]["shoe"][2] = True
    result = build_oracle_label(state)
    assert result["ambiguity_code"] != 0
    assert not result["label_validity_mask"]["grasp_contact_custody_state"]
