"""Pure, past-causal oracle labels for the SSC-V7 M2 measurement gate.

This module deliberately has no simulator dependency.  The runtime audit extracts a
small privileged snapshot from RoboFactory and passes it here together with the
previous causal-automaton memory.  Keeping the labeler pure makes determinism,
permutation equivariance, and dependency auditing testable without a renderer.
"""

from __future__ import annotations

from copy import deepcopy
from math import dist
from typing import Any, Mapping, Sequence


AMBIGUITY_NONE = 0
AMBIGUITY_CONTACT_THRESHOLD = 1
AMBIGUITY_CONTENTION = 2
AMBIGUITY_CUSTODY_ORDER = 4


TASK_STAGE_NAMES: dict[str, tuple[str, ...]] = {
    "lift_barrier": ("approach", "single_support", "dual_support", "lifted"),
    "camera_alignment": (
        "approach",
        "partial_support_or_grasp",
        "camera_and_meat_controlled",
        "camera_and_meat_lifted",
    ),
    "long_pipeline_delivery": (
        "approach",
        "custody_0",
        "handoff_01",
        "custody_1",
        "handoff_12",
        "custody_2",
        "handoff_23",
        "custody_3",
        "goal",
    ),
    "take_photo": (
        "approach",
        "partial_support_or_grasp",
        "camera_and_meat_controlled",
        "button_aligned",
        "photo_complete",
    ),
    "pass_shoe": (
        "approach",
        "agent0_custody",
        "handoff",
        "agent1_custody",
        "goal",
    ),
    "place_food": (
        "approach",
        "partial_grasp",
        "joint_control",
        "joint_lift",
        "aligned",
        "released",
    ),
}


TASK_OBJECTS: dict[str, tuple[str, ...]] = {
    "lift_barrier": ("barrier",),
    "camera_alignment": ("camera", "meat"),
    "long_pipeline_delivery": ("shoe",),
    "take_photo": ("camera", "meat"),
    "pass_shoe": ("shoe",),
    "place_food": ("meat", "pot"),
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _bools(value: Sequence[Any], count: int) -> list[bool]:
    result = [bool(item) for item in value]
    if len(result) != count:
        raise ValueError(f"agent-indexed value has {len(result)} slots, expected {count}")
    return result


def _role_agent(snapshot: Mapping[str, Any], role: str) -> int:
    value = int(snapshot["role_agents"][role])
    count = int(snapshot["agent_count"])
    if value < 0 or value >= count:
        raise ValueError(f"role {role!r} has invalid agent slot {value}")
    return value


def _point(snapshot: Mapping[str, Any], group: str, key: str) -> tuple[float, float, float]:
    value = snapshot[group][key]
    if len(value) != 3:
        raise ValueError(f"{group}.{key} is not a 3D point")
    return (float(value[0]), float(value[1]), float(value[2]))


def _object_agents(
    snapshot: Mapping[str, Any], field: str, object_name: str
) -> list[int]:
    count = int(snapshot["agent_count"])
    flags = _bools(snapshot[field][object_name], count)
    return [index for index, active in enumerate(flags) if active]


def _controllers(snapshot: Mapping[str, Any], object_name: str) -> list[int]:
    grasping = _object_agents(snapshot, "grasp", object_name)
    if grasping:
        return grasping
    return _object_agents(snapshot, "contact", object_name)


def _single_custodian(snapshot: Mapping[str, Any], object_name: str) -> int | None:
    grasping = _object_agents(snapshot, "grasp", object_name)
    return grasping[0] if len(grasping) == 1 else None


def _approach_progress(
    snapshot: Mapping[str, Any], object_names: Sequence[str], agent_slots: Sequence[int]
) -> float:
    tcp_positions = snapshot["tcp_positions"]
    distances: list[float] = []
    for slot in agent_slots:
        tcp = tuple(float(value) for value in tcp_positions[slot])
        for object_name in object_names:
            distances.append(dist(tcp, _point(snapshot, "object_positions", object_name)))
    nearest = min(distances) if distances else 1.0
    return _clamp(1.0 - nearest / 0.50)


def initial_automaton_state(task: str) -> dict[str, Any]:
    if task not in TASK_STAGE_NAMES:
        raise KeyError(f"unknown SSC-V7 task {task!r}")
    return {
        "achieved_predicate_latches": {},
        "completed_goal_mask": {},
        "completed_handoff_mask": [],
        "custody_transfer_count": 0,
        "last_confirmed_custodian": {},
        "seen_custodians": [],
        "previous_stage_id": "approach",
        "declared_recovery_state_latch": False,
    }


def _copy_memory(task: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    memory = initial_automaton_state(task) if value is None else deepcopy(dict(value))
    required = set(initial_automaton_state(task))
    if set(memory) != required:
        raise ValueError("causal automaton memory keys differ from the frozen schema")
    return memory


def _update_common_memory(
    task: str, snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    ambiguity = AMBIGUITY_NONE
    if bool(snapshot.get("contact_threshold_ambiguous", False)):
        ambiguity |= AMBIGUITY_CONTACT_THRESHOLD
    for object_name in TASK_OBJECTS[task]:
        grasping = _object_agents(snapshot, "grasp", object_name)
        if len(grasping) == 1:
            memory["last_confirmed_custodian"][object_name] = grasping[0]
        elif len(grasping) > 1 and task not in {
            "lift_barrier",
            "camera_alignment",
            "take_photo",
        }:
            ambiguity |= AMBIGUITY_CONTENTION
    if bool(snapshot["environment_success"]):
        memory["completed_goal_mask"]["terminal"] = True
    return memory, ambiguity


def _agent_rows(
    snapshot: Mapping[str, Any], role_assignments: Mapping[int, Sequence[tuple[str, str]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = int(snapshot["agent_count"])
    contributions: list[dict[str, Any]] = []
    role_slots: list[dict[str, Any]] = []
    for slot in range(count):
        assignments = [
            {"object": object_name, "role": role}
            for object_name, role in role_assignments.get(slot, ())
        ]
        contacts = [
            object_name
            for object_name in TASK_OBJECTS[str(snapshot["task"])]
            if slot in _object_agents(snapshot, "contact", object_name)
        ]
        grasps = [
            object_name
            for object_name in TASK_OBJECTS[str(snapshot["task"])]
            if slot in _object_agents(snapshot, "grasp", object_name)
        ]
        contributions.append(
            {
                "agent_slot": slot,
                "active": bool(assignments or contacts or grasps),
                "roles": [item["role"] for item in assignments],
                "objects": sorted({item["object"] for item in assignments}),
                "contact_objects": contacts,
                "grasp_objects": grasps,
            }
        )
        for object_name in TASK_OBJECTS[str(snapshot["task"] )]:
            matches = [
                item["role"] for item in assignments if item["object"] == object_name
            ]
            role_slots.append(
                {
                    "agent_slot": slot,
                    "object": object_name,
                    "roles": matches or ["none"],
                }
            )
    return contributions, role_slots


def _custody_state(
    task: str, snapshot: Mapping[str, Any], memory: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for object_name in TASK_OBJECTS[task]:
        grasp_agents = _object_agents(snapshot, "grasp", object_name)
        contact_agents = _object_agents(snapshot, "contact", object_name)
        controllers = _controllers(snapshot, object_name)
        result[object_name] = {
            "grasp_agents": grasp_agents,
            "contact_agents": contact_agents,
            "controller_agents": controllers,
            "current_custodian": grasp_agents[0] if len(grasp_agents) == 1 else None,
            "shared_control": len(controllers) > 1,
            "last_confirmed_custodian": memory["last_confirmed_custodian"].get(
                object_name
            ),
        }
    return result


def _risk_state(task: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    contested = [
        object_name
        for object_name in TASK_OBJECTS[task]
        if len(_object_agents(snapshot, "grasp", object_name)) > 1
        and task not in {"lift_barrier", "camera_alignment", "take_photo"}
    ]
    dropped = [
        object_name
        for object_name in TASK_OBJECTS[task]
        if bool(snapshot.get("drop_risk", {}).get(object_name, False))
    ]
    return {
        "robot_collision": bool(snapshot["robot_collision"]),
        "robot_proximity_risk": bool(snapshot["robot_proximity_risk"]),
        "dropped_objects": dropped,
        "contested_objects": contested,
    }


def _lift_barrier(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]]]:
    supports = _controllers(snapshot, "barrier")
    dual = len(supports) >= 2
    success = bool(snapshot["environment_success"])
    base_z = float(snapshot["reference_heights"]["robot_base"])
    barrier_z = _point(snapshot, "object_positions", "barrier")[2]
    height_progress = _clamp((barrier_z - base_z) / 0.15)
    if success:
        stage, progress = "lifted", 1.0
    elif dual:
        stage, progress = "dual_support", height_progress
    elif supports:
        stage, progress = "single_support", 0.5
    else:
        stage = "approach"
        progress = _approach_progress(snapshot, ("barrier",), range(int(snapshot["agent_count"])))
    predicates = {
        "agent0_support": 0 in supports,
        "agent1_support": 1 in supports,
        "dual_support": dual,
        "barrier_lifted": success,
    }
    remaining = {"dual_support": not dual, "lift_height": not success}
    roles = {slot: [("barrier", "support")] for slot in supports}
    return stage, progress, remaining, predicates, roles


def _camera_alignment(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]]]:
    supports = _controllers(snapshot, "camera")
    meat_handlers = _object_agents(snapshot, "grasp", "meat")
    meat_controlled = bool(meat_handlers)
    controlled = len(supports) >= 2 and meat_controlled
    base_z = float(snapshot["reference_heights"]["robot_base"])
    camera_lifted = _point(snapshot, "object_positions", "camera")[2] > base_z + 0.20
    meat_lifted = _point(snapshot, "object_positions", "meat")[2] > base_z + 0.20
    success = bool(snapshot["environment_success"])
    if success:
        stage, progress = "camera_and_meat_lifted", 1.0
    elif controlled:
        stage = "camera_and_meat_controlled"
        progress = 0.5 * float(camera_lifted) + 0.5 * float(meat_lifted)
    elif supports or meat_handlers:
        stage = "partial_support_or_grasp"
        progress = _clamp((len(supports) / 2.0 + float(meat_controlled)) / 2.0)
    else:
        stage = "approach"
        progress = _approach_progress(snapshot, ("camera", "meat"), range(int(snapshot["agent_count"])))
    predicates = {
        "camera_support_count_ge_2": len(supports) >= 2,
        "meat_grasped": meat_controlled,
        "camera_lifted": camera_lifted,
        "meat_lifted": meat_lifted,
    }
    remaining = {
        "camera_dual_support": len(supports) < 2,
        "meat_control": not meat_controlled,
        "camera_lift": not camera_lifted,
        "meat_lift": not meat_lifted,
    }
    roles: dict[int, list[tuple[str, str]]] = {
        slot: [("camera", "support")] for slot in supports
    }
    for slot in meat_handlers:
        roles.setdefault(slot, []).append(("meat", "handler"))
    return stage, progress, remaining, predicates, roles


def _update_delivery_memory(
    snapshot: Mapping[str, Any], memory: dict[str, Any], logical_roles: Sequence[str]
) -> tuple[int | None, bool]:
    physical = [_role_agent(snapshot, name) for name in logical_roles]
    custodian = _single_custodian(snapshot, "shoe")
    wrong_order = False
    if custodian is not None:
        if custodian not in physical:
            wrong_order = True
        else:
            logical = physical.index(custodian)
            seen = [int(value) for value in memory["seen_custodians"]]
            if logical not in seen:
                expected = len(seen)
                if logical != expected:
                    wrong_order = True
                else:
                    seen.append(logical)
                    memory["seen_custodians"] = seen
                    if logical > 0:
                        mask = [bool(value) for value in memory["completed_handoff_mask"]]
                        while len(mask) < logical:
                            mask.append(False)
                        mask[logical - 1] = True
                        memory["completed_handoff_mask"] = mask
                        memory["custody_transfer_count"] = logical
    return custodian, wrong_order


def _long_pipeline(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]], bool]:
    logical_roles = ("pipeline_0", "pipeline_1", "pipeline_2", "pipeline_3")
    physical = [_role_agent(snapshot, name) for name in logical_roles]
    custodian, wrong_order = _update_delivery_memory(snapshot, memory, logical_roles)
    seen_count = len(memory["seen_custodians"])
    success = bool(snapshot["environment_success"])
    if success:
        stage, progress = "goal", 1.0
    elif seen_count == 0:
        stage = "approach"
        progress = _approach_progress(snapshot, ("shoe",), (physical[0],))
    else:
        current_logical = physical.index(custodian) if custodian in physical else None
        last_logical = seen_count - 1
        if current_logical == last_logical:
            stage = f"custody_{last_logical}"
            if last_logical == 3:
                target = _point(snapshot, "goal_positions", "goal")
            else:
                target = tuple(
                    (
                        float(snapshot["base_positions"][physical[last_logical]][axis])
                        + float(snapshot["base_positions"][physical[last_logical + 1]][axis])
                    )
                    / 2.0
                    for axis in range(3)
                )
            progress = _clamp(1.0 - dist(_point(snapshot, "object_positions", "shoe"), target) / 0.70)
        elif last_logical < 3:
            stage = f"handoff_{last_logical}{last_logical + 1}"
            next_tcp = tuple(float(value) for value in snapshot["tcp_positions"][physical[last_logical + 1]])
            progress = _clamp(1.0 - dist(next_tcp, _point(snapshot, "object_positions", "shoe")) / 0.50)
        else:
            stage, progress = "custody_3", 0.0
    predicates = {
        "current_custodian": custodian,
        "completed_handoff_mask": list(memory["completed_handoff_mask"]),
        "distance_to_goal": dist(
            _point(snapshot, "object_positions", "shoe"),
            _point(snapshot, "goal_positions", "goal"),
        ),
    }
    remaining = {
        "handoff_01": len(memory["completed_handoff_mask"]) < 1
        or not bool(memory["completed_handoff_mask"][0]),
        "handoff_12": len(memory["completed_handoff_mask"]) < 2
        or not bool(memory["completed_handoff_mask"][1]),
        "handoff_23": len(memory["completed_handoff_mask"]) < 3
        or not bool(memory["completed_handoff_mask"][2]),
        "goal": not success,
    }
    roles: dict[int, list[tuple[str, str]]] = {}
    if custodian is not None:
        roles[custodian] = [("shoe", "custodian")]
    if stage.startswith("handoff_"):
        next_slot = physical[min(seen_count, 3)]
        roles.setdefault(next_slot, []).append(("shoe", "receiver"))
    return stage, progress, remaining, predicates, roles, wrong_order


def _take_photo(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]]]:
    supports = _controllers(snapshot, "camera")
    meat_handlers = _object_agents(snapshot, "grasp", "meat")
    controlled = len(supports) >= 2 and bool(meat_handlers)
    button_agent = _role_agent(snapshot, "button")
    button_aligned = bool(snapshot["task_predicates"]["button_aligned"])
    success = bool(snapshot["environment_success"])
    if success:
        stage, progress = "photo_complete", 1.0
    elif controlled and button_aligned:
        stage, progress = "button_aligned", 1.0
    elif controlled:
        stage = "camera_and_meat_controlled"
        progress = _approach_progress(snapshot, ("camera",), (button_agent,))
    elif supports or meat_handlers:
        stage = "partial_support_or_grasp"
        progress = _clamp((len(supports) / 2.0 + float(bool(meat_handlers))) / 2.0)
    else:
        stage = "approach"
        progress = _approach_progress(snapshot, ("camera", "meat"), range(int(snapshot["agent_count"])))
    predicates = {
        "camera_support_count_ge_2": len(supports) >= 2,
        "meat_grasped": bool(meat_handlers),
        "button_aligned": button_aligned,
        "photo_complete": success,
    }
    remaining = {
        "camera_dual_support": len(supports) < 2,
        "meat_control": not bool(meat_handlers),
        "button_alignment": not button_aligned,
        "photo": not success,
    }
    roles: dict[int, list[tuple[str, str]]] = {
        slot: [("camera", "support")] for slot in supports
    }
    for slot in meat_handlers:
        roles.setdefault(slot, []).append(("meat", "handler"))
    roles.setdefault(button_agent, []).append(("camera", "button_operator"))
    return stage, progress, remaining, predicates, roles


def _pass_shoe(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]], bool]:
    logical_roles = ("source", "receiver")
    physical = [_role_agent(snapshot, name) for name in logical_roles]
    custodian, wrong_order = _update_delivery_memory(snapshot, memory, logical_roles)
    seen_count = len(memory["seen_custodians"])
    success = bool(snapshot["environment_success"])
    if success:
        stage, progress = "goal", 1.0
    elif seen_count == 0:
        stage = "approach"
        progress = _approach_progress(snapshot, ("shoe",), (physical[0],))
    elif seen_count == 1 and custodian == physical[0]:
        stage = "agent0_custody"
        midpoint = tuple(
            (
                float(snapshot["base_positions"][physical[0]][axis])
                + float(snapshot["base_positions"][physical[1]][axis])
            )
            / 2.0
            for axis in range(3)
        )
        progress = _clamp(1.0 - dist(_point(snapshot, "object_positions", "shoe"), midpoint) / 0.70)
    elif seen_count == 1:
        stage = "handoff"
        receiver_tcp = tuple(float(value) for value in snapshot["tcp_positions"][physical[1]])
        progress = _clamp(1.0 - dist(receiver_tcp, _point(snapshot, "object_positions", "shoe")) / 0.50)
    else:
        stage = "agent1_custody"
        progress = _clamp(
            1.0
            - dist(
                _point(snapshot, "object_positions", "shoe"),
                _point(snapshot, "goal_positions", "goal"),
            )
            / 0.70
        )
    handoff_complete = len(memory["completed_handoff_mask"]) >= 1 and bool(
        memory["completed_handoff_mask"][0]
    )
    predicates = {
        "current_custodian": custodian,
        "handoff_complete": handoff_complete,
        "distance_to_goal": dist(
            _point(snapshot, "object_positions", "shoe"),
            _point(snapshot, "goal_positions", "goal"),
        ),
    }
    remaining = {"handoff": not handoff_complete, "goal": not success}
    roles: dict[int, list[tuple[str, str]]] = {}
    if custodian is not None:
        roles[custodian] = [("shoe", "custodian")]
    if stage == "handoff":
        roles.setdefault(physical[1], []).append(("shoe", "receiver"))
    return stage, progress, remaining, predicates, roles, wrong_order


def _place_food(
    snapshot: Mapping[str, Any], memory: dict[str, Any]
) -> tuple[str, float, dict[str, bool], dict[str, Any], dict[int, list[tuple[str, str]]]]:
    meat_handlers = _object_agents(snapshot, "grasp", "meat")
    pot_handlers = _object_agents(snapshot, "grasp", "pot")
    meat_grasped = bool(meat_handlers)
    pot_grasped = bool(pot_handlers)
    joint_control = meat_grasped and pot_grasped
    base_z = float(snapshot["reference_heights"]["robot_base"])
    meat_lifted = _point(snapshot, "object_positions", "meat")[2] > base_z + 0.10
    pot_lifted = _point(snapshot, "object_positions", "pot")[2] > base_z + 0.10
    joint_lift = joint_control and meat_lifted and pot_lifted
    planar_distance = float(snapshot["task_predicates"]["planar_meat_to_pot_distance"])
    aligned = planar_distance < 0.10
    success = bool(snapshot["environment_success"])
    if success:
        stage, progress = "released", 1.0
    elif aligned:
        stage = "aligned"
        progress = float(not meat_grasped)
    elif joint_lift:
        stage = "joint_lift"
        progress = _clamp(1.0 - planar_distance / 0.50)
    elif joint_control:
        stage = "joint_control"
        progress = 0.5 * float(meat_lifted) + 0.5 * float(pot_lifted)
    elif meat_grasped or pot_grasped:
        stage = "partial_grasp"
        progress = 0.5
    else:
        stage = "approach"
        progress = _approach_progress(snapshot, ("meat", "pot"), range(int(snapshot["agent_count"])))
    predicates = {
        "meat_grasped": meat_grasped,
        "pot_grasped": pot_grasped,
        "joint_lift": joint_lift,
        "planar_alignment": aligned,
        "release_complete": success,
    }
    remaining = {
        "meat_grasp": not meat_grasped and not success,
        "pot_grasp": not pot_grasped and not success,
        "joint_lift": not joint_lift and not success,
        "alignment": not aligned,
        "release": not success,
    }
    roles: dict[int, list[tuple[str, str]]] = {}
    for slot in meat_handlers:
        roles.setdefault(slot, []).append(("meat", "handler"))
    for slot in pot_handlers:
        roles.setdefault(slot, []).append(("pot", "handler"))
    return stage, progress, remaining, predicates, roles


def build_oracle_label(
    snapshot: Mapping[str, Any], automaton_state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return one deterministic label and the updated past-causal memory."""

    task = str(snapshot["task"])
    if task not in TASK_STAGE_NAMES:
        raise KeyError(f"unknown SSC-V7 task {task!r}")
    count = int(snapshot["agent_count"])
    if count <= 0 or len(snapshot["tcp_positions"]) != count:
        raise ValueError("invalid agent slots in privileged snapshot")
    memory = _copy_memory(task, automaton_state)
    memory, ambiguity = _update_common_memory(task, snapshot, memory)
    wrong_order = False
    if task == "lift_barrier":
        stage, progress, remaining, predicates, roles = _lift_barrier(snapshot, memory)
    elif task == "camera_alignment":
        stage, progress, remaining, predicates, roles = _camera_alignment(snapshot, memory)
    elif task == "long_pipeline_delivery":
        stage, progress, remaining, predicates, roles, wrong_order = _long_pipeline(snapshot, memory)
    elif task == "take_photo":
        stage, progress, remaining, predicates, roles = _take_photo(snapshot, memory)
    elif task == "pass_shoe":
        stage, progress, remaining, predicates, roles, wrong_order = _pass_shoe(snapshot, memory)
    elif task == "place_food":
        stage, progress, remaining, predicates, roles = _place_food(snapshot, memory)
    else:  # pragma: no cover - guarded above
        raise AssertionError(task)
    if wrong_order:
        ambiguity |= AMBIGUITY_CUSTODY_ORDER
    if stage not in TASK_STAGE_NAMES[task]:
        raise ValueError(f"invalid stage {stage!r} for {task}")
    memory["declared_recovery_state_latch"] = bool(
        TASK_STAGE_NAMES[task].index(stage)
        < TASK_STAGE_NAMES[task].index(str(memory["previous_stage_id"]))
    )
    memory["previous_stage_id"] = stage
    memory["achieved_predicate_latches"].update(
        {key: True for key, value in predicates.items() if value is True}
    )
    contributions, role_slots = _agent_rows(snapshot, roles)
    role_valid = not bool(
        ambiguity
        & (AMBIGUITY_CONTACT_THRESHOLD | AMBIGUITY_CONTENTION | AMBIGUITY_CUSTODY_ORDER)
    )
    validity = {
        "stage_id": True,
        "within_stage_progress": True,
        "per_agent_contribution": role_valid,
        "remaining_goal_mask": True,
        "agent_object_role_slots": role_valid,
        "grasp_contact_custody_state": role_valid,
        "collision_drop_contention_risk": True,
        "causal_automaton_state": True,
    }
    return {
        "stage_id": stage,
        "within_stage_progress": float(_clamp(progress)),
        "per_agent_contribution": contributions,
        "remaining_goal_mask": {key: bool(value) for key, value in remaining.items()},
        "agent_object_role_slots": role_slots,
        "grasp_contact_custody_state": _custody_state(task, snapshot, memory),
        "collision_drop_contention_risk": _risk_state(task, snapshot),
        "causal_automaton_state": memory,
        "label_validity_mask": validity,
        "ambiguity_code": int(ambiguity),
        "factorized_predicates": predicates,
        "task_complete": bool(snapshot["environment_success"]),
    }


def permute_agent_slots(value: Mapping[str, Any], permutation: Sequence[int]) -> dict[str, Any]:
    """Rename old agent slots according to ``old -> new`` mapping."""

    count = int(value["agent_count"])
    mapping = [int(item) for item in permutation]
    if sorted(mapping) != list(range(count)):
        raise ValueError("agent permutation is not a bijection")
    result = deepcopy(dict(value))

    def reorder(items: Sequence[Any]) -> list[Any]:
        output: list[Any] = [None] * count
        for old, item in enumerate(items):
            output[mapping[old]] = deepcopy(item)
        return output

    for key in ("tcp_positions", "base_positions"):
        result[key] = reorder(value[key])
    for key in ("grasp", "contact", "contact_force"):
        result[key] = {
            object_name: reorder(items) for object_name, items in value[key].items()
        }
    result["role_agents"] = {
        role: mapping[int(slot)] for role, slot in value["role_agents"].items()
    }
    return result


def permute_automaton_state(
    value: Mapping[str, Any], permutation: Sequence[int]
) -> dict[str, Any]:
    mapping = [int(item) for item in permutation]
    result = deepcopy(dict(value))
    result["last_confirmed_custodian"] = {
        object_name: None if slot is None else mapping[int(slot)]
        for object_name, slot in value["last_confirmed_custodian"].items()
    }
    return result


def permute_label_slots(
    value: Mapping[str, Any], permutation: Sequence[int]
) -> dict[str, Any]:
    mapping = [int(item) for item in permutation]
    result = deepcopy(dict(value))
    for key in ("per_agent_contribution", "agent_object_role_slots"):
        for row in result[key]:
            row["agent_slot"] = mapping[int(row["agent_slot"])]
        result[key] = sorted(
            result[key], key=lambda row: (int(row["agent_slot"]), str(row.get("object", "")))
        )
    for object_state in result["grasp_contact_custody_state"].values():
        for key in ("grasp_agents", "contact_agents", "controller_agents"):
            object_state[key] = sorted(mapping[int(slot)] for slot in object_state[key])
        for key in ("current_custodian", "last_confirmed_custodian"):
            if object_state[key] is not None:
                object_state[key] = mapping[int(object_state[key])]
    memory = result["causal_automaton_state"]
    memory["last_confirmed_custodian"] = {
        object_name: None if slot is None else mapping[int(slot)]
        for object_name, slot in memory["last_confirmed_custodian"].items()
    }
    predicates = result["factorized_predicates"]
    renamed_predicates: dict[str, Any] = {}
    for key, predicate_value in predicates.items():
        if key.startswith("agent") and key.endswith("_support"):
            old_slot = int(key[len("agent") : -len("_support")])
            renamed_predicates[f"agent{mapping[old_slot]}_support"] = predicate_value
        else:
            renamed_predicates[key] = predicate_value
    result["factorized_predicates"] = renamed_predicates
    latches = memory["achieved_predicate_latches"]
    memory["achieved_predicate_latches"] = {
        (
            f"agent{mapping[int(key[len('agent') : -len('_support')])]}_support"
            if key.startswith("agent") and key.endswith("_support")
            else key
        ): predicate_value
        for key, predicate_value in latches.items()
    }
    predicates = result["factorized_predicates"]
    if isinstance(predicates.get("current_custodian"), int):
        predicates["current_custodian"] = mapping[predicates["current_custodian"]]
    return result


__all__ = [
    "AMBIGUITY_NONE",
    "TASK_OBJECTS",
    "TASK_STAGE_NAMES",
    "build_oracle_label",
    "initial_automaton_state",
    "permute_agent_slots",
    "permute_automaton_state",
    "permute_label_slots",
]
