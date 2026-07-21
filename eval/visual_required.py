"""Pure records and fail-closed acceptance for the Phase M0 visual suite."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


FORMAT_VERSION = "wam.multimodal.m0.visual_required/2"
ACCEPTANCE_FORMAT_VERSION = "wam.multimodal.m0.visual_acceptance/2"
CAMERA_ORDER = ("fixed", "robot_0_camera", "robot_1_camera")

STATE_ONLY_POLICY = "state_only"
SCRIPTED_ORACLE_POLICY = "privileged_scripted_oracle"
VISION_ORACLE_POLICY = "vision_oracle"
SHUFFLED_VISION_POLICY = "vision_oracle_opposite_rgb"
REQUIRED_POLICIES = (
    STATE_ONLY_POLICY,
    SCRIPTED_ORACLE_POLICY,
    VISION_ORACLE_POLICY,
    SHUFFLED_VISION_POLICY,
)
EXPECTED_ACTION_SOURCE = {
    STATE_ONLY_POLICY: "state_only",
    SCRIPTED_ORACLE_POLICY: "privileged_scripted_oracle",
    VISION_ORACLE_POLICY: "vision_oracle",
    SHUFFLED_VISION_POLICY: "vision_oracle",
}

_FORBIDDEN_POLICY_PATH_PARTS = {
    "privileged_state",
    "cue_id",
    "cue_value",
    "cue_variant",
    "rendered_cue_variant",
    "event_truth",
    "goal_truth",
    "obstacle_truth",
    "task_truth",
    "target_truth",
}


@dataclass(frozen=True)
class VisualRequiredEpisode:
    """One paired visual-required rollout with auditable policy inputs."""

    task_id: str
    cue_id: int
    physical_seed: int
    policy: str
    success: bool
    failure: bool
    failure_reason: str
    steps: int
    total_reward: float
    presented_observation_paths: tuple[str, ...]
    consumed_observation_paths: tuple[str, ...]
    privileged_observation_seen: bool
    rgb_source_cue_id: int | None = None
    rgb_mapping_key: str | None = None
    action_source: str | None = None
    initial_proprioception_sha256: str | None = None
    task_condition_sha256: str | None = None
    scene_id: str | None = None
    object_combination_id: str | None = None
    camera_order: tuple[str, ...] = ()
    all_view_frame_counts: Mapping[str, int] | None = None
    all_view_first_rgb_sha256: Mapping[str, str] | None = None
    all_view_last_rgb_sha256: Mapping[str, str] | None = None
    active_rgb_sha256: Mapping[str, str] | None = None
    pre_signal_frame_counts: Mapping[str, int] | None = None
    pre_signal_sequence_sha256: Mapping[str, str] | None = None
    camera_translation_travel_m: Mapping[str, float] | None = None
    fixed_extrinsics_max_abs_delta: float | None = None
    cross_camera_sync: bool = False
    renderer_backend: str | None = None
    geometry_source: str | None = None
    mujoco_version: str | None = None
    mujoco_gl: str | None = None
    model_xml_sha256: str | None = None
    raw_unannotated: bool = False
    cue_visible_expected: Mapping[str, bool] | None = None
    visual_signal_active_observed: bool = False
    visual_signal_onset_step: int | None = None
    visual_signal_kind: str | None = None
    policy_rgb_stream: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_leaf_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    """Return deterministic dotted leaf paths for a nested observation."""

    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                child = f"{path}.{key}" if path else str(key)
                visit(item[key], child)
            return
        paths.append(path)

    visit(value, prefix)
    return tuple(path for path in paths if path)


def contains_privileged_path(paths: Iterable[str]) -> bool:
    """Reject task truth even when it is nested under an innocuous root."""

    for path in paths:
        parts = {part.lower() for part in str(path).split(".")}
        if parts & _FORBIDDEN_POLICY_PATH_PARTS:
            return True
    return False


def opposite_cue_mapping(
    tasks: Sequence[str],
    physical_seeds: Sequence[int],
    cue_variants: Sequence[int],
) -> tuple[dict[str, int], str]:
    """Build a deterministic, no-fixed-point cue derangement and hash it."""

    cues = tuple(int(value) for value in cue_variants)
    if len(cues) < 2 or len(set(cues)) != len(cues):
        raise ValueError("cue_variants must contain at least two unique values")
    rotated = cues[1:] + cues[:1]
    mapping: dict[str, int] = {}
    for task in sorted({str(value) for value in tasks}):
        for seed in sorted({int(value) for value in physical_seeds}):
            for source, target in zip(cues, rotated, strict=True):
                if source == target:  # defensive even though rotation is non-trivial.
                    raise RuntimeError("opposite cue mapping contains a fixed point")
                mapping[mapping_key(task, seed, source)] = int(target)
    serialized = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return mapping, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def mapping_key(task_id: str, physical_seed: int, cue_id: int) -> str:
    return f"{task_id}:{int(physical_seed)}:{int(cue_id)}"


def visual_required_acceptance(
    records: Sequence[VisualRequiredEpisode | Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    physical_seeds: Sequence[int],
    cue_variants: Sequence[int],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the paired benchmark and apply every M0 control gate.

    The function is intentionally independent of MuJoCo and filesystem state so
    the final CLI and unit tests use exactly the same acceptance implementation.
    """

    normalized = tuple(_coerce_record(record) for record in records)
    required_tasks = tuple(str(task) for task in tasks)
    if not required_tasks or len(set(required_tasks)) != len(required_tasks):
        raise ValueError("tasks must be a non-empty unique sequence")
    seeds = tuple(int(seed) for seed in physical_seeds)
    cues = tuple(int(cue) for cue in cue_variants)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("physical_seeds must be a non-empty unique sequence")
    if len(cues) < 2 or len(set(cues)) != len(cues):
        raise ValueError("cue_variants must contain at least two unique values")

    expected_pairs = {(seed, cue) for seed in seeds for cue in cues}
    grouped: dict[tuple[str, str], list[VisualRequiredEpisode]] = defaultdict(list)
    unknown_records: list[dict[str, Any]] = []
    for record in normalized:
        if (
            record.task_id not in required_tasks
            or record.policy not in REQUIRED_POLICIES
        ):
            unknown_records.append(record.as_dict())
        grouped[(record.task_id, record.policy)].append(record)

    exact_pairs = True
    unique_pairs = True
    labels_valid = not unknown_records
    pair_details: dict[str, Any] = {}
    for task in required_tasks:
        for policy in REQUIRED_POLICIES:
            items = grouped.get((task, policy), [])
            pairs = [(item.physical_seed, item.cue_id) for item in items]
            unique = len(pairs) == len(set(pairs))
            exact = set(pairs) == expected_pairs
            exact_pairs = exact_pairs and exact
            unique_pairs = unique_pairs and unique
            pair_details[f"{task}/{policy}"] = {
                "records": len(items),
                "expected": len(expected_pairs),
                "unique": unique,
                "exact": exact,
                "missing": sorted(expected_pairs - set(pairs)),
                "unexpected": sorted(set(pairs) - expected_pairs),
            }
            labels_valid = labels_valid and all(
                item.failure is (not item.success)
                and item.steps > 0
                and math.isfinite(item.total_reward)
                for item in items
            )

    mapping, mapping_sha256 = opposite_cue_mapping(required_tasks, seeds, cues)
    mapping_valid = True
    for record in normalized:
        if record.policy != SHUFFLED_VISION_POLICY:
            mapping_valid = mapping_valid and record.rgb_source_cue_id in {
                None,
                record.cue_id,
            }
            continue
        key = mapping_key(record.task_id, record.physical_seed, record.cue_id)
        mapping_valid = mapping_valid and bool(
            key in mapping
            and record.rgb_source_cue_id == mapping[key]
            and record.rgb_source_cue_id != record.cue_id
            and record.rgb_mapping_key == key
        )

    action_source_violations = [
        {
            "task_id": record.task_id,
            "policy": record.policy,
            "physical_seed": record.physical_seed,
            "cue_id": record.cue_id,
            "actual": record.action_source,
            "expected": EXPECTED_ACTION_SOURCE.get(record.policy),
        }
        for record in normalized
        if record.action_source != EXPECTED_ACTION_SOURCE.get(record.policy)
    ]
    identity_violations: list[dict[str, Any]] = []
    identity_groups: dict[tuple[str, int], list[VisualRequiredEpisode]] = defaultdict(
        list
    )
    for record in normalized:
        identity_groups[(record.task_id, record.physical_seed)].append(record)
        values = {
            "initial_proprioception_sha256": record.initial_proprioception_sha256,
            "task_condition_sha256": record.task_condition_sha256,
            "scene_id": record.scene_id,
            "object_combination_id": record.object_combination_id,
        }
        invalid = [
            key
            for key, value in values.items()
            if not isinstance(value, str)
            or not value
            or (key.endswith("sha256") and not _is_sha256(value))
        ]
        if invalid:
            identity_violations.append(
                {
                    "task_id": record.task_id,
                    "physical_seed": record.physical_seed,
                    "cue_id": record.cue_id,
                    "policy": record.policy,
                    "invalid_fields": invalid,
                }
            )
    identity_group_details: dict[str, Any] = {}
    for (task, seed), items in sorted(identity_groups.items()):
        fields = {
            name: sorted({str(getattr(item, name)) for item in items})
            for name in (
                "initial_proprioception_sha256",
                "task_condition_sha256",
                "scene_id",
                "object_combination_id",
            )
        }
        consistent = all(len(values) == 1 for values in fields.values())
        if not consistent:
            identity_violations.append(
                {
                    "task_id": task,
                    "physical_seed": seed,
                    "inconsistent_fields": [
                        name for name, values in fields.items() if len(values) != 1
                    ],
                }
            )
        identity_group_details[f"{task}:{seed}"] = {
            "consistent": consistent,
            "values": fields,
        }
    path_contract = _observation_contract(normalized)
    render_contract = _render_evidence_contract(normalized)
    task_metrics = {
        task: _task_metrics(grouped, task, thresholds) for task in required_tasks
    }
    macro_metrics = _macro_metrics(normalized, thresholds)
    each_task_passed = all(item["passed"] for item in task_metrics.values())
    macro_passed = bool(macro_metrics["passed"])

    checks = {
        "record_labels_valid": _check(labels_valid, unknown=unknown_records[:20]),
        "unique_task_policy_seed_cue_records": _check(
            unique_pairs, details=pair_details
        ),
        "identical_complete_seed_cue_pairs": _check(exact_pairs, details=pair_details),
        "opposite_cue_derangement_valid": _check(
            mapping_valid,
            mapping_sha256=mapping_sha256,
            entries=len(mapping),
        ),
        "policy_action_sources": _check(
            not action_source_violations,
            violations=action_source_violations[:100],
            violation_count=len(action_source_violations),
            expected=dict(EXPECTED_ACTION_SOURCE),
        ),
        "identical_physical_seed_worlds_across_cue_and_policy": _check(
            not identity_violations,
            violations=identity_violations[:100],
            violation_count=len(identity_violations),
            groups=identity_group_details,
        ),
        "policy_observation_contract": _check(
            path_contract["passed"],
            **{key: value for key, value in path_contract.items() if key != "passed"},
        ),
        "mujoco_three_camera_render_evidence": _check(
            render_contract["passed"],
            **{key: value for key, value in render_contract.items() if key != "passed"},
        ),
        "every_task_passes": _check(each_task_passed),
        "macro_passes": _check(macro_passed),
    }
    return {
        "format_version": ACCEPTANCE_FORMAT_VERSION,
        "passed": all(item["passed"] for item in checks.values()),
        "tasks": list(required_tasks),
        "policies": list(REQUIRED_POLICIES),
        "physical_seeds": list(seeds),
        "cue_variants": list(cues),
        "expected_records": (
            len(required_tasks) * len(REQUIRED_POLICIES) * len(expected_pairs)
        ),
        "observed_records": len(normalized),
        "thresholds": _plain(thresholds),
        "opposite_cue_mapping": mapping,
        "opposite_cue_mapping_sha256": mapping_sha256,
        "observation_contract": path_contract,
        "render_evidence_contract": render_contract,
        "by_task": task_metrics,
        "macro": macro_metrics,
        "checks": checks,
    }


def _task_metrics(
    grouped: Mapping[tuple[str, str], Sequence[VisualRequiredEpisode]],
    task: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    rates = {
        policy: _rate(grouped.get((task, policy), ())) for policy in REQUIRED_POLICIES
    }
    drop = rates[VISION_ORACLE_POLICY]["rate"] - rates[SHUFFLED_VISION_POLICY]["rate"]
    checks = _rate_checks(rates, drop, thresholds)
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "policies": rates,
        "clean_minus_opposite_rgb_success_drop": drop,
        "checks": checks,
    }


def _macro_metrics(
    records: Sequence[VisualRequiredEpisode], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    rates = {
        policy: _rate(tuple(item for item in records if item.policy == policy))
        for policy in REQUIRED_POLICIES
    }
    drop = rates[VISION_ORACLE_POLICY]["rate"] - rates[SHUFFLED_VISION_POLICY]["rate"]
    checks = _rate_checks(rates, drop, thresholds)
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "policies": rates,
        "clean_minus_opposite_rgb_success_drop": drop,
        "checks": checks,
    }


def _rate_checks(
    rates: Mapping[str, Mapping[str, Any]],
    drop: float,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    state_limit = _threshold(thresholds, "maximum_state_only_success_rate", "<=")
    scripted_floor = _threshold(
        thresholds, "minimum_scripted_oracle_success_rate", ">="
    )
    vision_floor = _threshold(thresholds, "minimum_vision_oracle_success_rate", ">=")
    drop_floor = _threshold(thresholds, "minimum_opposite_rgb_success_drop", ">=")
    return {
        "state_only_below_ceiling": _threshold_check(
            rates[STATE_ONLY_POLICY]["rate"], state_limit
        ),
        "scripted_oracle_above_floor": _threshold_check(
            rates[SCRIPTED_ORACLE_POLICY]["rate"], scripted_floor
        ),
        "vision_oracle_above_floor": _threshold_check(
            rates[VISION_ORACLE_POLICY]["rate"], vision_floor
        ),
        "opposite_rgb_drop_above_floor": _threshold_check(drop, drop_floor),
    }


def _observation_contract(records: Sequence[VisualRequiredEpisode]) -> dict[str, Any]:
    presented: dict[str, set[str]] = defaultdict(set)
    consumed: dict[str, set[str]] = defaultdict(set)
    privileged_seen = False
    for record in records:
        presented[record.policy].update(record.presented_observation_paths)
        consumed[record.policy].update(record.consumed_observation_paths)
        privileged_seen = privileged_seen or record.privileged_observation_seen
        privileged_seen = privileged_seen or contains_privileged_path(
            record.presented_observation_paths
        )
        privileged_seen = privileged_seen or contains_privileged_path(
            record.consumed_observation_paths
        )

    common_required = {
        "proprioception",
        "task.id",
        "task.text",
        "past_executed_actions",
    }
    visual_required = common_required | {"images.fixed"}
    input_driven = {
        STATE_ONLY_POLICY,
        VISION_ORACLE_POLICY,
        SHUFFLED_VISION_POLICY,
    }
    violations: list[dict[str, Any]] = []
    nonvisual_presented_rgb: list[dict[str, Any]] = []
    for record in records:
        if record.policy in {STATE_ONLY_POLICY, SCRIPTED_ORACLE_POLICY}:
            presented_rgb = sorted(
                path
                for path in record.presented_observation_paths
                if path.startswith("images.")
            )
            if presented_rgb:
                nonvisual_presented_rgb.append(
                    {
                        "task_id": record.task_id,
                        "policy": record.policy,
                        "physical_seed": record.physical_seed,
                        "cue_id": record.cue_id,
                        "presented_rgb": presented_rgb,
                    }
                )
        if record.policy not in input_driven:
            continue
        record_consumed = set(record.consumed_observation_paths)
        record_presented = set(record.presented_observation_paths)
        required = (
            common_required if record.policy == STATE_ONLY_POLICY else visual_required
        )
        missing = sorted(required - record_consumed)
        unexpected_rgb = sorted(
            path
            for path in record_consumed
            if record.policy == STATE_ONLY_POLICY and path.startswith("images.")
        )
        unpresented = sorted(record_consumed - record_presented)
        if missing or unexpected_rgb or unpresented:
            violations.append(
                {
                    "task_id": record.task_id,
                    "policy": record.policy,
                    "physical_seed": record.physical_seed,
                    "cue_id": record.cue_id,
                    "missing": missing,
                    "unexpected_rgb": unexpected_rgb,
                    "consumed_but_not_presented": unpresented,
                }
            )

    state_records_valid = not any(
        item["policy"] == STATE_ONLY_POLICY for item in violations
    )
    clean_records_valid = not any(
        item["policy"] == VISION_ORACLE_POLICY for item in violations
    )
    shuffled_records_valid = not any(
        item["policy"] == SHUFFLED_VISION_POLICY for item in violations
    )
    all_records_reported = all(
        bool(record.consumed_observation_paths)
        for record in records
        if record.policy in input_driven
    )
    passed = bool(
        not privileged_seen
        and all_records_reported
        and state_records_valid
        and clean_records_valid
        and shuffled_records_valid
        and not violations
        and not nonvisual_presented_rgb
    )
    return {
        "passed": passed,
        "presented_leaf_paths": {
            policy: sorted(values) for policy, values in sorted(presented.items())
        },
        "consumed_leaf_paths": {
            policy: sorted(values) for policy, values in sorted(consumed.items())
        },
        "privileged_observation_seen": privileged_seen,
        "all_input_driven_records_reported_consumed_paths": all_records_reported,
        "required_common_consumed_paths": sorted(common_required),
        "required_visual_consumed_paths": sorted(visual_required),
        "state_only_records_satisfy_contract": state_records_valid,
        "vision_oracle_records_satisfy_contract": clean_records_valid,
        "shuffled_vision_records_satisfy_contract": shuffled_records_valid,
        "per_record_violations": violations[:100],
        "per_record_violation_count": len(violations),
        "nonvisual_policy_presented_rgb_violations": nonvisual_presented_rgb[:100],
        "nonvisual_policy_presented_rgb_violation_count": len(nonvisual_presented_rgb),
    }


def _render_evidence_contract(
    records: Sequence[VisualRequiredEpisode],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    cue_pairs: dict[tuple[str, int, str], list[VisualRequiredEpisode]] = defaultdict(
        list
    )
    mujoco_versions: set[str] = set()
    mujoco_gl_values: set[str] = set()
    model_xml_sha256: set[str] = set()
    for record in records:
        mappings = {
            "all_view_frame_counts": record.all_view_frame_counts,
            "all_view_first_rgb_sha256": record.all_view_first_rgb_sha256,
            "all_view_last_rgb_sha256": record.all_view_last_rgb_sha256,
            "active_rgb_sha256": record.active_rgb_sha256,
            "pre_signal_frame_counts": record.pre_signal_frame_counts,
            "pre_signal_sequence_sha256": record.pre_signal_sequence_sha256,
            "camera_translation_travel_m": record.camera_translation_travel_m,
            "cue_visible_expected": record.cue_visible_expected,
        }
        mapping_keys_valid = all(
            isinstance(value, Mapping) and tuple(value) == CAMERA_ORDER
            for value in mappings.values()
        )
        counts = record.all_view_frame_counts or {}
        pre_counts = record.pre_signal_frame_counts or {}
        hash_mappings = (
            record.all_view_first_rgb_sha256 or {},
            record.all_view_last_rgb_sha256 or {},
            record.active_rgb_sha256 or {},
            record.pre_signal_sequence_sha256 or {},
        )
        hashes_valid = all(
            _is_sha256(str(values.get(camera, "")))
            for values in hash_mappings
            for camera in CAMERA_ORDER
        )
        frame_counts_valid = bool(
            mapping_keys_valid
            and all(
                type(counts.get(camera)) is int and int(counts[camera]) > 0
                for camera in CAMERA_ORDER
            )
            and len({int(counts[camera]) for camera in CAMERA_ORDER}) == 1
            and all(
                type(pre_counts.get(camera)) is int and int(pre_counts[camera]) >= 0
                for camera in CAMERA_ORDER
            )
        )
        travel = record.camera_translation_travel_m or {}
        calibration_valid = bool(
            mapping_keys_valid
            and record.fixed_extrinsics_max_abs_delta is not None
            and math.isfinite(float(record.fixed_extrinsics_max_abs_delta))
            and float(record.fixed_extrinsics_max_abs_delta) <= 1e-6
            and all(
                math.isfinite(float(travel.get(camera, math.nan)))
                and float(travel[camera]) >= 0.001
                for camera in CAMERA_ORDER[1:]
            )
        )
        visibility = record.cue_visible_expected or {}
        expected_policy_stream = (
            "fixed"
            if record.policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
            else None
        )
        signal_timing_valid = bool(
            (
                record.task_id == "visual_event_stop"
                and type(record.visual_signal_onset_step) is int
                and int(record.visual_signal_onset_step) > 0
                and all(int(pre_counts.get(camera, 0)) > 0 for camera in CAMERA_ORDER)
            )
            or (
                record.task_id != "visual_event_stop"
                and record.visual_signal_onset_step == 0
            )
        )
        valid = bool(
            tuple(record.camera_order) == CAMERA_ORDER
            and frame_counts_valid
            and hashes_valid
            and calibration_valid
            and record.cross_camera_sync is True
            and record.renderer_backend == "mujoco.Renderer"
            and record.geometry_source == "mujoco_xml"
            and isinstance(record.mujoco_version, str)
            and bool(record.mujoco_version)
            and isinstance(record.mujoco_gl, str)
            and bool(record.mujoco_gl)
            and _is_sha256(str(record.model_xml_sha256 or ""))
            and record.raw_unannotated is True
            and all(visibility.get(camera) is True for camera in CAMERA_ORDER)
            and record.visual_signal_active_observed is True
            and type(record.visual_signal_onset_step) is int
            and int(record.visual_signal_onset_step) >= 0
            and signal_timing_valid
            and isinstance(record.visual_signal_kind, str)
            and bool(record.visual_signal_kind)
            and record.policy_rgb_stream == expected_policy_stream
        )
        if not valid:
            violations.append(
                {
                    "task_id": record.task_id,
                    "policy": record.policy,
                    "physical_seed": record.physical_seed,
                    "cue_id": record.cue_id,
                    "mapping_keys_valid": mapping_keys_valid,
                    "frame_counts_valid": frame_counts_valid,
                    "hashes_valid": hashes_valid,
                    "calibration_valid": calibration_valid,
                    "signal_timing_valid": signal_timing_valid,
                }
            )
        cue_pairs[(record.task_id, record.physical_seed, record.policy)].append(record)
        mujoco_versions.add(str(record.mujoco_version or ""))
        mujoco_gl_values.add(str(record.mujoco_gl or ""))
        model_xml_sha256.add(str(record.model_xml_sha256 or ""))

    pre_onset_pair_violations: list[dict[str, Any]] = []
    active_cue_pair_violations: list[dict[str, Any]] = []
    for (task_id, physical_seed, policy), pair in cue_pairs.items():
        cues = {item.cue_id for item in pair}
        cue_map = {item.cue_id: item for item in pair}
        active_diverges = bool(
            len(pair) == 2
            and cues == {0, 1}
            and all(
                str((cue_map[0].active_rgb_sha256 or {}).get(camera, ""))
                != str((cue_map[1].active_rgb_sha256 or {}).get(camera, ""))
                for camera in CAMERA_ORDER
            )
        )
        if not active_diverges:
            active_cue_pair_violations.append(
                {
                    "task_id": task_id,
                    "physical_seed": physical_seed,
                    "policy": policy,
                    "cues": sorted(cues),
                }
            )
        if task_id != "visual_event_stop":
            continue
        signatures = {
            tuple(
                (
                    camera,
                    int((item.pre_signal_frame_counts or {}).get(camera, -1)),
                    str((item.pre_signal_sequence_sha256 or {}).get(camera, "")),
                )
                for camera in CAMERA_ORDER
            )
            for item in pair
        }
        if len(pair) != 2 or cues != {0, 1} or len(signatures) != 1:
            pre_onset_pair_violations.append(
                {
                    "task_id": task_id,
                    "physical_seed": physical_seed,
                    "policy": policy,
                    "cues": sorted(cues),
                    "signature_count": len(signatures),
                }
            )
    provenance_consistent = bool(
        len(mujoco_versions) == 1
        and "" not in mujoco_versions
        and len(mujoco_gl_values) == 1
        and "" not in mujoco_gl_values
        and len(model_xml_sha256) == 1
        and all(_is_sha256(value) for value in model_xml_sha256)
    )
    return {
        "passed": (
            not violations
            and not pre_onset_pair_violations
            and not active_cue_pair_violations
            and provenance_consistent
        ),
        "camera_order": list(CAMERA_ORDER),
        "records": len(records),
        "mujoco_versions": sorted(mujoco_versions),
        "mujoco_gl": sorted(mujoco_gl_values),
        "model_xml_sha256": sorted(model_xml_sha256),
        "record_provenance_consistent": provenance_consistent,
        "violations": violations[:100],
        "violation_count": len(violations),
        "active_cue_pair_violations": active_cue_pair_violations[:100],
        "active_cue_pair_violation_count": len(active_cue_pair_violations),
        "event_pre_onset_pair_violations": pre_onset_pair_violations[:100],
        "event_pre_onset_pair_violation_count": len(pre_onset_pair_violations),
    }


def _rate(records: Sequence[VisualRequiredEpisode]) -> dict[str, Any]:
    successes = sum(record.success for record in records)
    total = len(records)
    return {
        "successes": successes,
        "episodes": total,
        "rate": successes / total if total else 0.0,
    }


def _threshold(
    thresholds: Mapping[str, Any], name: str, expected_operator: str
) -> dict[str, Any]:
    raw = thresholds.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"missing acceptance threshold {name!r}")
    operator = str(raw.get("operator", ""))
    value = raw.get("value")
    if operator != expected_operator or not isinstance(value, (int, float)):
        raise ValueError(f"invalid acceptance threshold {name!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite acceptance threshold {name!r}")
    return {"name": name, "operator": operator, "value": float(value)}


def _threshold_check(actual: float, threshold: Mapping[str, Any]) -> dict[str, Any]:
    operator = str(threshold["operator"])
    expected = float(threshold["value"])
    if operator == "<=":
        passed = actual <= expected or math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12
        )
    elif operator == "<":
        passed = actual < expected
    elif operator == ">=":
        passed = actual >= expected or math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12
        )
    elif operator == ">":
        passed = actual > expected
    else:  # pragma: no cover - guarded by _threshold in public flow.
        raise ValueError(f"unsupported threshold operator {operator!r}")
    return {
        "passed": bool(passed),
        "actual": float(actual),
        "operator": operator,
        "threshold": expected,
    }


def _coerce_record(
    record: VisualRequiredEpisode | Mapping[str, Any],
) -> VisualRequiredEpisode:
    if isinstance(record, VisualRequiredEpisode):
        return record
    payload = dict(record)
    for key in ("presented_observation_paths", "consumed_observation_paths"):
        payload[key] = tuple(str(value) for value in payload.get(key, ()))
    payload["camera_order"] = tuple(
        str(value) for value in payload.get("camera_order", ())
    )
    return VisualRequiredEpisode(**payload)


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    details.pop("passed", None)
    return {"passed": bool(passed), **details}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "ACCEPTANCE_FORMAT_VERSION",
    "CAMERA_ORDER",
    "FORMAT_VERSION",
    "EXPECTED_ACTION_SOURCE",
    "REQUIRED_POLICIES",
    "SCRIPTED_ORACLE_POLICY",
    "SHUFFLED_VISION_POLICY",
    "STATE_ONLY_POLICY",
    "VISION_ORACLE_POLICY",
    "VisualRequiredEpisode",
    "contains_privileged_path",
    "mapping_key",
    "observation_leaf_paths",
    "opposite_cue_mapping",
    "visual_required_acceptance",
]
