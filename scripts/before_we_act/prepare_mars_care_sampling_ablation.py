#!/usr/bin/env python3
"""Freeze and evaluate the independent MARS CARE sampling ablation.

The frozen main manifest is read-only.  Event-aware anchors are selected from
the same episode/focal-arm rows and inside the same temporal strata, before any
branch outcome or closed-loop result exists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from before_we_act.care_training_data import family_targets
from before_we_act.mars_temporal_data import MARS_TASKS, PD_ACTION_HIGH, PD_ACTION_LOW


SPEC_FORMAT = "before-we-act.care-mars-event-aware-ablation-spec/1"
MANIFEST_FORMAT = "before-we-act.care-mars-family-manifest/1"
REPORT_FORMAT = "before-we-act.care-mars-sampling-ablation-report/1"
FINAL_FORMAT = "before-we-act.care-mars-sampling-ablation-final/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != SPEC_FORMAT:
        raise ValueError("wrong event-aware ablation spec")
    weights = value["sampling"]["feature_weights"]
    if set(weights) != {
        "visual_change",
        "action_transition",
        "joint_speed",
        "gripper_transition",
        "multi_arm_coordination",
    } or not np.isclose(sum(float(item) for item in weights.values()), 1.0):
        raise ValueError("event-aware feature weights must be complete and sum to one")
    return value


def empirical_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)
    return np.searchsorted(ordered, values, side="right") / max(1, len(values))


def moving_max(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("smoothing radius must be nonnegative")
    result = np.empty_like(values)
    for index in range(len(values)):
        result[index] = np.max(values[max(0, index - radius) : index + radius + 1])
    return result


def event_trace(
    family: Mapping[str, Any], visual_cache_root: Path, spec: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import h5py

    length = int(family["source_episode_length"])
    focal = int(family["focal_agent"])
    with h5py.File(family["source_episode_path"], "r") as source:
        group = source[str(family["source_trajectory"])]
        arms = sorted(
            int(key.rsplit("-", 1)[-1]) for key in group["actions"].keys()
        )
        actions = np.stack(
            [np.asarray(group[f"actions/panda-{arm}"][:length], np.float32) for arm in arms],
            axis=1,
        )
        qvel = np.stack(
            [np.asarray(group[f"obs/agent/panda-{arm}/qvel"][:length], np.float32) for arm in arms],
            axis=1,
        )
    cache_path = visual_cache_root / str(family["task"]) / f"{family['scenario_group_id']}.npz"
    with np.load(cache_path, allow_pickle=False) as source:
        visual = np.stack(
            [np.asarray(source[f"agent_{arm}"][:length], np.float32) for arm in arms],
            axis=1,
        )
    if actions.shape[0] != length or qvel.shape[0] != length or visual.shape[0] != length:
        raise RuntimeError(f"event feature length drift for {family['snapshot_id']}")

    joint_range = np.maximum(PD_ACTION_HIGH[:7] - PD_ACTION_LOW[:7], 1e-6)
    action_delta_by_arm = np.zeros((length, len(arms)), dtype=np.float64)
    action_delta_by_arm[1:] = np.linalg.norm(
        (actions[1:, :, :7] - actions[:-1, :, :7]) / joint_range[None, None, :],
        axis=2,
    )
    action_transition = action_delta_by_arm.mean(axis=1)
    joint_speed = np.linalg.norm(qvel[:, :, :7], axis=2).mean(axis=1)

    visual_change = np.zeros(length, dtype=np.float64)
    left = visual[:-1]
    right = visual[1:]
    denominator = np.linalg.norm(left, axis=2) * np.linalg.norm(right, axis=2)
    cosine = np.sum(left * right, axis=2) / np.maximum(denominator, 1e-8)
    visual_change[1:] = np.mean(np.clip(1.0 - cosine, 0.0, 2.0), axis=1)

    gripper_transition = np.zeros(length, dtype=np.float64)
    gripper_transition[1:] = np.max(
        np.abs(actions[1:, :, 7] - actions[:-1, :, 7]), axis=1
    )
    gripper_transition = (gripper_transition >= 0.5).astype(np.float64)

    per_arm_percentile = np.stack(
        [empirical_percentile(action_delta_by_arm[:, index]) for index in range(len(arms))],
        axis=1,
    )
    multi_arm_coordination = np.sort(per_arm_percentile, axis=1)[:, -2]
    features = {
        "visual_change": empirical_percentile(visual_change),
        "action_transition": empirical_percentile(action_transition),
        "joint_speed": empirical_percentile(joint_speed),
        "gripper_transition": gripper_transition,
        "multi_arm_coordination": multi_arm_coordination,
    }
    weights = spec["sampling"]["feature_weights"]
    score = sum(float(weights[name]) * features[name] for name in weights)
    score = moving_max(score, int(spec["sampling"]["smoothing_radius_steps"]))
    # The focal arm identity is deliberately not used for scoring.  All robots
    # share the same event location, while policy deployment remains local.
    if focal not in arms:
        raise ValueError("focal arm missing from source episode")
    return score, features


def phase_bin(
    stratum: str, ordinal: int, count: int, spec: Mapping[str, Any]
) -> tuple[float, float]:
    if count <= 0 or not 0 <= ordinal < count:
        raise ValueError("invalid within-stratum ordinal")
    key = "critical_phase_range" if stratum == "critical" else "uniform_phase_range"
    lower, upper = (float(value) for value in spec["sampling"][key])
    width = (upper - lower) / count
    return lower + ordinal * width, lower + (ordinal + 1) * width


def event_family(
    family: Mapping[str, Any], ordinal: int, count: int, visual_cache_root: Path,
    spec: Mapping[str, Any], spec_sha256: str,
) -> dict[str, Any]:
    stratum = str(family["sampling_stratum"])
    if stratum not in {"critical", "uniform"}:
        raise ValueError("event-aware ablation expects critical/uniform strata")
    maximum = max(1, int(family["source_episode_length"]) - 65)
    phase_lower, phase_upper = phase_bin(stratum, ordinal, count, spec)
    anchor_lower = max(1, int(np.ceil(phase_lower * maximum)))
    anchor_upper = min(maximum, int(np.floor(phase_upper * maximum)))
    if anchor_lower > anchor_upper:
        anchor_lower = anchor_upper = min(maximum, max(1, int(family["anchor_step"])))
    lead = int(spec["sampling"]["event_lead_steps"])
    score, features = event_trace(family, visual_cache_root, spec)
    event_lower = min(len(score) - 1, anchor_lower + lead)
    event_upper = min(len(score) - 1, anchor_upper + lead)
    candidates = np.arange(event_lower, event_upper + 1, dtype=np.int64)
    # np.argmax is a frozen earliest-index tie break.
    event_step = int(candidates[int(np.argmax(score[candidates]))])
    anchor = min(anchor_upper, max(anchor_lower, event_step - lead))
    parent = str(family["snapshot_id"])
    snapshot_id = hashlib.sha256(
        f"care-mars-event-aware-hybrid-v1|{parent}|{anchor}|{spec_sha256}".encode()
    ).hexdigest()
    result = dict(family)
    result.update(
        {
            "snapshot_id": snapshot_id,
            "anchor_step": anchor,
            "sampling_protocol": "event_aware_hybrid_v1",
            "parent_snapshot_id": parent,
            "parent_anchor_step": int(family["anchor_step"]),
            "event_selection": {
                "event_step": event_step,
                "event_lead_steps": lead,
                "event_score": float(score[event_step]),
                "event_features": {
                    name: float(value[event_step]) for name, value in features.items()
                },
                "phase_bin": [phase_lower, phase_upper],
                "anchor_search_steps": [anchor_lower, anchor_upper],
                "within_stratum_ordinal": ordinal,
                "within_stratum_count": count,
                "anchor_changed": anchor != int(family["anchor_step"]),
            },
        }
    )
    return result


def prepare_manifest(
    main_manifest_path: Path, visual_cache_root: Path, spec_path: Path,
    output: Path, smoke_output: Path,
) -> dict[str, Any]:
    if output.exists() or smoke_output.exists():
        raise RuntimeError("refusing to overwrite a frozen sampling ablation manifest")
    main = json.loads(main_manifest_path.read_text(encoding="utf-8"))
    if main.get("format_version") != MANIFEST_FORMAT or int(main.get("family_count", -1)) != 120:
        raise ValueError("sampling ablation requires the frozen 120-family main manifest")
    spec = load_spec(spec_path)
    spec_hash = sha256_file(spec_path)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for family in main["families"]:
        grouped[(str(family["task"]), str(family["sampling_stratum"]))].append(family)
    expected = {"critical": 20, "uniform": 10}
    families: list[dict[str, Any]] = []
    smoke_parent_ids: list[str] = []
    smoke_ordinals = spec["sampling"]["smoke_ordinals_within_stratum"]
    for task in MARS_TASKS:
        for stratum in ("critical", "uniform"):
            rows = grouped[(task, stratum)]
            if len(rows) != expected[stratum]:
                raise RuntimeError(f"main stratum drift: {task}/{stratum}/{len(rows)}")
            requested = {int(value) for value in smoke_ordinals[stratum]}
            if not requested.issubset(range(len(rows))):
                raise ValueError("smoke ordinal lies outside its frozen stratum")
            for ordinal, family in enumerate(rows):
                families.append(
                    event_family(family, ordinal, len(rows), visual_cache_root, spec, spec_hash)
                )
                if ordinal in requested:
                    smoke_parent_ids.append(str(family["snapshot_id"]))
    if len(families) != 120 or len(smoke_parent_ids) != 16:
        raise AssertionError((len(families), len(smoke_parent_ids)))
    result = {
        "format_version": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "status": "FROZEN_ABLATION",
        "families_per_task": 30,
        "family_count": 120,
        "branches_per_family": 24,
        "sampling": "event-aware ranking inside frozen critical/uniform temporal strata",
        "sampling_protocol": spec["name"],
        "promotion_scope": spec["promotion_scope"],
        "main_protocol_unchanged": True,
        "parent_main_manifest": str(main_manifest_path.resolve()),
        "parent_main_manifest_sha256": sha256_file(main_manifest_path),
        "ablation_spec": str(spec_path.resolve()),
        "ablation_spec_sha256": spec_hash,
        "smoke_parent_snapshot_ids": smoke_parent_ids,
        "families": families,
    }
    smoke_ids = set(smoke_parent_ids)
    smoke = dict(result)
    smoke["family_count"] = 16
    smoke["families_per_task"] = 4
    smoke["sampling"] += "; preregistered matched smoke subset"
    smoke["families"] = [
        row for row in families if str(row["parent_snapshot_id"]) in smoke_ids
    ]
    atomic_json(output, result)
    atomic_json(smoke_output, smoke)
    return {
        "status": "FROZEN",
        "manifest": str(output.resolve()),
        "manifest_sha256": sha256_file(output),
        "smoke_manifest": str(smoke_output.resolve()),
        "smoke_manifest_sha256": sha256_file(smoke_output),
        "families": 120,
        "smoke_families": 16,
        "changed_anchors": sum(bool(row["event_selection"]["anchor_changed"]) for row in families),
    }


def usable_quality(family: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for horizon in (8, 16, 32, 64):
        use = all(
            str(horizon) in branch.get("outcomes", {})
            and bool(branch.get("candidate_valid"))
            and not str(branch.get("status", "")).startswith("SIMULATOR_FATAL")
            for branch in family["branches"]
        )
        rows[str(horizon)] = {"label": "USE" if use else "DROP"}
    return {"horizons": rows}


def signal_counts(family: Mapping[str, Any], epsilon: float) -> dict[str, int | float]:
    target, unsafe, usable = family_targets(family, usable_quality(family))
    significant = units = signal_bearing_branches = branch_units = 0
    effective_pairs = pair_units = 0
    for horizon_index, use in enumerate(usable.tolist()):
        if not use:
            continue
        for repeat in (0, 1):
            values = target[horizon_index, :, repeat]
            for candidate in range(1, 6):
                # A hard-safety branch is not an effective signal: it can be
                # selected solely because the intervention is unsafe.  The
                # reference and candidate must both be safe for credit.
                if bool(unsafe[horizon_index, candidate, repeat]) or bool(
                    unsafe[horizon_index, 0, repeat]
                ):
                    continue
                branch_units += 1
                branch_signal = any(
                    abs(float(values[candidate, component])) >= epsilon
                    for component in range(3)
                )
                signal_bearing_branches += int(branch_signal)
                for component in range(3):
                    units += 1
                    significant += int(abs(float(values[candidate, component])) >= epsilon)
            for left in range(6):
                for right in range(left + 1, 6):
                    if bool(unsafe[horizon_index, left, repeat]) or bool(
                        unsafe[horizon_index, right, repeat]
                    ):
                        continue
                    pair_units += 1
                    effective_pairs += int(
                        any(
                            abs(float(values[left, component] - values[right, component]))
                            >= epsilon
                            for component in range(3)
                        )
                    )
                    # Effective pair count is a candidate-pair row, not a
                    # per-output-component count.
    return {
        "significant_branch_signals": significant,
        "branch_signal_units": units,
        "branch_signal_density": significant / units if units else 0.0,
        "signal_bearing_branches": signal_bearing_branches,
        "branch_units": branch_units,
        "signal_bearing_branch_density": (
            signal_bearing_branches / branch_units if branch_units else 0.0
        ),
        "effective_pairs": effective_pairs,
        "pair_units": pair_units,
        "effective_pair_density": effective_pairs / pair_units if pair_units else 0.0,
    }


def load_family(root: Path, task: str, snapshot_id: str) -> dict[str, Any]:
    path = root / task / f"{snapshot_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("branch_count", -1)) != 24:
        raise RuntimeError(f"incomplete branch family: {path}")
    return value


def bootstrap_mean_interval(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("bootstrap requires at least one matched row")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    significant = sum(int(row["significant_branch_signals"]) for row in rows)
    units = sum(int(row["branch_signal_units"]) for row in rows)
    signal_bearing = sum(int(row["signal_bearing_branches"]) for row in rows)
    branch_units = sum(int(row["branch_units"]) for row in rows)
    pairs = sum(int(row["effective_pairs"]) for row in rows)
    pair_units = sum(int(row["pair_units"]) for row in rows)
    return {
        "families": len(rows),
        "significant_branch_signals": significant,
        "branch_signal_units": units,
        "branch_signal_density": significant / units if units else 0.0,
        "signal_bearing_branches": signal_bearing,
        "branch_units": branch_units,
        "signal_bearing_branch_density": signal_bearing / branch_units if branch_units else 0.0,
        "effective_pairs": pairs,
        "pair_units": pair_units,
        "effective_pair_density": pairs / pair_units if pair_units else 0.0,
    }


def ratio(numerator: float, denominator: float) -> float | str:
    if denominator == 0:
        return "inf" if numerator > 0 else 1.0
    return numerator / denominator


def numeric_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator > 0 else 1.0
    return numerator / denominator


def sampling_report(
    main_root: Path, hybrid_root: Path, manifest_path: Path, spec_path: Path,
    output: Path, smoke_only: bool,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = load_spec(spec_path)
    gate = spec["signal_gate"]
    epsilon = float(gate["utility_epsilon"])
    smoke_ids = set(manifest["smoke_parent_snapshot_ids"])
    selected = [
        row for row in manifest["families"]
        if not smoke_only or str(row["parent_snapshot_id"]) in smoke_ids
    ]
    pairs: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        task = str(row["task"])
        parent_id = str(row["parent_snapshot_id"])
        main = signal_counts(load_family(main_root, task, parent_id), epsilon)
        hybrid = signal_counts(load_family(hybrid_root, task, str(row["snapshot_id"])), epsilon)
        pair = {
            "task": task,
            "parent_snapshot_id": parent_id,
            "hybrid_snapshot_id": str(row["snapshot_id"]),
            "main_anchor_step": int(row["parent_anchor_step"]),
            "hybrid_anchor_step": int(row["anchor_step"]),
            "event_score": float(row["event_selection"]["event_score"]),
            "main": main,
            "hybrid": hybrid,
            "density_delta": float(hybrid["signal_bearing_branch_density"]) - float(main["signal_bearing_branch_density"]),
        }
        pairs.append(pair)
        by_task[task].append(pair)
    main_total = aggregate_counts([row["main"] for row in pairs])
    hybrid_total = aggregate_counts([row["hybrid"] for row in pairs])
    density_delta = float(hybrid_total["signal_bearing_branch_density"]) - float(main_total["signal_bearing_branch_density"])
    density_ratio = numeric_ratio(
        float(hybrid_total["signal_bearing_branch_density"]), float(main_total["signal_bearing_branch_density"])
    )
    pair_ratio = numeric_ratio(
        float(hybrid_total["effective_pairs"]), float(main_total["effective_pairs"])
    )
    interval = bootstrap_mean_interval(
        [float(row["density_delta"]) for row in pairs],
        int(gate["bootstrap_samples"]), int(gate["bootstrap_seed"]),
    )
    task_rows: dict[str, Any] = {}
    task_regressions = []
    for task in MARS_TASKS:
        rows = by_task[task]
        main_task = aggregate_counts([row["main"] for row in rows])
        hybrid_task = aggregate_counts([row["hybrid"] for row in rows])
        delta = float(hybrid_task["signal_bearing_branch_density"]) - float(main_task["signal_bearing_branch_density"])
        task_rows[task] = {"main": main_task, "hybrid": hybrid_task, "density_delta": delta}
        task_regressions.append(delta)
    checks = {
        "absolute_density_gain": density_delta >= float(gate["minimum_absolute_density_gain"]),
        "relative_density_ratio": density_ratio >= float(gate["minimum_relative_density_ratio"]),
        "effective_pair_ratio": pair_ratio >= float(gate["minimum_effective_pair_ratio"]),
        "paired_bootstrap_lower_95": interval[0] > float(gate["minimum_paired_bootstrap_lower_95"]),
        "no_per_task_material_regression": min(task_regressions) >= -float(gate["maximum_per_task_density_regression"]),
    }
    result = {
        "format_version": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "scope": "preregistered_smoke" if smoke_only else "all_families",
        "matched_families": len(pairs),
        "utility_epsilon": epsilon,
        "main": main_total,
        "hybrid": hybrid_total,
        "branch_signal_density_delta": density_delta,
        "component_signal_density_delta": float(hybrid_total["branch_signal_density"]) - float(main_total["branch_signal_density"]),
        "branch_signal_density_ratio": ratio(
            float(hybrid_total["branch_signal_density"]), float(main_total["branch_signal_density"])
        ),
        "effective_pair_ratio": ratio(
            float(hybrid_total["effective_pairs"]), float(main_total["effective_pairs"])
        ),
        "paired_family_density_delta_bootstrap_95": list(interval),
        "tasks": task_rows,
        "preregistered_gate": dict(gate),
        "gate_checks": checks,
        "signal_gate_passed": all(checks.values()),
        "promotion_scope": "next_formal_run_only",
        "main_protocol_unchanged": True,
        "manifest_sha256": sha256_file(manifest_path),
        "spec_sha256": sha256_file(spec_path),
        "pairs": pairs,
    }
    atomic_json(output, result)
    return result


def validation_rows(root: Path) -> dict[tuple[str, int], bool]:
    rows: dict[tuple[str, int], bool] = {}
    for task in MARS_TASKS:
        value = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        if value.get("status") != "complete" or int(value.get("episodes", -1)) != 20:
            raise RuntimeError(f"incomplete Validation20: {root}/{task}")
        for row in value["rows"]:
            rows[(task, int(row["seed"]))] = bool(row["success"])
    if len(rows) != 80:
        raise RuntimeError(f"Validation20 must contain 80 task-seed rows, got {len(rows)}")
    return rows


def final_report(
    smoke_report_path: Path, main_validation_root: Path, hybrid_validation_root: Path,
    spec_path: Path, output: Path,
) -> dict[str, Any]:
    smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    spec = load_spec(spec_path)
    gate = spec["final_success_gate"]
    main = validation_rows(main_validation_root)
    hybrid = validation_rows(hybrid_validation_root)
    if set(main) != set(hybrid):
        raise RuntimeError("main/hybrid Validation20 task-seed support differs")
    keys = sorted(main)
    deltas = [float(hybrid[key]) - float(main[key]) for key in keys]
    interval = bootstrap_mean_interval(
        deltas, int(gate["bootstrap_samples"]), int(gate["bootstrap_seed"])
    )
    main_rate = float(np.mean([main[key] for key in keys]))
    hybrid_rate = float(np.mean([hybrid[key] for key in keys]))
    tasks: dict[str, Any] = {}
    task_deltas = []
    for task in MARS_TASKS:
        task_keys = [key for key in keys if key[0] == task]
        left = float(np.mean([main[key] for key in task_keys]))
        right = float(np.mean([hybrid[key] for key in task_keys]))
        tasks[task] = {"main_success_rate": left, "hybrid_success_rate": right, "delta": right - left}
        task_deltas.append(right - left)
    checks = {
        "smoke_signal_gate_passed": bool(smoke.get("signal_gate_passed")),
        "overall_success_rate_gain": hybrid_rate - main_rate >= float(gate["minimum_overall_success_rate_gain"]),
        "no_per_task_material_regression": min(task_deltas) >= -float(gate["maximum_per_task_success_rate_regression"]),
        "paired_bootstrap_lower_95": interval[0] >= float(gate["minimum_paired_bootstrap_lower_95"]),
    }
    result = {
        "format_version": FINAL_FORMAT,
        "created_at_utc": utc_now(),
        "episodes": len(keys),
        "main_successes": sum(int(main[key]) for key in keys),
        "hybrid_successes": sum(int(hybrid[key]) for key in keys),
        "main_success_rate": main_rate,
        "hybrid_success_rate": hybrid_rate,
        "success_rate_delta": hybrid_rate - main_rate,
        "paired_success_delta_bootstrap_95": list(interval),
        "tasks": tasks,
        "preregistered_gate": dict(gate),
        "promotion_checks": checks,
        "eligible_for_next_formal_main_protocol": all(checks.values()),
        "current_main_protocol_unchanged": True,
        "smoke_report_sha256": sha256_file(smoke_report_path),
        "spec_sha256": sha256_file(spec_path),
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--main-manifest", type=Path, required=True)
    manifest.add_argument("--visual-cache-root", type=Path, required=True)
    manifest.add_argument("--spec", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--smoke-output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--main-family-root", type=Path, required=True)
    report.add_argument("--hybrid-family-root", type=Path, required=True)
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--spec", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--smoke-only", action="store_true")
    final = sub.add_parser("final")
    final.add_argument("--smoke-report", type=Path, required=True)
    final.add_argument("--main-validation-root", type=Path, required=True)
    final.add_argument("--hybrid-validation-root", type=Path, required=True)
    final.add_argument("--spec", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        result = prepare_manifest(
            args.main_manifest, args.visual_cache_root, args.spec, args.output, args.smoke_output
        )
    elif args.command == "report":
        result = sampling_report(
            args.main_family_root, args.hybrid_family_root, args.manifest,
            args.spec, args.output, args.smoke_only,
        )
    else:
        result = final_report(
            args.smoke_report, args.main_validation_root, args.hybrid_validation_root,
            args.spec, args.output,
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
