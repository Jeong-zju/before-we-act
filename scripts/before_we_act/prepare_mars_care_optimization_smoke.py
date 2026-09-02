#!/usr/bin/env python3
"""Freeze leakage-disjoint MARS CARE smoke families and audit branch support.

This utility is deliberately isolated from the completed formal MARS run.  The
manifest subcommand selects four development families per task directly from
the immutable 600-demonstration sidecars.  Selection never reads a branch
outcome or a closed-loop result: episodes are ranked by a fixed hash and
anchors are the centers of fixed temporal strata.

The branch-audit subcommand is fail closed.  Every serialized outcome must
report at least as many observed simulator steps as it requested; terminal
padding or a merely present horizon key is not accepted as evidence of full
support.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from before_we_act.mars_temporal_data import ARMS, ENV_DIR, MARS_TASKS
from deployment.mars_care.common import TASK_BY_NAME


MANIFEST_FORMAT = "before-we-act.care-mars-optimization-smoke-manifest/1"
BRANCH_AUDIT_FORMAT = "before-we-act.care-mars-branch-completeness-audit/1"
SELECTION_NAMESPACE = "care-mars-optimization-smoke-v1"
FAMILIES_PER_TASK = 4
STRATA = (
    ("critical", 0.35, 0.80, 2),
    ("uniform", 0.10, 0.85, 2),
)

# Frozen in configs/act/mars_control_full_data_v1.json in the baseline source
# tree and in the archived MARS ACT Validation20 receipts.  Keep this mapping
# task-specific: only place_cube_in_cup shares the 20260820 range.
ACT_VALIDATION_SEED_RANGES: dict[str, tuple[int, int]] = {
    "place_cube_in_cup": (20260820, 20260839),
    "strike_cube_hard": (20261820, 20261839),
    "three_robots_place_shoes": (20262820, 20262839),
    "four_robots_stack_cube": (20263820, 20263839),
}
CARE_VALIDATION_SEED_RANGE = (20260827, 20260846)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def inclusive_range(bounds: Sequence[int]) -> set[int]:
    if len(bounds) != 2:
        raise ValueError("seed range must contain exactly [start,end]")
    start, end = (int(value) for value in bounds)
    if start < 0 or end < start:
        raise ValueError("seed range must be non-negative and ordered")
    return set(range(start, end + 1))


def _sidecar_episode_seed(row: Mapping[str, Any]) -> int:
    reset = row.get("reset_kwargs", {})
    value = row.get("episode_seed")
    if value is None and isinstance(reset, Mapping):
        value = reset.get("seed")
    if value is None:
        raise ValueError("MARS sidecar episode is missing its reset seed")
    seed = int(value)
    if seed < 0:
        raise ValueError("MARS episode seed must be non-negative")
    return seed


def load_sidecar_episodes(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read exactly 150 successful sidecar rows per task without opening HDF5."""

    raw_root = raw_root.resolve(strict=True)
    episodes: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for task in MARS_TASKS:
        directory = raw_root / ENV_DIR[task] / "motionplanning"
        paths = sorted(directory.glob(f"{task}*.json"))
        if not paths:
            raise FileNotFoundError(f"no MARS sidecars for {task}: {directory}")
        task_seeds: set[int] = set()
        for sidecar_path in paths:
            h5_path = sidecar_path.with_suffix(".h5")
            if not h5_path.is_file():
                raise FileNotFoundError(f"sidecar has no paired HDF5: {sidecar_path}")
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            rows = payload.get("episodes")
            if not isinstance(rows, list):
                raise ValueError(f"MARS sidecar has no episode list: {sidecar_path}")
            sidecars.append(
                {
                    "task": task,
                    "path": str(sidecar_path.resolve()),
                    "sha256": sha256_file(sidecar_path),
                    "episode_count": len(rows),
                }
            )
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError(f"non-object episode row in {sidecar_path}")
                seed = _sidecar_episode_seed(row)
                if seed in task_seeds:
                    raise ValueError(f"duplicate episode seed for {task}: {seed}")
                task_seeds.add(seed)
                episode_id = int(row["episode_id"])
                length = int(row.get("elapsed_steps", 0))
                if length < 66:
                    raise ValueError(
                        f"MARS episode cannot support a 64-step branch: "
                        f"{sidecar_path}:traj_{episode_id} length={length}"
                    )
                if not bool(row.get("success", False)):
                    raise ValueError(
                        f"MARS formal sidecar contains a failed demo: "
                        f"{sidecar_path}:traj_{episode_id}"
                    )
                trajectory = f"traj_{episode_id}"
                identity = f"{h5_path.resolve()}:{trajectory}:{length}".encode()
                episodes.append(
                    {
                        "task": task,
                        "episode_seed": seed,
                        "source_episode_path": str(h5_path.resolve()),
                        "source_sidecar_path": str(sidecar_path.resolve()),
                        "source_trajectory": trajectory,
                        "source_episode_length": length,
                        "scenario_group_id": hashlib.sha256(identity).hexdigest(),
                        "arms": tuple(range(ARMS[task])),
                    }
                )
                counts[task] += 1
        if len(task_seeds) != 150:
            raise ValueError(
                f"MARS smoke source must contain 150 unique seeds for {task}, "
                f"got {len(task_seeds)}"
            )
    if counts != Counter({task: 150 for task in MARS_TASKS}) or len(episodes) != 600:
        raise ValueError(f"MARS smoke source must be exactly 600 demos: {counts}")
    return episodes, sidecars


def load_formal_family_seeds(path: Path) -> tuple[dict[str, set[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("families")
    if not isinstance(rows, list):
        raise ValueError("old formal CARE manifest has no families")
    result = {task: set() for task in MARS_TASKS}
    for row in rows:
        task = str(row.get("task"))
        if task not in result:
            raise ValueError(f"unknown task in old CARE manifest: {task}")
        result[task].add(int(row["episode_seed"]))
    missing = [task for task in MARS_TASKS if not result[task]]
    if missing:
        raise ValueError(f"old formal CARE manifest omits tasks: {missing}")
    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "format_version": payload.get("format_version"),
        "family_count": len(rows),
        "episode_seed_count_by_task": {
            task: len(result[task]) for task in MARS_TASKS
        },
    }
    return result, provenance


def exclusion_sets(
    formal_seeds: Mapping[str, set[int]],
) -> dict[str, dict[str, set[int]]]:
    # Treat the old CARE family seed namespace as global.  Although the
    # benchmark task is part of the simulator identity, reusing a seed from a
    # different task is still needlessly ambiguous when a smoke corpus is
    # compared against archived families.  A global exclusion is therefore
    # the conservative, auditable interpretation of "no old family seed".
    global_formal = set().union(*(set(values) for values in formal_seeds.values()))
    care_validation = inclusive_range(CARE_VALIDATION_SEED_RANGE)
    result: dict[str, dict[str, set[int]]] = {}
    for task in MARS_TASKS:
        if task not in formal_seeds or task not in ACT_VALIDATION_SEED_RANGES:
            raise ValueError(f"missing seed exclusion contract for {task}")
        result[task] = {
            "old_formal_care_family_episode_seed": set(global_formal),
            "old_care_validation20_seed": set(care_validation),
            "act_validation20_seed": inclusive_range(
                ACT_VALIDATION_SEED_RANGES[task]
            ),
        }
    return result


def _rank_episode(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{SELECTION_NAMESPACE}|{row['task']}|"
            f"{row['scenario_group_id']}|{row['episode_seed']}"
        ).encode()
    ).hexdigest()


def _anchor(
    row: Mapping[str, Any], *, lower: float, upper: float, ordinal: int, count: int
) -> tuple[int, float, tuple[float, float]]:
    if count <= 0 or not 0 <= ordinal < count or not 0.0 <= lower < upper <= 1.0:
        raise ValueError("invalid fixed temporal stratum")
    bin_lower = lower + (upper - lower) * ordinal / count
    bin_upper = lower + (upper - lower) * (ordinal + 1) / count
    phase = 0.5 * (bin_lower + bin_upper)
    spec = TASK_BY_NAME[str(row["task"])]
    maximum = max(
        1,
        min(
            int(row["source_episode_length"]) - 65,
            int(spec.max_steps) - 65,
        ),
    )
    anchor = min(maximum, max(1, int(round(phase * maximum))))
    return anchor, phase, (bin_lower, bin_upper)


def build_smoke_manifest(
    episodes: Sequence[Mapping[str, Any]],
    *,
    formal_seeds: Mapping[str, set[int]],
    formal_provenance: Mapping[str, Any],
    raw_root: Path,
    sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select 2 critical + 2 uniform families per task before outcomes exist."""

    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        task = str(row["task"])
        if task not in MARS_TASKS:
            raise ValueError(f"unknown MARS smoke source task: {task}")
        by_task[task].append(row)
    exclusions = exclusion_sets(formal_seeds)
    families: list[dict[str, Any]] = []
    exclusion_report: dict[str, Any] = {}
    for task in MARS_TASKS:
        source = by_task[task]
        if len(source) != 150:
            raise ValueError(f"MARS smoke selection expects 150 demos for {task}")
        reason_counts = {
            reason: sum(int(row["episode_seed"]) in seeds for row in source)
            for reason, seeds in exclusions[task].items()
        }
        union = set().union(*exclusions[task].values())
        eligible = [row for row in source if int(row["episode_seed"]) not in union]
        eligible.sort(key=_rank_episode)
        if len(eligible) < FAMILIES_PER_TASK:
            raise ValueError(f"not enough leakage-disjoint demos for {task}")
        selected = eligible[:FAMILIES_PER_TASK]
        exclusion_report[task] = {
            "source_demo_count": len(source),
            "excluded_by_reason": reason_counts,
            "excluded_unique": len(source) - len(eligible),
            "eligible": len(eligible),
            "selected": len(selected),
        }
        family_ordinal = 0
        for stratum, lower, upper, stratum_count in STRATA:
            for within_stratum in range(stratum_count):
                source_row = selected[family_ordinal]
                focal = family_ordinal % len(source_row["arms"])
                anchor, phase, phase_bin = _anchor(
                    source_row,
                    lower=lower,
                    upper=upper,
                    ordinal=within_stratum,
                    count=stratum_count,
                )
                seed = int(source_row["episode_seed"])
                snapshot_id = hashlib.sha256(
                    (
                        f"{SELECTION_NAMESPACE}|{task}|{seed}|{anchor}|{focal}|"
                        f"{stratum}|{source_row['scenario_group_id']}"
                    ).encode()
                ).hexdigest()
                families.append(
                    {
                        "snapshot_id": snapshot_id,
                        "task": task,
                        "episode_seed": seed,
                        "anchor_step": anchor,
                        "focal_agent": focal,
                        "sampling_stratum": stratum,
                        "sampling_protocol": "fixed_stratified_smoke_v1",
                        "fixed_phase": phase,
                        "fixed_phase_bin": list(phase_bin),
                        "within_stratum_ordinal": within_stratum,
                        "scenario_group_id": source_row["scenario_group_id"],
                        "source_episode_path": source_row["source_episode_path"],
                        "source_sidecar_path": source_row["source_sidecar_path"],
                        "source_trajectory": source_row["source_trajectory"],
                        "source_episode_length": int(
                            source_row["source_episode_length"]
                        ),
                    }
                )
                family_ordinal += 1

        task_rows = [row for row in families if row["task"] == task]
        focal_counts = Counter(int(row["focal_agent"]) for row in task_rows)
        all_arm_counts = [focal_counts[arm] for arm in range(ARMS[task])]
        if max(all_arm_counts) - min(all_arm_counts) > 1:
            raise AssertionError(f"unbalanced focal arms for {task}: {focal_counts}")
        if Counter(row["sampling_stratum"] for row in task_rows) != Counter(
            {"critical": 2, "uniform": 2}
        ):
            raise AssertionError(f"smoke strata drift for {task}")

    sidecar_rows = sorted(
        (dict(row) for row in sidecars), key=lambda row: (row["task"], row["path"])
    )
    return {
        "format_version": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "status": "FROZEN",
        "purpose": "independent development/closed-loop smoke only",
        "selection_namespace": SELECTION_NAMESPACE,
        "source_demo_count": len(episodes),
        "families_per_task": FAMILIES_PER_TASK,
        "family_count": len(families),
        "branches_per_family": 24,
        "sampling": {
            "protocol": "fixed stratified; no branch/closed-loop outcome read",
            "strata": [
                {
                    "name": name,
                    "phase_range": [lower, upper],
                    "families_per_task": count,
                    "anchor": "center of equal-width phase bin",
                }
                for name, lower, upper, count in STRATA
            ],
            "episode_rank": "sha256(namespace,task,scenario_group_id,episode_seed)",
            "outcome_fields_used_for_selection": [],
            "focal_assignment": "family ordinal modulo task arm count",
            "focal_balance_max_count_difference": 1,
        },
        "seed_exclusions": {
            "old_formal_care_manifest": dict(formal_provenance),
            "old_care_validation20": {
                "inclusive_range": list(CARE_VALIDATION_SEED_RANGE)
            },
            "act_validation20_by_task": {
                task: {"inclusive_range": list(ACT_VALIDATION_SEED_RANGES[task])}
                for task in MARS_TASKS
            },
            "counts_by_task": exclusion_report,
            "formal_family_seed_scope": "global_union_across_tasks",
        },
        "source_sidecars": {
            "raw_root": str(raw_root.resolve()),
            "count": len(sidecar_rows),
            "canonical_sha256": canonical_sha256(sidecar_rows),
            "files": sidecar_rows,
        },
        "families": families,
    }


def branch_completeness_report(
    family_paths: Iterable[Path], *, expected_families: int | None = None
) -> dict[str, Any]:
    """Audit serialized outcomes; no short or terminal-padded label is complete."""

    paths = sorted(Path(path) for path in family_paths)
    if expected_families is not None and len(paths) != int(expected_families):
        raise ValueError(
            f"branch audit expected {expected_families} families, got {len(paths)}"
        )
    if not paths:
        raise ValueError("branch audit requires at least one family")
    issues: list[dict[str, Any]] = []
    branch_count = 0
    outcome_count = 0
    family_rows: list[dict[str, Any]] = []
    for path in paths:
        family = json.loads(path.read_text(encoding="utf-8"))
        branches = family.get("branches")
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"family has no branches: {path}")
        snapshot_id = str(family.get("snapshot_id", path.stem))
        current_outcomes = 0
        for branch_index, branch in enumerate(branches):
            outcomes = branch.get("outcomes")
            if not isinstance(outcomes, Mapping) or not outcomes:
                issues.append(
                    {
                        "snapshot_id": snapshot_id,
                        "branch_index": branch_index,
                        "reason": "missing_outcomes",
                    }
                )
                continue
            branch_count += 1
            for horizon, outcome in outcomes.items():
                outcome_count += 1
                current_outcomes += 1
                if not isinstance(outcome, Mapping):
                    issues.append(
                        {
                            "snapshot_id": snapshot_id,
                            "branch_index": branch_index,
                            "horizon": str(horizon),
                            "reason": "non_object_outcome",
                        }
                    )
                    continue
                requested = int(outcome.get("requested_steps", -1))
                observed = int(outcome.get("observed_steps", -1))
                if requested < 1 or int(horizon) != requested:
                    issues.append(
                        {
                            "snapshot_id": snapshot_id,
                            "branch_index": branch_index,
                            "horizon": str(horizon),
                            "requested_steps": requested,
                            "observed_steps": observed,
                            "reason": "requested_steps_contract_mismatch",
                        }
                    )
                elif observed < requested:
                    issues.append(
                        {
                            "snapshot_id": snapshot_id,
                            "branch_index": branch_index,
                            "horizon": str(horizon),
                            "requested_steps": requested,
                            "observed_steps": observed,
                            "reason": "observed_steps_shorter_than_requested",
                        }
                    )
        family_rows.append(
            {
                "snapshot_id": snapshot_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "branches": len(branches),
                "outcomes": current_outcomes,
            }
        )
    return {
        "format_version": BRANCH_AUDIT_FORMAT,
        "created_at_utc": utc_now(),
        "status": "PASSED" if not issues else "FAILED",
        "requirement": "every outcome observed_steps >= requested_steps",
        "family_count": len(paths),
        "branch_count_with_outcomes": branch_count,
        "outcome_count": outcome_count,
        "issue_count": len(issues),
        "issues": issues,
        "families": family_rows,
    }


def _family_json_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for task in MARS_TASKS
        for path in (root / task).glob("*.json")
        if not path.name.endswith(".quality.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--raw-root", type=Path, required=True)
    manifest.add_argument("--old-formal-manifest", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit-branches")
    audit.add_argument("--family-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-families", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen output: {args.output}")
    if args.command == "manifest":
        episodes, sidecars = load_sidecar_episodes(args.raw_root)
        formal_seeds, formal_provenance = load_formal_family_seeds(
            args.old_formal_manifest
        )
        result = build_smoke_manifest(
            episodes,
            formal_seeds=formal_seeds,
            formal_provenance=formal_provenance,
            raw_root=args.raw_root,
            sidecars=sidecars,
        )
    else:
        result = branch_completeness_report(
            _family_json_paths(args.family_root),
            expected_families=args.expected_families,
        )
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result.get("status") == "FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
