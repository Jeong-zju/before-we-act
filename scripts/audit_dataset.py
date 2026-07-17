"""Audit every integrity requirement for a proprioceptive WAM dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
    )
except ImportError:  # --no-progress remains available in minimal environments.
    Console = None
    Progress = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from policies.collection import BEHAVIOR_WEIGHTS  # noqa: E402
from train.trajectory_dataset import (  # noqa: E402
    ProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)


REQUIRED_DATASETS = (
    "data/observation/state",
    "data/commanded_action",
    "data/executed_action",
    "data/next_observation/state",
    "data/reward",
    "data/done",
    "data/success",
    "data/failure",
    "data/behavior_id",
    "data/perturbation_config",
    "data/environment_config",
    "data/randomization_config",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail unless a wam.proprio/1.0 dataset passes the full integrity contract."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=-1)
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--mixture-tolerance", type=float, default=0.02)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_episodes == 0 or args.expected_episodes < -1:
        raise ValueError("expected_episodes must be -1 or positive")
    if args.state_dim <= 0 or args.action_dim <= 0:
        raise ValueError("state_dim and action_dim must be positive")
    if not 0.0 <= args.mixture_tolerance < 1.0:
        raise ValueError("mixture_tolerance must be in [0,1)")

    paths = discover_episode_paths(args.data_dir)
    errors: list[str] = []
    behavior_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    behavior_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    episode_seeds: dict[Path, int] = {}
    total_transitions = 0

    progress = _progress(len(paths), enabled=not args.no_progress)
    if progress is not None:
        progress.start()
        task_id = progress.add_task("audit episodes", total=len(paths))
    else:
        task_id = None
    try:
        for path in paths:
            try:
                episode = _audit_episode(
                    path,
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                errors.append(f"{path}: {exc}")
            else:
                total_transitions += episode["steps"]
                behavior_counts[episode["behavior_id"]] += 1
                outcome_counts[episode["outcome"]] += 1
                behavior_outcomes[episode["behavior_id"]][episode["outcome"]] += 1
                episode_seeds[path] = episode["seed"]
            if progress is not None and task_id is not None:
                progress.advance(task_id)
    finally:
        if progress is not None:
            progress.stop()

    checks: dict[str, dict[str, Any]] = {}
    checks["episode_count"] = _check(
        args.expected_episodes < 0 or len(paths) == args.expected_episodes,
        actual=len(paths),
        expected=args.expected_episodes,
    )
    checks["all_episode_contracts"] = _check(
        not errors,
        error_count=len(errors),
        errors=errors[:50],
    )
    checks["success_and_failure"] = _check(
        outcome_counts["success"] > 0
        and outcome_counts["failure"] > 0
        and outcome_counts["incomplete"] == 0,
        outcomes=dict(sorted(outcome_counts.items())),
    )
    checks["behavior_coverage"] = _behavior_check(
        behavior_counts,
        episode_count=len(paths),
        tolerance=args.mixture_tolerance,
    )
    split_check = _split_check(paths, episode_seeds, split_seed=args.split_seed)
    checks["episode_seed_split"] = split_check
    checks["sequence_padding"] = _sequence_check(
        paths,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )

    passed = all(item["passed"] for item in checks.values())
    report = {
        "gate": "A",
        "passed": passed,
        "schema_version": PROPRIO_WAM_SCHEMA_VERSION,
        "episodes": len(paths),
        "transitions": total_transitions,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "behavior_outcomes": {
            key: dict(sorted(value.items()))
            for key, value in sorted(behavior_outcomes.items())
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


def _audit_episode(path: Path, *, state_dim: int, action_dim: int) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        if str(file.attrs.get("schema_profile", "")) != "wam_proprio":
            raise ValueError("schema_profile is not wam_proprio")
        if str(file.attrs.get("schema_version", "")) != PROPRIO_WAM_SCHEMA_VERSION:
            raise ValueError("schema_version is not wam.proprio/1.0")
        missing = [name for name in REQUIRED_DATASETS if name not in file]
        if missing:
            raise KeyError(f"missing datasets: {missing}")
        all_paths: list[str] = []
        file.visit(all_paths.append)
        forbidden = [
            name for name in all_paths if "image" in name.lower() or "privileged" in name.lower()
        ]
        if forbidden:
            raise ValueError(f"contains forbidden image/privileged paths: {forbidden[:5]}")

        steps = int(file.attrs.get("num_steps", -1))
        expected_shapes = {
            "data/observation/state": (steps, state_dim),
            "data/commanded_action": (steps, action_dim),
            "data/executed_action": (steps, action_dim),
            "data/next_observation/state": (steps, state_dim),
        }
        for name, expected in expected_shapes.items():
            if tuple(file[name].shape) != expected:
                raise ValueError(f"{name} has shape {file[name].shape}, expected {expected}")
            if not np.isfinite(file[name][:]).all():
                raise ValueError(f"{name} contains NaN/Inf")
        for name in REQUIRED_DATASETS:
            if int(file[name].shape[0]) != steps:
                raise ValueError(f"{name} length does not match num_steps")

        behavior_id = str(file.attrs.get("behavior_id", ""))
        if behavior_id not in dict(BEHAVIOR_WEIGHTS):
            raise ValueError(f"unknown behavior_id {behavior_id!r}")
        frame_behaviors = np.asarray(file["data/behavior_id"].asstr()[:])
        if not np.all(frame_behaviors == behavior_id):
            raise ValueError("behavior_id changes inside the episode")
        for name in (
            "data/perturbation_config",
            "data/environment_config",
            "data/randomization_config",
        ):
            for value in file[name].asstr()[:1]:
                json.loads(value)

        success = bool(np.asarray(file["data/success"][-1]).item())
        failure = bool(np.asarray(file["data/failure"][-1]).item())
        if success and failure:
            raise ValueError("final transition cannot be both success and failure")
        outcome = "success" if success else "failure" if failure else "incomplete"
        return {
            "steps": steps,
            "seed": int(file.attrs.get("seed", -1)),
            "behavior_id": behavior_id,
            "outcome": outcome,
        }


def _behavior_check(
    counts: Counter[str], *, episode_count: int, tolerance: float
) -> dict[str, Any]:
    expected = {
        behavior_id: weight / 100.0 for behavior_id, weight in BEHAVIOR_WEIGHTS
    }
    actual = {
        behavior_id: counts[behavior_id] / max(episode_count, 1)
        for behavior_id in expected
    }
    missing = [behavior_id for behavior_id in expected if counts[behavior_id] == 0]
    deviation = {
        behavior_id: abs(actual[behavior_id] - expected[behavior_id])
        for behavior_id in expected
    }
    passed = not missing and all(value <= tolerance for value in deviation.values())
    return _check(
        passed,
        counts=dict(sorted(counts.items())),
        expected_fraction=expected,
        actual_fraction=actual,
        absolute_deviation=deviation,
        tolerance=tolerance,
        missing=missing,
    )


def _split_check(
    paths: list[Path], episode_seeds: dict[Path, int], *, split_seed: int
) -> dict[str, Any]:
    if len(episode_seeds) != len(paths):
        return _check(False, reason="some episodes failed contract audit")
    split = split_episode_paths(paths, seed=split_seed)
    path_sets = {name: set(items) for name, items in split.items()}
    seed_sets = {
        name: {episode_seeds[path] for path in items} for name, items in split.items()
    }
    overlaps = {
        f"{left}-{right}": {
            "episodes": len(path_sets[left] & path_sets[right]),
            "seeds": len(seed_sets[left] & seed_sets[right]),
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    passed = all(
        values["episodes"] == 0 and values["seeds"] == 0
        for values in overlaps.values()
    )
    return _check(
        passed,
        partition_episodes={name: len(items) for name, items in split.items()},
        overlaps=overlaps,
    )


def _sequence_check(
    paths: list[Path], *, state_dim: int, action_dim: int
) -> dict[str, Any]:
    if not paths:
        return _check(False, reason="no episodes")
    sample_paths = paths[: min(2, len(paths))]
    try:
        dataset = ProprioSequenceDataset(
            paths=sample_paths,
            history_horizon=32,
            forecast_horizon=16,
            state_dim=state_dim,
            action_dim=action_dim,
            allow_legacy_wam=False,
            hdf5_cache_size=1,
        )
        first = dataset[0]
        first_ok = int(first["valid_mask"].sum()) == 1
        boundary_index = dataset.records[0].num_steps - 1
        boundary = dataset[boundary_index]
        boundary_ok = int(boundary["forecast_mask"].sum()) == 1
        second_ok = True
        if len(sample_paths) > 1:
            second = dataset[dataset.records[0].num_steps]
            second_ok = (
                int(second["episode_index"]) == dataset.records[1].episode_index
                and int(second["valid_mask"].sum()) == 1
            )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        return _check(False, reason=str(exc))
    finally:
        if "dataset" in locals():
            dataset.close()
    return _check(
        first_ok and boundary_ok and second_ok,
        left_padding=first_ok,
        right_padding=boundary_ok,
        episode_boundary=second_ok,
    )


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def _progress(total: int, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    if Progress is None or Console is None:
        print("Progress unavailable: install rich or pass --no-progress.", file=sys.stderr)
        return None
    return Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan"),
        MofNCompleteColumn(),
        console=Console(stderr=True),
        expand=True,
        refresh_per_second=10,
    )


if __name__ == "__main__":
    raise SystemExit(main())
