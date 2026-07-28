#!/usr/bin/env python3
"""Validate and compare paired LPD fixed-seed gate artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.m1_statistics import exact_mcnemar, wilson_interval  # noqa: E402


FORMAT_VERSION = "wam.robofactory.lpd_experiment_comparison/1"
GATE_FORMATS = {
    "wam.robofactory.lpd_fixed_seed_gate/1",
    "wam.robofactory.lpd_fixed_seed_gate/2",
}
TASKS = ("lift_barrier", "long_pipeline_delivery")


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    gate: Mapping[str, Any]
    episodes: Mapping[str, tuple[tuple[int, bool], ...]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=GATE",
        help="repeat for every gate_summary.json or gate output directory",
    )
    parser.add_argument(
        "--baseline",
        help="candidate name used for paired deltas; defaults to the first candidate",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = tuple(_load_candidate(value) for value in args.candidate)
    _validate_names(candidates)
    baseline = args.baseline or candidates[0].name
    if baseline not in {candidate.name for candidate in candidates}:
        raise ValueError(f"baseline {baseline!r} is not a candidate")
    comparison = compare_candidates(candidates, baseline=baseline)
    markdown = render_markdown(comparison)
    _write(args.output_json, json.dumps(comparison, indent=2) + "\n", force=args.force)
    _write(args.output_markdown, markdown, force=args.force)
    return 0


def compare_candidates(
    candidates: Sequence[Candidate],
    *,
    baseline: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    by_name = {candidate.name: candidate for candidate in candidates}
    if baseline not in by_name:
        raise ValueError(f"unknown baseline {baseline!r}")
    reference = by_name[baseline]
    results: dict[str, Any] = {}
    for candidate in candidates:
        task_results: dict[str, Any] = {}
        for task in TASKS:
            candidate_rows = candidate.episodes[task]
            baseline_rows = reference.episodes[task]
            candidate_seeds = tuple(seed for seed, _ in candidate_rows)
            baseline_seeds = tuple(seed for seed, _ in baseline_rows)
            if candidate_seeds != baseline_seeds:
                raise ValueError(
                    f"{candidate.name}/{task} seed schedule differs from {baseline}"
                )
            candidate_success = tuple(success for _, success in candidate_rows)
            baseline_success = tuple(success for _, success in baseline_rows)
            successes = sum(candidate_success)
            interval = wilson_interval(successes, len(candidate_success))
            task_result: dict[str, Any] = {
                "seeds": list(candidate_seeds),
                "successes": successes,
                "episodes": len(candidate_success),
                "success_rate": interval["rate"],
                "success_rate_wilson_95": [
                    interval["lower"],
                    interval["upper"],
                ],
                "delta_vs_baseline": (
                    interval["rate"]
                    - sum(baseline_success) / len(baseline_success)
                ),
                "paired_mcnemar_vs_baseline": exact_mcnemar(
                    candidate_success,
                    baseline_success,
                ),
            }
            task_results[task] = task_result
        results[candidate.name] = {
            "gate": str(candidate.path),
            "mode": candidate.gate.get("mode"),
            "candidate_identity": candidate.gate.get("candidate"),
            "tasks": task_results,
        }
    return {
        "format_version": FORMAT_VERSION,
        "baseline": baseline,
        "pairing_key": ["task", "evaluation_seed"],
        "candidates": results,
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    baseline = str(comparison["baseline"])
    lines = [
        "# LPD fixed-seed experiment comparison",
        "",
        f"Paired baseline: `{baseline}`. Every delta uses identical task seeds.",
        "",
        "| Candidate | Task | Success | Rate | Wilson 95% | Δ vs baseline | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    candidates = _mapping(comparison, "candidates")
    for name, raw_candidate in candidates.items():
        candidate = _mapping(raw_candidate)
        tasks = _mapping(candidate, "tasks")
        for task in TASKS:
            result = _mapping(tasks, task)
            lower, upper = result["success_rate_wilson_95"]
            mcnemar = _mapping(result, "paired_mcnemar_vs_baseline")
            lines.append(
                "| "
                f"{name} | {task} | "
                f"{result['successes']}/{result['episodes']} | "
                f"{result['success_rate']:.1%} | "
                f"[{lower:.1%}, {upper:.1%}] | "
                f"{result['delta_vs_baseline']:+.1%} | "
                f"{mcnemar['p_value_two_sided']:.4g} |"
            )
    lines.append("")
    lines.append(
        "McNemar values are descriptive paired tests; retain Wilson intervals "
        "and training-seed replication for the primary claim."
    )
    lines.append("")
    return "\n".join(lines)


def _load_candidate(value: str) -> Candidate:
    if "=" not in value:
        raise ValueError("candidate must be NAME=GATE")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError("candidate must contain non-empty NAME and GATE")
    path = Path(raw_path).expanduser().resolve(strict=True)
    gate_path = path / "gate_summary.json" if path.is_dir() else path
    gate = _read_mapping(gate_path)
    if gate.get("format_version") not in GATE_FORMATS:
        raise ValueError(f"{gate_path} is not an LPD fixed-seed gate")
    episodes = {
        task: _load_task_episodes(gate, gate_path=gate_path, task=task)
        for task in TASKS
    }
    return Candidate(name=name, path=gate_path, gate=gate, episodes=episodes)


def _load_task_episodes(
    gate: Mapping[str, Any],
    *,
    gate_path: Path,
    task: str,
) -> tuple[tuple[int, bool], ...]:
    task_summary = _mapping(gate, task)
    raw_episodes = task_summary.get("episodes")
    if raw_episodes is None:
        rollout = _read_mapping(gate_path.parent / task / "rollout_summary.json")
        raw_episodes = rollout.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{gate_path}/{task} has no episode records")
    rows: list[tuple[int, bool]] = []
    for position, raw in enumerate(raw_episodes):
        episode = _mapping(raw)
        seed = episode.get("seed")
        success = episode.get("success")
        if not isinstance(seed, int) or not isinstance(success, bool):
            raise ValueError(f"{gate_path}/{task} episode {position} is invalid")
        rows.append((seed, success))
    if len({seed for seed, _ in rows}) != len(rows):
        raise ValueError(f"{gate_path}/{task} contains duplicate seeds")
    protocol = _mapping(gate, "seed_protocol")
    expected_count = int(protocol["episodes_per_task"])
    expected_start = int(protocol["seed_start"])
    expected_seeds = tuple(range(expected_start, expected_start + expected_count))
    if tuple(seed for seed, _ in rows) != expected_seeds:
        raise ValueError(f"{gate_path}/{task} violates its declared seed protocol")
    return tuple(rows)


def _validate_names(candidates: Sequence[Candidate]) -> None:
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    return _mapping(value)


def _mapping(value: Any, key: str | None = None) -> Mapping[str, Any]:
    result = value if key is None else value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key or 'value'} must be a mapping")
    return result


def _write(path: Path, content: str, *, force: bool) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
