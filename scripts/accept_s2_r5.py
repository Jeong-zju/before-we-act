#!/usr/bin/env python3
"""Apply the special S2-R5 protected-team gate and select a winner."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.s2_model_registry import validate_s2_r5_candidate  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s2_r5.acceptance/1"
EVALUATION_FORMAT = "wam.robofactory.s2_r5.protected_team_evaluation/1"
REQUIRED_TASKS = {
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p0 = _load(args.p0)
    p1 = _load(args.p1)
    payload = build_acceptance(p0, p1)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"acceptance": str(output), "passed": payload["passed"]},
            sort_keys=True,
        )
    )
    return 4 if args.strict and not payload["passed"] else 0


def build_acceptance(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_evaluation(p0, candidate_id="P0", team_mixer="shared")
    _validate_evaluation(p1, candidate_id="P1", team_mixer="role_mot")
    contract_equal = p0["comparison_contract"] == p1["comparison_contract"]
    p0_tasks = set(_mapping(p0, "per_task"))
    p1_tasks = set(_mapping(p1, "per_task"))
    task_set_passed = p0_tasks == p1_tasks == REQUIRED_TASKS
    candidate_results = {
        "P0": _candidate_gate(p0, task_set_passed),
        "P1": _candidate_gate(p1, task_set_passed),
    }
    checks = {
        "same_comparison_contract": contract_equal,
        "exact_five_task_set": task_set_passed,
        "at_least_one_candidate_passes_special_gate": any(
            value["passed"] for value in candidate_results.values()
        ),
    }
    passed = all(checks.values())
    eligible = [
        candidate_id
        for candidate_id, value in candidate_results.items()
        if value["passed"]
    ]
    winner = None
    if len(eligible) == 1:
        winner = eligible[0]
    elif len(eligible) == 2:
        p0_macro = float(candidate_results["P0"]["macro_peer_shared_loss"])
        p1_macro = float(candidate_results["P1"]["macro_peer_shared_loss"])
        winner = "P1" if p1_macro < p0_macro else "P0"
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_id": "s2-r5",
        "passed": passed,
        "winner": winner,
        "decision": (
            f"pass_select_{winner.lower()}_enter_s3"
            if winner is not None
            else "fail_stop_before_s3"
        ),
        "checks": checks,
        "candidates": candidate_results,
        "p0_evaluation": p0.get("checkpoint"),
        "p1_evaluation": p1.get("checkpoint"),
        "special_rule": (
            "Every eligible S2-R5 candidate must preserve exact protected-P0 "
            "own outputs and hashes, beat persistence for peer/shared loss, "
            "and have positive peer-action-shuffle mean and episode-bootstrap "
            "95% lower bound on all five tasks. If both pass, lower five-task "
            "macro peer/shared loss wins; an exact tie selects simpler P0."
        ),
    }


def _candidate_gate(
    value: Mapping[str, Any], task_set_passed: bool
) -> dict[str, Any]:
    protected = _mapping(value, "protected_own_evidence")
    own_exact = bool(
        protected.get("elementwise_exact")
        and float(protected.get("maximum_absolute_difference", 1.0)) == 0.0
        and protected.get("checkpoint_stable")
        and protected.get("optimizer_excluded")
    )
    action_equivalence = bool(_mapping(value, "action_equivalence").get("passed"))
    frozen = bool(_mapping(value, "frozen_parent").get("passed"))
    per_task: dict[str, Any] = {}
    baseline_all = task_set_passed
    shuffle_all = task_set_passed
    losses: list[float] = []
    tasks = _mapping(value, "per_task")
    for task_id in sorted(REQUIRED_TASKS):
        if task_id not in tasks:
            per_task[task_id] = {"available": False, "passed": False}
            baseline_all = False
            shuffle_all = False
            continue
        metrics = _mapping(tasks, task_id)
        peer_shared = _mapping(metrics, "peer_shared")
        normal = float(peer_shared["normal_composite_future_loss"])
        persistence = float(peer_shared["persistence_composite_future_loss"])
        delta = float(peer_shared["shuffle_delta"])
        lower = float(
            _mapping(peer_shared, "shuffle_delta_bootstrap_95")["lower"]
        )
        beats = normal < persistence
        shuffle = delta > 0.0 and lower > 0.0
        baseline_all = baseline_all and beats
        shuffle_all = shuffle_all and shuffle
        losses.append(normal)
        per_task[task_id] = {
            "available": True,
            "peer_shared_loss": normal,
            "persistence_loss": persistence,
            "beats_persistence": beats,
            "peer_action_shuffle_delta": delta,
            "peer_action_shuffle_bootstrap_95_lower": lower,
            "shuffle_mean_and_lower_positive": shuffle,
            "passed": beats and shuffle,
        }
    checks = {
        "protected_p0_own_elementwise_exact_and_hash_stable": own_exact,
        "peer_shared_beats_persistence_on_every_task": baseline_all,
        "peer_action_shuffle_mean_and_ci_lower_positive_on_every_task": shuffle_all,
        "predictor_disabled_f1_action_equivalence": action_equivalence,
        "flow_and_dinov3_frozen": frozen,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "macro_peer_shared_loss": sum(losses) / len(losses) if losses else None,
        "per_task": per_task,
    }


def _validate_evaluation(
    value: Mapping[str, Any],
    *,
    candidate_id: str,
    team_mixer: str,
) -> None:
    if value.get("format_version") != EVALUATION_FORMAT:
        raise ValueError("input is not an S2-R5 protected-team evaluation")
    observed_id, _, observed_mixer = validate_s2_r5_candidate(value)
    if observed_id != candidate_id:
        raise ValueError(f"expected {candidate_id} evaluation")
    if observed_mixer != team_mixer:
        raise ValueError(f"{candidate_id} team-mixer identity drifted")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be an object")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
