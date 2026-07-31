#!/usr/bin/env python3
"""Apply the special S2-R4 team/shared capability gate."""

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

from train.s2_model_registry import validate_s2_r4_candidate  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s2_r4.acceptance/1"
EVALUATION_FORMAT = "wam.robofactory.s2_r4.future_scope_evaluation/1"
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
    _validate_evaluation(p0, candidate_id="P0", team_shared=False)
    _validate_evaluation(p1, candidate_id="P1", team_shared=True)
    contract_equal = p0["comparison_contract"] == p1["comparison_contract"]
    p0_tasks = set(_mapping(p0, "per_task"))
    p1_tasks = set(_mapping(p1, "per_task"))
    task_set_passed = p0_tasks == p1_tasks == REQUIRED_TASKS
    per_task: dict[str, Any] = {}
    own_no_regression_all = True
    peer_shared_baseline_all = True
    peer_shuffle_all = True
    for task_id in sorted(REQUIRED_TASKS):
        if task_id not in p0_tasks or task_id not in p1_tasks:
            per_task[task_id] = {
                "available": False,
                "own_target_no_regression": False,
                "peer_shared_beats_persistence": False,
                "peer_shuffle_positive": False,
                "peer_shuffle_bootstrap_lower_positive": False,
            }
            own_no_regression_all = False
            peer_shared_baseline_all = False
            peer_shuffle_all = False
            continue
        control = _mapping(_mapping(p0, "per_task"), task_id)
        candidate = _mapping(_mapping(p1, "per_task"), task_id)
        p0_loss = float(control["normal_composite_future_loss"])
        own = _mapping(candidate, "own")
        peer_shared = _mapping(candidate, "peer_shared")
        p1_loss = float(own["normal_composite_future_loss"])
        margin = p0_loss - p1_loss
        no_regression = margin >= -1e-12
        peer_shared_loss = float(
            peer_shared["normal_composite_future_loss"]
        )
        persistence_loss = float(
            peer_shared["persistence_composite_future_loss"]
        )
        beats_persistence = peer_shared_loss < persistence_loss
        shuffle_delta = float(peer_shared["shuffle_delta"])
        lower = float(
            _mapping(peer_shared, "shuffle_delta_bootstrap_95")["lower"]
        )
        shuffle_positive = shuffle_delta > 0.0
        lower_positive = lower > 0.0
        own_no_regression_all = own_no_regression_all and no_regression
        peer_shared_baseline_all = (
            peer_shared_baseline_all and beats_persistence
        )
        peer_shuffle_all = (
            peer_shuffle_all and shuffle_positive and lower_positive
        )
        per_task[task_id] = {
            "available": True,
            "p0_normal_composite_future_loss": p0_loss,
            "p1_own_normal_composite_future_loss": p1_loss,
            "p0_minus_p1": margin,
            "own_target_no_regression": no_regression,
            "p1_peer_shared_composite_future_loss": peer_shared_loss,
            "persistence_composite_future_loss": persistence_loss,
            "persistence_minus_p1_peer_shared": (
                persistence_loss - peer_shared_loss
            ),
            "peer_shared_beats_persistence": beats_persistence,
            "p1_peer_action_shuffle_delta": shuffle_delta,
            "p1_peer_action_shuffle_bootstrap_95_lower": lower,
            "peer_shuffle_positive": shuffle_positive,
            "peer_shuffle_bootstrap_lower_positive": lower_positive,
        }
    equivalence = bool(
        _mapping(p0, "action_equivalence").get("passed")
        and _mapping(p1, "action_equivalence").get("passed")
    )
    frozen = bool(
        _mapping(p0, "frozen_parent").get("passed")
        and _mapping(p1, "frozen_parent").get("passed")
    )
    checks = {
        "same_comparison_contract": contract_equal,
        "exact_five_task_set": task_set_passed,
        "p1_own_target_no_worse_on_every_task": own_no_regression_all,
        "p1_peer_shared_beats_persistence_on_every_task": (
            peer_shared_baseline_all
        ),
        "p1_peer_action_shuffle_mean_and_ci_lower_positive_on_every_task": (
            peer_shuffle_all
        ),
        "predictor_disabled_f1_action_equivalence": equivalence,
        "flow_and_dinov3_frozen": frozen,
    }
    passed = all(checks.values())
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_id": "s2-r4",
        "passed": passed,
        "decision": "pass_enter_s3" if passed else "fail_keep_r3_local",
        "checks": checks,
        "per_task": per_task,
        "p0_evaluation": p0.get("checkpoint"),
        "p1_evaluation": p1.get("checkpoint"),
        "special_rule": (
            "S2-R4 requires non-regressing own targets, peer/shared loss below "
            "persistence, and positive paired peer-action-shuffle episode "
            "bootstrap lower bounds on all five tasks; closed-loop success is "
            "not an R4 selection signal."
        ),
    }


def _validate_evaluation(
    value: Mapping[str, Any],
    *,
    candidate_id: str,
    team_shared: bool,
) -> None:
    if value.get("format_version") != EVALUATION_FORMAT:
        raise ValueError("input is not an S2-R4 future-scope evaluation")
    observed_id, _, observed_team_shared = validate_s2_r4_candidate(value)
    if observed_id != candidate_id:
        raise ValueError(f"expected {candidate_id} evaluation")
    if observed_team_shared is not team_shared:
        raise ValueError(f"{candidate_id} future-scope identity drifted")


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
