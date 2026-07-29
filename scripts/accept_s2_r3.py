#!/usr/bin/env python3
"""Apply the S2-R3 five-task loss, shuffle, and frozen-parent gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


FORMAT_VERSION = "wam.robofactory.s2_r3.acceptance/1"
EVALUATION_FORMAT = "wam.robofactory.s2_r3.action_shuffle_evaluation/1"
REQUIRED_TASKS = {
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w0", type=Path, required=True)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    w0 = _load(args.w0)
    w1 = _load(args.w1)
    payload = build_acceptance(w0, w1)
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
    w0: Mapping[str, Any],
    w1: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_evaluation(w0, candidate_id="W0", action_conditioning=False)
    _validate_evaluation(w1, candidate_id="W1", action_conditioning=True)
    contract_equal = w0["comparison_contract"] == w1["comparison_contract"]
    w0_tasks = set(_mapping(w0, "per_task"))
    w1_tasks = set(_mapping(w1, "per_task"))
    task_set_passed = w0_tasks == w1_tasks == REQUIRED_TASKS
    per_task: dict[str, Any] = {}
    no_regression_all = True
    any_strict_improvement = False
    shuffle_all = True
    for task_id in sorted(REQUIRED_TASKS):
        if task_id not in w0_tasks or task_id not in w1_tasks:
            per_task[task_id] = {
                "available": False,
                "no_future_loss_regression": False,
                "w1_shuffle_positive": False,
                "w1_shuffle_bootstrap_lower_positive": False,
            }
            no_regression_all = False
            shuffle_all = False
            continue
        control = _mapping(_mapping(w0, "per_task"), task_id)
        candidate = _mapping(_mapping(w1, "per_task"), task_id)
        w0_loss = float(control["normal_composite_future_loss"])
        w1_loss = float(candidate["normal_composite_future_loss"])
        margin = w0_loss - w1_loss
        no_regression = margin >= -1e-12
        strict_improvement = margin > 1e-12
        shuffle_delta = float(candidate["shuffle_delta"])
        lower = float(
            _mapping(candidate, "shuffle_delta_bootstrap_95")["lower"]
        )
        shuffle_positive = shuffle_delta > 0.0
        lower_positive = lower > 0.0
        no_regression_all = no_regression_all and no_regression
        any_strict_improvement = any_strict_improvement or strict_improvement
        shuffle_all = shuffle_all and shuffle_positive and lower_positive
        per_task[task_id] = {
            "available": True,
            "w0_normal_composite_future_loss": w0_loss,
            "w1_normal_composite_future_loss": w1_loss,
            "w0_minus_w1": margin,
            "no_future_loss_regression": no_regression,
            "strict_future_loss_improvement": strict_improvement,
            "w1_shuffle_delta": shuffle_delta,
            "w1_shuffle_bootstrap_95_lower": lower,
            "w1_shuffle_positive": shuffle_positive,
            "w1_shuffle_bootstrap_lower_positive": lower_positive,
        }
    equivalence = bool(
        _mapping(w0, "action_equivalence").get("passed")
        and _mapping(w1, "action_equivalence").get("passed")
    )
    frozen = bool(
        _mapping(w0, "frozen_parent").get("passed")
        and _mapping(w1, "frozen_parent").get("passed")
    )
    checks = {
        "same_comparison_contract": contract_equal,
        "exact_five_task_set": task_set_passed,
        "w1_no_worse_on_every_task": no_regression_all,
        "w1_strictly_better_on_at_least_one_task": any_strict_improvement,
        "w1_action_shuffle_mean_and_ci_lower_positive_on_every_task": shuffle_all,
        "predictor_disabled_f1_action_equivalence": equivalence,
        "flow_and_dinov3_frozen": frozen,
    }
    passed = all(checks.values())
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_id": "s2-r3",
        "passed": passed,
        "decision": "pass_enter_r4" if passed else "fail_stop_before_r4",
        "checks": checks,
        "per_task": per_task,
        "w0_evaluation": w0.get("checkpoint"),
        "w1_evaluation": w1.get("checkpoint"),
        "special_rule": (
            "S2-R3 is selected by held-out future loss plus paired own-action "
            "shuffle episode bootstrap, not by closed-loop success."
        ),
    }


def _validate_evaluation(
    value: Mapping[str, Any],
    *,
    candidate_id: str,
    action_conditioning: bool,
) -> None:
    if value.get("format_version") != EVALUATION_FORMAT:
        raise ValueError("input is not an S2-R3 action-shuffle evaluation")
    if value.get("candidate_id") != candidate_id:
        raise ValueError(f"expected {candidate_id} evaluation")
    if value.get("action_conditioning") is not action_conditioning:
        raise ValueError(f"{candidate_id} action-conditioning identity drifted")


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
