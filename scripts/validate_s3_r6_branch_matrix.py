#!/usr/bin/env python3
"""Fail closed unless four S3 configs isolate pair identity/scope/injection."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_static_rgb_act_moe import _load_yaml, _mapping  # noqa: E402
from train.s3_model_registry import validate_s3_r6_candidate  # noqa: E402


TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r6l_p0", "r6l_p1", "r6j_p0", "r6j_p1"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.r6l_p0, args.r6l_p1, args.r6j_p0, args.r6j_p1]
    configs = [_load_yaml(path.expanduser().resolve(strict=True)) for path in paths]
    observed = [validate_s3_r6_candidate(_mapping(value, "round")) for value in configs]
    expected = [
        ("R6L", "P0", "s3_r6l_protected_local_aux", "local", False),
        ("R6L", "P1", "s3_r6l_protected_local_gated", "local", True),
        ("R6J", "P0", "s3_r6j_protected_team_offpath", "team_shared", False),
        ("R6J", "P1", "s3_r6j_protected_team_gated", "team_shared", True),
    ]
    if observed != expected:
        raise ValueError(f"S3 branch ordering/identities differ: {observed!r}")
    for raw in configs:
        evaluation = _mapping(raw, "evaluation")
        if tuple(evaluation.get("tasks", ())) != TASKS:
            raise ValueError("S3 branch evaluation must cover the exact five tasks")
        if evaluation.get("rule") != "five_task_macro_average_p1_greater_or_equal_p0":
            raise ValueError("S3 branch evaluation must use the five-task macro rule")
    flow = _load_yaml(ROOT / "configs/wam_flow/s3_r6_flow_five_task.yaml")
    flow_manifests = tuple(
        Path(str(value)).parent.name for value in _mapping(flow, "data")["manifests"]
    )
    if (
        flow_manifests != TASKS
        or int(_mapping(flow, "training").get("updates", 0)) != 80000
        or _mapping(flow, "round").get("training_scope")
        != "five_task_from_scratch_per_candidate"
    ):
        raise ValueError("S3 fresh Flow config is not the frozen five-task/80k recipe")
    normalized = []
    for raw in configs:
        value = copy.deepcopy(raw)
        value["name"] = "paired"
        value["round"] = {
            "round_id": "s3-r6",
            "micro_round": "paired",
            "candidate_id": "paired",
            "model_kind": "paired",
            "future_scope": "paired",
            "injection": "paired",
        }
        for key in tuple(value["checkpoint"]):
            value["checkpoint"][key] = f"paired/{key}"
        normalized.append(value)
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError(
            "S3 branch configs drift beyond registered identity/future scope/injection/output"
        )
    print(
        "S3-R6 matrix valid: every candidate freshly trains the same 80k five-task "
        "Flow recipe and validates all five tasks; only R6L-vs-R6J scope and "
        "P0-vs-P1 injection differ."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
