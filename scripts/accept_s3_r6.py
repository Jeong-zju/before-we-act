#!/usr/bin/env python3
"""Apply the S3 five-task macro-average closed-loop no-regression rule."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_lpd_gate_summary import FORMAT_VERSION as GATE_FORMAT  # noqa: E402
from scripts.train_s3_r6_world_action_flow import (  # noqa: E402
    CHECKPOINT_FORMAT,
)
from train.s3_model_registry import S3_R6_MODEL_KINDS  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s3_r6.acceptance/2"
TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair")
    pair.add_argument("--micro-round", choices=("R6L", "R6J"), required=True)
    pair.add_argument("--p0-gate", type=Path, required=True)
    pair.add_argument("--p1-gate", type=Path, required=True)
    pair.add_argument("--p0-checkpoint", type=Path, required=True)
    pair.add_argument("--p1-checkpoint", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--r6l", type=Path, required=True)
    final.add_argument("--r6j", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pair":
        payload = build_pair_acceptance(
            args.micro_round,
            _read_json(args.p0_gate),
            _read_json(args.p1_gate),
            _read_checkpoint(args.p0_checkpoint),
            _read_checkpoint(args.p1_checkpoint),
        )
    else:
        payload = build_final_acceptance(
            _read_json(args.r6l), _read_json(args.r6j)
        )
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_pair_acceptance(
    micro_round: str,
    p0_gate: Mapping[str, Any],
    p1_gate: Mapping[str, Any],
    p0_checkpoint: Mapping[str, Any],
    p1_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if micro_round not in {"R6L", "R6J"}:
        raise ValueError("micro_round must be R6L or R6J")
    for gate in (p0_gate, p1_gate):
        if gate.get("format_version") != GATE_FORMAT or gate.get("mode") != "gate":
            raise ValueError("S3 acceptance requires completed fixed-seed Gate summaries")
        if _mapping(gate, "candidate").get("policy_kind") != "s3_flow":
            raise ValueError("S3 acceptance rejects non-S3 policy kinds")
        if tuple(gate.get("task_order", ())) != TASKS:
            raise ValueError("S3 acceptance requires the exact five-task gate")
    expected_kinds = {
        "R6L": (
            "s3_r6l_protected_local_aux",
            "s3_r6l_protected_local_gated",
        ),
        "R6J": (
            "s3_r6j_protected_team_offpath",
            "s3_r6j_protected_team_gated",
        ),
    }[micro_round]
    for checkpoint, candidate_id, expected_kind in (
        (p0_checkpoint, "P0", expected_kinds[0]),
        (p1_checkpoint, "P1", expected_kinds[1]),
    ):
        method = _mapping(checkpoint, "method")
        if (
            checkpoint.get("format_version") != CHECKPOINT_FORMAT
            or method.get("micro_round") != micro_round
            or method.get("candidate_id") != candidate_id
            or method.get("model_kind") != expected_kind
            or expected_kind not in S3_R6_MODEL_KINDS
            or method.get("flow_training_scope")
            != "five_task_from_scratch_per_candidate"
        ):
            raise ValueError("checkpoint is outside the registered S3 pair")
        data = _mapping(checkpoint, "data")
        manifests = data.get("manifests")
        checkpoint_tasks = tuple(
            str(value.get("task_id"))
            for value in manifests
            if isinstance(value, Mapping)
        ) if isinstance(manifests, list) else ()
        if checkpoint_tasks != TASKS:
            raise ValueError("S3 checkpoint was not trained on the exact five tasks")
    if p0_gate.get("seed_protocol") != p1_gate.get("seed_protocol"):
        raise ValueError("P0/P1 closed-loop seed protocols differ")

    structural: dict[str, bool] = {}
    for candidate_id, checkpoint in (("P0", p0_checkpoint), ("P1", p1_checkpoint)):
        invariant = _mapping(checkpoint, "structural_invariants")
        structural[candidate_id] = all(
            invariant.get(name) is True
            for name in (
                "protected_own_elementwise_exact",
                "protected_parent_model_hashes_unchanged",
                "parent_files_unchanged",
                "parents_excluded_from_optimizer",
            )
        )
    p0_parent = _mapping(p0_checkpoint, "parent_identity")
    p1_parent = _mapping(p1_checkpoint, "parent_identity")
    structural["paired_five_task_flow_model_exact"] = (
        p0_parent.get("flow_model_sha256") == p1_parent.get("flow_model_sha256")
        and isinstance(p0_parent.get("flow_model_sha256"), str)
        and len(str(p0_parent.get("flow_model_sha256"))) == 64
    )
    tasks: dict[str, Any] = {}
    for task in TASKS:
        p0 = _mapping(p0_gate, task)
        p1 = _mapping(p1_gate, task)
        p0_episodes = p0.get("episodes")
        p1_episodes = p1.get("episodes")
        if not isinstance(p0_episodes, list) or not isinstance(p1_episodes, list):
            raise ValueError("gate summary lacks paired episode records")
        p0_seeds = [row.get("seed") for row in p0_episodes if isinstance(row, Mapping)]
        p1_seeds = [row.get("seed") for row in p1_episodes if isinstance(row, Mapping)]
        if p0_seeds != p1_seeds or len(p0_seeds) != len(p0_episodes):
            raise ValueError(f"{task} P0/P1 episode seeds are not paired")
        p0_successes = int(p0["successes"])
        p1_successes = int(p1["successes"])
        tasks[task] = {
            "p0_successes": p0_successes,
            "p1_successes": p1_successes,
            "episodes": len(p0_seeds),
            "p0_success_rate": float(p0["success_rate"]),
            "p1_success_rate": float(p1["success_rate"]),
            "delta_successes": p1_successes - p0_successes,
            "passed_no_regression": p1_successes >= p0_successes,
            "acceptance_required": False,
        }
    p0_macro = sum(row["p0_success_rate"] for row in tasks.values()) / len(TASKS)
    p1_macro = sum(row["p1_success_rate"] for row in tasks.values()) / len(TASKS)
    closed_loop_passed = p1_macro >= p0_macro
    passed = closed_loop_passed and all(structural.values())
    return {
        "format_version": FORMAT_VERSION,
        "scope": "pair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "micro_round": micro_round,
        "rule": (
            "P1 five-task macro-average success rate >= P0; ties pass; "
            "per-task success rates are report-only"
        ),
        "diagnostics_are_not_extra_gates": [
            "gate_zero_equivalence",
            "zero_or_noise_future",
            "future_shuffle",
            "fallback",
            "numerical_diagnostics",
        ],
        "structural_invariants": structural,
        "tasks": tasks,
        "per_task_success_rates_are_report_only": True,
        "macro_average": {
            "p0_success_rate": p0_macro,
            "p1_success_rate": p1_macro,
            "delta_success_rate": p1_macro - p0_macro,
            "passed_no_regression": closed_loop_passed,
        },
        "closed_loop_macro_average_passed": closed_loop_passed,
        "passed": passed,
        "decision": (
            f"pass_{micro_round.lower()}_p1"
            if passed
            else f"fail_{micro_round.lower()}_retain_p0"
        ),
    }


def build_final_acceptance(
    r6l: Mapping[str, Any], r6j: Mapping[str, Any]
) -> dict[str, Any]:
    for value, expected in ((r6l, "R6L"), (r6j, "R6J")):
        if (
            value.get("format_version") != FORMAT_VERSION
            or value.get("scope") != "pair"
            or value.get("micro_round") != expected
        ):
            raise ValueError("final acceptance requires the R6L and R6J pair reports")
    r6l_passed = r6l.get("passed") is True
    r6j_passed = r6j.get("passed") is True
    return {
        "format_version": FORMAT_VERSION,
        "scope": "final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "r6l": dict(r6l),
        "r6j": dict(r6j),
        "r6l_passed": r6l_passed,
        "r6j_passed": r6j_passed,
        "all_pairs_passed": r6l_passed and r6j_passed,
        "passed_for_r7": r6j_passed,
        "decision": (
            "pass_r6j_p1_enter_r7"
            if r6j_passed
            else "fail_r6j_retain_p0_stop_before_r7"
        ),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    value = torch.load(
        path.expanduser().resolve(strict=True), map_location="cpu", weights_only=False
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
