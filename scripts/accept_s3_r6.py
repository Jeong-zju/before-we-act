#!/usr/bin/env python3
"""Apply the S3-only per-task closed-loop no-regression rule."""

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


FORMAT_VERSION = "wam.robofactory.s3_r6.acceptance/1"
TASKS = ("lift_barrier", "long_pipeline_delivery")


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
        ):
            raise ValueError("checkpoint is outside the registered S3 pair")
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
        }
    closed_loop_passed = all(
        row["passed_no_regression"] for row in tasks.values()
    )
    passed = closed_loop_passed and all(structural.values())
    return {
        "format_version": FORMAT_VERSION,
        "scope": "pair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "micro_round": micro_round,
        "rule": "P1 successes >= P0 successes independently on every task; ties pass",
        "diagnostics_are_not_extra_gates": [
            "gate_zero_equivalence",
            "zero_or_noise_future",
            "future_shuffle",
            "fallback",
            "numerical_diagnostics",
        ],
        "structural_invariants": structural,
        "tasks": tasks,
        "closed_loop_no_regression_passed": closed_loop_passed,
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
