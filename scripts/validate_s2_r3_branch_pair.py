#!/usr/bin/env python3
"""Fail closed unless W0/W1 differ only in action-conditioning identity/paths."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w0-config", type=Path, required=True)
    parser.add_argument("--w1-config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    w0 = _load(args.w0_config)
    w1 = _load(args.w1_config)
    _validate_identity(w0, candidate_id="W0", conditioned=False)
    _validate_identity(w1, candidate_id="W1", conditioned=True)
    normalized_w0 = _normalized(w0)
    normalized_w1 = _normalized(w1)
    if normalized_w0 != normalized_w1:
        raise ValueError(
            "S2-R3 branch configs differ outside action conditioning and "
            "candidate-isolated output paths"
        )
    print("S2-R3 W0/W1 pair contract verified: action_conditioning is unique")
    return 0


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"config root must be an object: {path}")
    return value


def _validate_identity(
    value: dict[str, Any],
    *,
    candidate_id: str,
    conditioned: bool,
) -> None:
    round_config = value.get("round")
    if not isinstance(round_config, dict):
        raise ValueError("round config must be an object")
    expected_kind = (
        "s2_r3_local_action_conditioned"
        if conditioned
        else "s2_r3_local_action_independent"
    )
    if (
        round_config.get("candidate_id") != candidate_id
        or round_config.get("model_kind") != expected_kind
        or round_config.get("action_conditioning") is not conditioned
    ):
        raise ValueError(f"{candidate_id} branch identity is invalid")


def _normalized(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["name"] = "wam.robofactory/s2-r3-local-future"
    round_config = result["round"]
    round_config["candidate_id"] = "W"
    round_config["model_kind"] = "s2_r3_local_action_x"
    round_config["action_conditioning"] = "UNIQUE_VARIABLE"
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint config must be an object")
    for key in (
        "output",
        "resume",
        "progress_log",
        "evaluation",
        "evaluation_progress",
    ):
        if key not in checkpoint:
            raise ValueError(f"checkpoint.{key} is required")
        checkpoint[key] = f"<candidate-isolated>/{key}"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
