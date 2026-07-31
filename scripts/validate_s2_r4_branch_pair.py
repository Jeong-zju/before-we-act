#!/usr/bin/env python3
"""Fail closed unless the R4 pair differs only in team/shared scope."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-config", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    p0 = _load(args.p0_config)
    p1 = _load(args.p1_config)
    _validate_identity(p0, candidate_id="P0", team_shared=False)
    _validate_identity(p1, candidate_id="P1", team_shared=True)
    normalized_p0 = _normalized(p0)
    normalized_p1 = _normalized(p1)
    if normalized_p0 != normalized_p1:
        raise ValueError(
            "S2-R4 branch configs differ outside team/shared scope and "
            "candidate-isolated output paths"
        )
    print("S2-R4 P0/P1 pair contract verified: team_shared is unique")
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
    team_shared: bool,
) -> None:
    round_config = value.get("round")
    if not isinstance(round_config, dict):
        raise ValueError("round config must be an object")
    expected_kind = (
        "s2_r4_team_shared_action_conditioned"
        if team_shared
        else "s2_r4_local_action_conditioned"
    )
    if (
        round_config.get("candidate_id") != candidate_id
        or round_config.get("model_kind") != expected_kind
        or round_config.get("action_conditioning") is not True
        or round_config.get("team_shared") is not team_shared
    ):
        raise ValueError(f"{candidate_id} branch identity is invalid")
    team_model = value.get("team_model")
    if team_shared and not isinstance(team_model, dict):
        raise ValueError("P1 must declare team_model")
    if not team_shared and team_model is not None:
        raise ValueError("P0 must not declare team_model")


def _normalized(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["name"] = "wam.robofactory/s2-r4-local-future"
    round_config = result["round"]
    round_config["candidate_id"] = "P"
    round_config["model_kind"] = "s2_r4_future_scope_x"
    round_config["team_shared"] = "UNIQUE_VARIABLE"
    result.pop("team_model", None)
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
