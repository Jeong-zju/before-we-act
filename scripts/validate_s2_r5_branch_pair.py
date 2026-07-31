#!/usr/bin/env python3
"""Fail closed unless the two S2-R5 configs differ only by mixer identity."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_static_rgb_act_moe import _load_yaml, _mapping  # noqa: E402
from train.s2_model_registry import validate_s2_r5_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-config", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    args = parser.parse_args()
    p0 = _load_yaml(args.p0_config.expanduser().resolve(strict=True))
    p1 = _load_yaml(args.p1_config.expanduser().resolve(strict=True))
    if validate_s2_r5_candidate(_mapping(p0, "round"))[0] != "P0":
        raise ValueError("--p0-config is not registered P0")
    if validate_s2_r5_candidate(_mapping(p1, "round"))[0] != "P1":
        raise ValueError("--p1-config is not registered P1")
    left = copy.deepcopy(p0)
    right = copy.deepcopy(p1)
    for value in (left, right):
        value["name"] = "paired"
        round_config = value["round"]
        round_config["candidate_id"] = "paired"
        round_config["model_kind"] = "paired"
        round_config["team_mixer"] = "paired"
        value["team_model"]["team_mixer"] = "paired"
        checkpoint = value["checkpoint"]
        for key in tuple(checkpoint):
            checkpoint[key] = f"paired/{key}"
    if left != right:
        raise ValueError(
            "S2-R5 branch configs drift beyond candidate/mixer/output identity"
        )
    print(
        "S2-R5 pair valid: same data/seed/budget/validation; "
        "only shared vs role_mot differs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
