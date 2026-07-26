#!/usr/bin/env python3
"""Create the controller-bound 1/2/3/4-Panda action codec used by M2 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam.action_codec import AffineActionCodecConfig  # noqa: E402


PANDA_LOW = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0)
PANDA_HIGH = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite action codec {output}")
    agents = [f"panda-{index}" for index in range(args.agents)]
    config = AffineActionCodecConfig(
        codec_id=f"robofactory.{args.agents}x_panda_pd_joint_pos/1",
        low=PANDA_LOW * args.agents,
        high=PANDA_HIGH * args.agents,
        raw_domain="raw_pd_joint_pos_commanded",
        metadata={
            "agent_order": agents,
            "control_mode": "pd_joint_pos",
            "per_agent_action_order": [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "gripper_normalized_command",
            ],
            "provenance": {
                "arm_bounds": "ManiSkill Panda panda_v2.urdf joint limits; arm normalize_action=false",
                "gripper_bounds": "ManiSkill PDJointPosMimicController normalized controller input",
            },
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "semantic_sha256": config.sha256()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
