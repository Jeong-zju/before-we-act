#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from before_we_act.data.raw_team_windows import CachedTeamWindows  # noqa: E402
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_r11_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    first = PredictiveBeliefModel(config).to(device).eval()
    second = PredictiveBeliefModel(config).to(device).eval()
    first.load_state_dict(checkpoint["model"], strict=True)
    second.load_state_dict(checkpoint["model"], strict=True)
    data = CachedTeamWindows(args.cache, "validation")
    batch = {key: value[:2].to(device) for key, value in data.data.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        left, right = first(batch), second(batch)
    names = ("future_visual", "partner_action", "shared_progress")
    checks = {
        name: {
            "exact": bool(torch.equal(left[name], right[name])),
            "finite": bool(torch.isfinite(left[name]).all()),
            "max_abs": float((left[name].float() - right[name].float()).abs().max()),
        }
        for name in names
    }
    result = {
        "schema_version": 1,
        "round": "R11",
        "candidate_id": config.candidate_id,
        "checkpoint_update": checkpoint.get("update"),
        "strict_state_restore": True,
        "checks": checks,
    }
    result["passed"] = checkpoint.get("update") == 2 and all(row["exact"] and row["finite"] for row in checks.values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
