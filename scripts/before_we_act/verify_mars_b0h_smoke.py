#!/usr/bin/env python3
"""Verify the MARS B0-H fresh/resume and benchmark-adapter contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.replace(temporary,path)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--resumed",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    left=torch.load(args.reference,map_location="cpu",weights_only=False)
    right=torch.load(args.resumed,map_location="cpu",weights_only=False)
    if left.get("update") != 4 or right.get("update") != 4:
        raise ValueError("MARS F1 checkpoints must end at update 4")
    if left["sample_cursor"] != right["sample_cursor"]:
        raise AssertionError("MARS F1 sample cursor drifted")
    maximum=0.0; differing=0
    for key,value in left["model"].items():
        other=right["model"][key]
        if not torch.equal(value,other):
            differing+=1
            maximum=max(maximum,float((value-other).abs().max()))
    if maximum > 1e-7:
        raise AssertionError(f"MARS resume differs: tensors={differing}, max={maximum}")
    config=right["config"]; stats=right["stats"]
    checks={
        "fresh_resume_max_abs_le_1e_7":maximum <= 1e-7,
        "all_600_policy_episodes":int(stats["episodes"]) == 600,
        "absolute_action":stats.get("action_encoding") == "absolute_pd_joint_pos",
        "native_320x240":config.get("image_width") == 320 and config.get("image_height") == 240,
        "frozen_dinov3":config.get("vision") == "dinov3_vitb16_frozen",
        "strict_local":config.get("strict_local") is True,
        "role_context":config.get("role_context") == "own_base_xy_in_task_context",
        "official_protocol_updates":config.get("protocol_updates") == 120000,
    }
    if not all(checks.values()): raise AssertionError(checks)
    atomic_json(args.output,{"status":"PASSED","checks":checks,
        "resume_model_max_abs":maximum,"resume_differing_tensor_count":differing})
    print("MARS_B0H_SMOKE_PASSED")


if __name__ == "__main__": main()
