#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN = (
    "stereo_core",
    "NoWristPAIRRoute",
    "ARCADecoder",
    "forced_role",
    "role_prototypes",
    "RGBDPatchFusion",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--candidate-module", required=True)
    parser.add_argument(
        "--round", choices=("R12-R3", "R12-R4", "R12-E1"), default="R12-R3"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if args.round == "R12-E1":
        relative_paths = [
            "before_we_act/action_generator/r4_base.py",
            "before_we_act/action_generator/evolution.py",
            "before_we_act/action_generator/spatial_bridge.py",
            args.candidate_module,
            "before_we_act/spatial_observation.py",
            "before_we_act/train_action_generator_evolution.py",
            "before_we_act/evaluate_action_generator_evolution.py",
            "before_we_act/evaluate_action_generator_evolution_offline.py",
        ]
    elif args.round == "R12-R4":
        relative_paths = [
            "before_we_act/action_generator/r4_base.py",
            "before_we_act/action_generator/spatial_bridge.py",
            args.candidate_module,
            "before_we_act/spatial_observation.py",
            "before_we_act/train_action_generator_r4.py",
            "before_we_act/evaluate_action_generator_r4.py",
            "before_we_act/evaluate_action_generator_r4_offline.py",
        ]
    else:
        relative_paths = [
            "before_we_act/action_generator/base.py",
            args.candidate_module,
            "before_we_act/evaluate_action_generator.py",
        ]
    paths = [root / relative for relative in dict.fromkeys(relative_paths)]
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                findings.append({"path": str(path.relative_to(root)), "token": token})
    result = {
        "schema_version": 1,
        "round": args.round,
        "candidate_id": Path(args.candidate_module).stem,
        "audited_files": [str(path.relative_to(root)) for path in paths],
        "forbidden_runtime_references": findings,
        "core_directory_required": False,
        "passed": not findings,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
