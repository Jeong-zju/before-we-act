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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    paths = [
        root / "before_we_act/action_generator/base.py",
        root / args.candidate_module,
        root / "before_we_act/evaluate_action_generator.py",
    ]
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                findings.append({"path": str(path.relative_to(root)), "token": token})
    result = {
        "schema_version": 1,
        "round": "R12",
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
