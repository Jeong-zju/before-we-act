#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    changed = subprocess.run(
        ("git", "diff", "--name-only", args.parent, args.head),
        check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    planner = any(path.startswith("before_we_act/planner/") for path in changed)
    evaluator = "before_we_act/evaluate_world_guided_decision.py" in changed
    forbidden = [path for path in changed if "stereo_core" in path.lower()]
    passed = planner and evaluator and not forbidden
    result = {
        "schema_version": 1,
        "round": "R14",
        "action_affecting": True,
        "planner_changes_final_action": True,
        "gate20_required": True,
        "runtime_core_forbidden": True,
        "forbidden_changed_paths": forbidden,
        "changed_files": changed,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
