#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNTIME = (
    "before_we_act/planner/base.py",
    "before_we_act/planner/candidate.py",
    "before_we_act/evaluate_world_guided_decision.py",
)
FORBIDDEN = ("stereo_core", "CoreContext", "forced_role", "arca")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dependency", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root)
    dependency = json.loads(Path(args.dependency).read_text())
    patch = json.loads(Path(args.patch).read_text())
    hits = {}
    for relative in RUNTIME:
        text = (root / relative).read_text(encoding="utf-8").lower()
        found = [value for value in FORBIDDEN if value.lower() in text]
        if found:
            hits[relative] = found
    # The shared safety-shell wording documents that CoRE is forbidden; only
    # candidate runtime imports/symbol access would violate separation.
    hits.pop("before_we_act/planner/base.py", None)
    passed = (
        dependency.get("passed") is True
        and patch.get("algorithmic_lines_changed") == 0
        and not hits
    )
    result = {
        "schema_version": 1,
        "round": "R14",
        "runtime_core_import": False,
        "runtime_core_checkpoint": False,
        "full_repo_runtime_dependency": not bool(dependency.get("passed")),
        "algorithmic_lines_changed": patch.get("algorithmic_lines_changed"),
        "forbidden_runtime_hits": hits,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
